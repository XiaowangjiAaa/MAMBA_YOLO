from .common_utils_mbyolo import *
from .block import Bottleneck, C3k
from .conv import Conv
import torch.nn.functional as F

__all__ = (
    "VSSBlock",
    "CrackVSSBlock",
    "CrackVSSBlockV2",
    "OrientationVSSBlock",
    "DiagonalOrientationVSSBlock",
    "CrackWriteVSSBlock",
    "CenteredCrackWriteVSSBlock",
    "LastCrackWriteVSSStage",
    "LastCenteredCrackWriteVSSStage",
    "CrackMemoryVSSBlock",
    "CrackStructureVSSBlock",
    "UnifiedCrackAwareVSSBlock",
    "LastUnifiedCrackAwareVSSStage",
    "EfficientCrackAlignedState",
    "AdaptiveC3k2CASP",
    "AdaptiveC2fCASP",
    "SparseCrackPathState",
    "AdaptiveC3k2CrackPath",
    "CrackDetailStemLite",
    "CrackDetailStemDirectional",
    "CrackMergeLite",
    "CrackMergeDirectional",
    "SimpleStem",
    "VisionClueMerge",
    "XSSBlock",
    "CrackXSSBlock",
    "CrackXSSBlockV2",
)


class SS2D(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            crack_guided_delta=False,
            crack_delta_centered=False,
            guidance_alpha_max=0.5,
            guidance_alpha_init=0.05,
            crack_guided_write=False,
            write_guidance_centered=False,
            write_beta_max=0.5,
            write_beta_init=0.0,
            orientation_scan=False,
            diagonal_scan=False,
            orientation_temperature=2.0,
            orientation_gate_max=1.0,
            orientation_gate_init=None,
            unified_crack_guidance=False,
            nonnegative_gates=False,
            unified_enable_write=True,
            unified_enable_scan=True,
            orientation_family_logits=False,
            crack_aligned_edges=False,
            edge_transition_init=0.05,
            edge_transition_max=0.5,
            edge_write_init=0.05,
            edge_write_max=0.25,
            edge_enable_transition=True,
            edge_enable_write=True,
            edge_enable_fusion=True,
            structure_kernel=1,
            structure_init_std=0.0,
            direction_mix=1.0,
            # ======================
            forward_type="v2",
            **kwargs,
    ):
        """
        ssm_rank_ratio would be used in the future...
        """
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_expand = int(ssm_ratio * d_model)
        d_inner = int(min(ssm_rank_ratio, ssm_ratio) * d_model) if ssm_rank_ratio > 0 else d_expand
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state  # 20240109
        self.d_conv = d_conv
        self.crack_aligned_edges = bool(crack_aligned_edges)
        if self.crack_aligned_edges:
            unified_crack_guidance = True
            orientation_scan = True
        if unified_crack_guidance:
            crack_guided_write = bool(unified_enable_write) and not self.crack_aligned_edges
            write_guidance_centered = bool(unified_enable_write)
            orientation_scan = bool(unified_enable_scan) or self.crack_aligned_edges
        self.scan_mode = "oriented_hvd" if diagonal_scan else ("oriented_hv" if orientation_scan else "cross")
        self.K = 8 if self.scan_mode == "oriented_hvd" else 4
        self.crack_guided_delta = crack_guided_delta
        self.crack_delta_centered = crack_delta_centered
        self.crack_guided_write = crack_guided_write
        self.write_guidance_centered = write_guidance_centered
        self.orientation_scan = orientation_scan or diagonal_scan
        self.orientation_temperature = orientation_temperature
        self.unified_crack_guidance = unified_crack_guidance
        self.nonnegative_gates = bool(nonnegative_gates)
        self.unified_enable_write = bool(unified_enable_write)
        self.unified_enable_scan = bool(unified_enable_scan)
        self.edge_enable_transition = bool(edge_enable_transition)
        self.edge_enable_write = bool(edge_enable_write)
        self.edge_enable_fusion = bool(edge_enable_fusion)
        self.structure_kernel = int(structure_kernel)
        self.structure_init_std = float(structure_init_std)
        self.direction_mix = float(direction_mix)
        if self.structure_kernel not in {1, 3, 5} or self.structure_kernel % 2 == 0:
            raise ValueError("structure_kernel must be one of {1, 3, 5}")
        if self.structure_init_std < 0.0:
            raise ValueError("structure_init_std must be nonnegative")
        if not 0.0 <= self.direction_mix <= 1.0:
            raise ValueError("direction_mix must be between 0 and 1")
        # Backward-compatible switch. Legacy oriented_hv accidentally projected the
        # two-channel head onto [+x, -x], leaving the second channel unused. New
        # experiments interpret the two channels directly as H/V family logits.
        self.orientation_family_logits = bool(orientation_family_logits)
        if self.orientation_family_logits and self.scan_mode != "oriented_hv":
            raise ValueError("orientation_family_logits currently requires oriented_hv scan mode")

        # tags for forward_type ==============================
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        self.disable_force32, forward_type = checkpostfix("no32", forward_type)
        self.disable_z, forward_type = checkpostfix("noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("nozact", forward_type)

        self.out_norm = nn.LayerNorm(d_inner)

        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            v2=partial(self.forward_corev2, force_fp32=None, SelectiveScan=SelectiveScanCore),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, FORWARD_TYPES.get("v2", None))

        # in proj =======================================
        d_proj = d_expand if self.disable_z else (d_expand * 2)
        self.in_proj = nn.Conv2d(d_model, d_proj, kernel_size=1, stride=1, groups=1, bias=bias, **factory_kwargs)
        self.act: nn.Module = nn.GELU()

        # conv =======================================
        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=d_expand,
                out_channels=d_expand,
                groups=d_expand,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # rank ratio =====================================
        self.ssm_low_rank = False
        if d_inner < d_expand:
            self.ssm_low_rank = True
            self.in_rank = nn.Conv2d(d_expand, d_inner, kernel_size=1, bias=False, **factory_kwargs)
            self.out_rank = nn.Linear(d_inner, d_expand, bias=False, **factory_kwargs)

        # The unified variant predicts probability and undirected orientation
        # from one structure head. Legacy ablations keep their separate heads.
        if self.unified_crack_guidance:
            # The head is zero-initialized, so preserve the RNG stream as well: with the
            # same seed, all following baseline SS2D parameters retain paired initialization.
            rng_state = torch.get_rng_state()
            if self.structure_kernel == 1:
                self.structure_head = nn.Conv2d(d_inner, 3, kernel_size=1, bias=True, **factory_kwargs)
                output_head = self.structure_head
            else:
                self.structure_head = nn.Sequential(
                    nn.Conv2d(
                        d_inner, d_inner, kernel_size=self.structure_kernel,
                        padding=self.structure_kernel // 2, groups=d_inner, bias=False, **factory_kwargs
                    ),
                    nn.GELU(),
                    nn.Conv2d(d_inner, 3, kernel_size=1, bias=True, **factory_kwargs),
                )
                output_head = self.structure_head[-1]
            if self.structure_init_std > 0.0:
                nn.init.normal_(output_head.weight, mean=0.0, std=self.structure_init_std)
            else:
                nn.init.zeros_(output_head.weight)
            nn.init.zeros_(output_head.bias)
            torch.set_rng_state(rng_state)
        elif self.crack_guided_delta or self.crack_guided_write:
            self.guidance = nn.Sequential(
                nn.Conv2d(d_inner, 1, kernel_size=1, bias=True, **factory_kwargs),
                nn.Sigmoid(),
            )
            nn.init.zeros_(self.guidance[0].weight)
            nn.init.zeros_(self.guidance[0].bias)

        if self.crack_guided_delta:
            if self.crack_delta_centered:
                if not 0.0 < guidance_alpha_init < guidance_alpha_max:
                    raise ValueError("guidance_alpha_init must be between 0 and guidance_alpha_max")
                alpha_ratio = guidance_alpha_init / guidance_alpha_max
                alpha_logit = math.log(alpha_ratio / (1.0 - alpha_ratio))
                self.guidance_alpha = nn.Parameter(torch.tensor(alpha_logit, **factory_kwargs))
                self.guidance_alpha_max = guidance_alpha_max
            else:
                self.guidance_alpha = nn.Parameter(torch.zeros((), **factory_kwargs))
        if self.crack_guided_write:
            if self.nonnegative_gates:
                if not 0.0 < write_beta_init < write_beta_max:
                    raise ValueError("nonnegative write_beta_init must be between 0 and write_beta_max")
                beta_ratio = write_beta_init / write_beta_max
                beta_raw = math.log(beta_ratio / (1.0 - beta_ratio))
            else:
                if not -write_beta_max < write_beta_init < write_beta_max:
                    raise ValueError("abs(write_beta_init) must be smaller than write_beta_max")
                beta_ratio = write_beta_init / write_beta_max
                beta_raw = math.atanh(beta_ratio)
            self.write_beta = nn.Parameter(torch.tensor(beta_raw, **factory_kwargs))
            self.write_beta_max = write_beta_max

        if self.orientation_scan:
            if not self.unified_crack_guidance:
                self.orientation_head = nn.Conv2d(d_inner, 2, kernel_size=1, bias=True, **factory_kwargs)
                nn.init.zeros_(self.orientation_head.weight)
                nn.init.zeros_(self.orientation_head.bias)
            basis = torch.tensor(
                [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], **factory_kwargs
            )
            self.register_buffer("orientation_basis", basis, persistent=False)
            # A disabled fusion branch must not leave behind a trainable gate.
            # Such a parameter is harmless in a single-process forward check but
            # is unused by autograd and therefore breaks DDP training when
            # find_unused_parameters=False (the W07 ablation).
            create_orientation_gate = orientation_gate_init is not None and (
                not self.crack_aligned_edges or self.edge_enable_fusion
            )
            if create_orientation_gate:
                if self.nonnegative_gates:
                    if not 0.0 < orientation_gate_init < orientation_gate_max:
                        raise ValueError(
                            "nonnegative orientation_gate_init must be between 0 and orientation_gate_max"
                        )
                    gate_ratio = orientation_gate_init / orientation_gate_max
                    gate_raw = math.log(gate_ratio / (1.0 - gate_ratio))
                else:
                    if not -orientation_gate_max < orientation_gate_init < orientation_gate_max:
                        raise ValueError("abs(orientation_gate_init) must be smaller than orientation_gate_max")
                    gate_ratio = orientation_gate_init / orientation_gate_max
                    gate_raw = math.atanh(gate_ratio)
                self.orientation_gate = nn.Parameter(torch.tensor(gate_raw, **factory_kwargs))
                self.orientation_gate._no_weight_decay = True
                self.orientation_gate_max = orientation_gate_max

        def bounded_logit(initial, maximum, name):
            if not 0.0 < initial < maximum:
                raise ValueError(f"{name} must be between 0 and {maximum}")
            ratio = initial / maximum
            return math.log(ratio / (1.0 - ratio))

        if self.crack_aligned_edges and self.edge_enable_transition:
            raw = bounded_logit(edge_transition_init, edge_transition_max, "edge_transition_init")
            self.edge_transition_raw = nn.Parameter(torch.tensor(raw, **factory_kwargs))
            self.edge_transition_raw._no_weight_decay = True
            self.edge_transition_max = float(edge_transition_max)
        if self.crack_aligned_edges and self.edge_enable_write:
            raw = bounded_logit(edge_write_init, edge_write_max, "edge_write_init")
            self.edge_write_raw = nn.Parameter(torch.tensor(raw, **factory_kwargs))
            self.edge_write_raw._no_weight_decay = True
            self.edge_write_max = float(edge_write_max)

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (self.dt_rank + self.d_state * 2), bias=False,
                      **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0).clone())  # (K, N, inner)
        del self.x_proj

        self.last_guidance = None
        self.last_orientation = None
        self.visual_guidance = None
        self.visual_orientation = None

        # out proj =======================================
        self.out_proj = nn.Conv2d(d_expand, d_model, kernel_size=1, stride=1, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        # simple init dt_projs, A_logs, Ds
        self.Ds = nn.Parameter(torch.ones((self.K * d_inner)))
        self.A_logs = nn.Parameter(
            torch.zeros((self.K * d_inner, self.d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
        self.dt_projs_weight = nn.Parameter(torch.randn((self.K, d_inner, self.dt_rank)))
        self.dt_projs_bias = nn.Parameter(torch.randn((self.K, d_inner)))

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('last_guidance', None)
        state.pop('last_orientation', None)
        state.pop('visual_guidance', None)
        state.pop('visual_orientation', None)
        return state

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def effective_write_gate(self):
        """Return the bounded write strength used by the state update."""
        if not self.crack_guided_write:
            return None
        transform = self.write_beta.sigmoid() if self.nonnegative_gates else self.write_beta.tanh()
        return self.write_beta_max * transform

    def effective_orientation_gate(self):
        """Return the bounded scan mixing strength used by direction-aware propagation."""
        if not hasattr(self, "orientation_gate"):
            return None
        transform = self.orientation_gate.sigmoid() if self.nonnegative_gates else self.orientation_gate.tanh()
        return self.orientation_gate_max * transform

    def effective_edge_transition(self):
        """Return the nonnegative strength for edge-conditioned state decay."""
        if not hasattr(self, "edge_transition_raw"):
            return None
        return self.edge_transition_max * self.edge_transition_raw.sigmoid()

    def effective_edge_write(self):
        """Return the nonnegative strength for edge-conditioned state writing."""
        if not hasattr(self, "edge_write_raw"):
            return None
        return self.edge_write_max * self.edge_write_raw.sigmoid()

    def crack_edge_confidence(self, probability: torch.Tensor, family_probability: torch.Tensor) -> torch.Tensor:
        """Build symmetric H/V crack-edge confidence from one shared structure field."""
        p_h = 0.5 * (
            F.pad(probability[..., :, :-1], (1, 0, 0, 0), mode="replicate")
            + F.pad(probability[..., :, 1:], (0, 1, 0, 0), mode="replicate")
        )
        p_v = 0.5 * (
            F.pad(probability[..., :-1, :], (0, 0, 1, 0), mode="replicate")
            + F.pad(probability[..., 1:, :], (0, 0, 0, 1), mode="replicate")
        )
        f_h = family_probability[:, :1]
        f_v = family_probability[:, 1:2]
        f_hn = 0.5 * (
            F.pad(f_h[..., :, :-1], (1, 0, 0, 0), mode="replicate")
            + F.pad(f_h[..., :, 1:], (0, 1, 0, 0), mode="replicate")
        )
        f_vn = 0.5 * (
            F.pad(f_v[..., :-1, :], (0, 0, 1, 0), mode="replicate")
            + F.pad(f_v[..., 1:, :], (0, 0, 0, 1), mode="replicate")
        )
        probability_pair = torch.sqrt(
            torch.clamp(probability * torch.cat((p_h, p_v), dim=1), min=1e-6)
        )
        direction_pair = torch.sqrt(
            torch.clamp(family_probability * torch.cat((f_hn, f_vn), dim=1), min=1e-6)
        )
        # Keep probability as the stable base signal and let direction refine it.
        # direction_mix=1 reproduces the 8.26 product; smaller values are more
        # tolerant of uncertain orientation on curved and branching cracks.
        direction_factor = (1.0 - self.direction_mix) + self.direction_mix * direction_pair
        return (probability_pair * direction_factor).clamp_(0.0, 1.0)

    def orientation_scores(self, orientation: torch.Tensor) -> torch.Tensor:
        """Map the structure-head output to scan-family scores.

        The corrected H/V path uses both channels directly. The legacy projection
        remains available so existing YAMLs and checkpoints keep their old behavior.
        """
        if self.orientation_family_logits:
            if orientation.shape[1] != 2:
                raise ValueError(f"H/V family logits require 2 channels, got {orientation.shape[1]}")
            return orientation
        family_count = 4 if self.scan_mode == "oriented_hvd" else 2
        basis = self.orientation_basis[:family_count].to(dtype=orientation.dtype)
        return torch.einsum("bchw,fc->bfhw", orientation, basis)

    def forward_corev2(self, x: torch.Tensor, channel_first=False, SelectiveScan=SelectiveScanCore,
                       cross_selective_scan=cross_selective_scan, force_fp32=None):
        force_fp32 = (self.training and (not self.disable_force32)) if force_fp32 is None else force_fp32
        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.ssm_low_rank:
            x = self.in_rank(x)
        has_probability = self.crack_guided_delta or self.crack_guided_write
        if self.unified_crack_guidance:
            structure = self.structure_head(x)
            delta_guidance = structure[:, :1].sigmoid()
            orientation = structure[:, 1:].tanh()
            # Retain all shared-head channels in the autograd graph for write-only/scan-only
            # DDP ablations, while contributing exactly zero to the enabled path.
            if self.unified_enable_write and not self.unified_enable_scan:
                delta_guidance = delta_guidance + 0.0 * orientation.mean(dim=1, keepdim=True)
            elif self.unified_enable_scan and not self.unified_enable_write:
                orientation = orientation + 0.0 * delta_guidance
        else:
            delta_guidance = self.guidance(x) if has_probability else None
            orientation = torch.tanh(self.orientation_head(x)) if self.orientation_scan else None

        # Loss consumes the live tensors. Detached copies are visualization-only.
        self.last_guidance = delta_guidance
        self.visual_guidance = delta_guidance.detach() if delta_guidance is not None else None
        if self.crack_guided_delta and self.crack_delta_centered:
            delta_alpha = self.guidance_alpha_max * self.guidance_alpha.sigmoid()
        else:
            delta_alpha = self.guidance_alpha if self.crack_guided_delta else None
        write_beta = self.effective_write_gate()

        transition_guidance = None
        transition_alpha = None
        edge_write_guidance = None
        edge_write_beta = None
        if self.crack_aligned_edges:
            direction_scores = self.orientation_scores(orientation)
            family_probability = torch.softmax(self.orientation_temperature * direction_scores, dim=1)
            edge_confidence = self.crack_edge_confidence(delta_guidance, family_probability)
            self.last_edge_confidence = edge_confidence
            self.visual_edge_confidence = edge_confidence.detach()
            if self.edge_enable_fusion:
                adaptive_weights = edge_confidence.shape[1] * edge_confidence / (
                    edge_confidence.sum(dim=1, keepdim=True) + 1e-6
                )
                if hasattr(self, "orientation_gate"):
                    scan_gate = self.effective_orientation_gate()
                    direction_weights = 1.0 + scan_gate * (adaptive_weights - 1.0)
                else:
                    direction_weights = adaptive_weights
            else:
                direction_weights = None
            if self.edge_enable_transition:
                transition_guidance = edge_confidence
                transition_alpha = self.effective_edge_transition()
            if self.edge_enable_write:
                edge_write_guidance = 2.0 * edge_confidence - 1.0
                edge_write_beta = self.effective_edge_write()
            self.last_orientation = orientation
            self.visual_orientation = orientation.detach()
        elif self.orientation_scan:
            self.last_edge_confidence = None
            self.visual_edge_confidence = None
            self.last_orientation = orientation
            self.visual_orientation = orientation.detach()
            direction_scores = self.orientation_scores(orientation)
            family_count = direction_scores.shape[1]
            adaptive_weights = family_count * torch.softmax(
                self.orientation_temperature * direction_scores, dim=1
            )
            if hasattr(self, "orientation_gate"):
                scan_gate = self.effective_orientation_gate()
                direction_weights = 1.0 + scan_gate * (adaptive_weights - 1.0)
            else:
                direction_weights = adaptive_weights
        else:
            self.last_orientation = None
            self.visual_orientation = None
            direction_weights = None
            self.last_edge_confidence = None
            self.visual_edge_confidence = None
        x = cross_selective_scan(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds,
            out_norm=getattr(self, "out_norm", None),
            out_norm_shape=getattr(self, "out_norm_shape", "v0"),
            delta_softplus=True, force_fp32=force_fp32,
            SelectiveScan=SelectiveScan, ssoflex=self.training,  # output fp32
            delta_guidance=delta_guidance if self.crack_guided_delta else None,
            delta_alpha=delta_alpha,
            delta_guidance_center=0.5 if self.crack_delta_centered else 1.0,
            write_guidance=edge_write_guidance if self.crack_aligned_edges else ((2.0 * delta_guidance - 1.0) if (
                self.crack_guided_write and self.write_guidance_centered
            ) else (delta_guidance if self.crack_guided_write else None)),
            write_beta=edge_write_beta if self.crack_aligned_edges else write_beta,
            transition_guidance=transition_guidance,
            transition_alpha=transition_alpha,
            scan_mode=self.scan_mode,
            direction_weights=direction_weights,
        )
        if self.ssm_low_rank:
            x = self.out_rank(x)
        return x

    def forward(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=1)  # (b, d, h, w)
            if not self.disable_z_act:
                z1 = self.act(z)
        if self.d_conv > 0:
            x = self.conv2d(x)  # (b, d, h, w)
        x = self.act(x)
        y = self.forward_core(x, channel_first=(self.d_conv > 1))
        y = y.permute(0, 3, 1, 2).contiguous()
        if not self.disable_z:
            y = y * z1
        out = self.dropout(self.out_proj(y))
        return out


class RGBlock(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, kernel_size=1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=True,
                                groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x, v = self.fc1(x).chunk(2, dim=1)
        x = self.act(self.dwconv(x) + x) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LSBlock(nn.Module):
    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU, drop=0):
        super().__init__()
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=3, padding=3 // 2, groups=hidden_features)
        self.norm = nn.BatchNorm2d(hidden_features)
        self.fc2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=1, padding=0)
        self.act = act_layer()
        self.fc3 = nn.Conv2d(hidden_features, in_features, kernel_size=1, padding=0)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        input = x
        x = self.fc1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        x = input + self.drop(x)
        return x


class XSSBlock(nn.Module):
    def __init__(
            self,
            in_channels: int = 0,
            hidden_dim: int = 0,
            n: int = 1,
            mlp_ratio=4.0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(LayerNorm2d, eps=1e-6),
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2",
            # =============================
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            crack_guided_delta: bool = False,
            crack_delta_centered: bool = False,
            crack_guided_write: bool = False,
            write_guidance_centered: bool = False,
            write_beta_init: float = 0.0,
            write_beta_max: float = 0.5,
            orientation_scan: bool = False,
            diagonal_scan: bool = False,
            orientation_gate_init=None,
            orientation_gate_max: float = 1.0,
            orientation_temperature: float = 2.0,
            unified_crack_guidance: bool = False,
            nonnegative_gates: bool = False,
            unified_enable_write: bool = True,
            unified_enable_scan: bool = True,
            orientation_family_logits: bool = False,
            **kwargs,
    ):
        super().__init__()

        self.in_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU()
        ) if in_channels != hidden_dim else nn.Identity()
        self.hidden_dim = hidden_dim
        # ==========SSM============================
        self.norm = norm_layer(hidden_dim)
        self.ss2d = nn.Sequential(*(SS2D(d_model=self.hidden_dim,
                                         d_state=ssm_d_state,
                                         ssm_ratio=ssm_ratio,
                                         ssm_rank_ratio=ssm_rank_ratio,
                                         dt_rank=ssm_dt_rank,
                                         act_layer=ssm_act_layer,
                                         d_conv=ssm_conv,
                                         conv_bias=ssm_conv_bias,
                                         dropout=ssm_drop_rate,
                                         crack_guided_delta=crack_guided_delta,
                                         crack_delta_centered=crack_delta_centered,
                                         crack_guided_write=crack_guided_write,
                                         write_guidance_centered=write_guidance_centered,
                                         write_beta_init=write_beta_init,
                                         write_beta_max=write_beta_max,
                                         orientation_scan=orientation_scan,
                                         diagonal_scan=diagonal_scan,
                                         orientation_gate_init=orientation_gate_init,
                                         orientation_gate_max=orientation_gate_max,
                                         orientation_temperature=orientation_temperature,
                                         unified_crack_guidance=unified_crack_guidance,
                                          nonnegative_gates=nonnegative_gates,
                                          unified_enable_write=unified_enable_write,
                                          unified_enable_scan=unified_enable_scan,
                                          orientation_family_logits=orientation_family_logits, ) for _ in range(n)))
        self.drop_path = DropPath(drop_path)
        self.lsblock = LSBlock(hidden_dim, hidden_dim)
        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = RGBlock(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                               drop=mlp_drop_rate)

    def forward(self, input):
        input = self.in_proj(input)
        # ====================
        X1 = self.lsblock(input)
        input = input + self.drop_path(self.ss2d(self.norm(X1)))
        # ===================
        if self.mlp_branch:
            input = input + self.drop_path(self.mlp(self.norm2(input)))
        return input


class VSSBlock(nn.Module):
    def __init__(
            self,
            in_channels: int = 0,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(LayerNorm2d, eps=1e-6),
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2",
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            crack_guided_delta: bool = False,
            crack_delta_centered: bool = False,
            crack_guided_write: bool = False,
            write_guidance_centered: bool = False,
            write_beta_init: float = 0.0,
            write_beta_max: float = 0.5,
            orientation_scan: bool = False,
            diagonal_scan: bool = False,
            orientation_gate_init=None,
            orientation_gate_max: float = 1.0,
            orientation_temperature: float = 2.0,
            unified_crack_guidance: bool = False,
            nonnegative_gates: bool = False,
            unified_enable_write: bool = True,
            unified_enable_scan: bool = True,
            orientation_family_logits: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        # proj
        self.proj_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU()
        )

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = SS2D(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_rank_ratio=ssm_rank_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                crack_guided_delta=crack_guided_delta,
                crack_delta_centered=crack_delta_centered,
                crack_guided_write=crack_guided_write,
                write_guidance_centered=write_guidance_centered,
                write_beta_init=write_beta_init,
                write_beta_max=write_beta_max,
                orientation_scan=orientation_scan,
                diagonal_scan=diagonal_scan,
                orientation_gate_init=orientation_gate_init,
                orientation_gate_max=orientation_gate_max,
                orientation_temperature=orientation_temperature,
                unified_crack_guidance=unified_crack_guidance,
                nonnegative_gates=nonnegative_gates,
                unified_enable_write=unified_enable_write,
                unified_enable_scan=unified_enable_scan,
                orientation_family_logits=orientation_family_logits,
                # bias=False,
                # ==========================
                # dt_min=0.001,
                # dt_max=0.1,
                # dt_init="random",
                # dt_scale="random",
                # dt_init_floor=1e-4,
                initialize=ssm_init,
                # ==========================
                forward_type=forward_type,
            )

        self.drop_path = DropPath(drop_path)
        self.lsblock = LSBlock(hidden_dim, hidden_dim)
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = RGBlock(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                               drop=mlp_drop_rate, channels_first=False)

    def forward(self, input: torch.Tensor):
        input = self.proj_conv(input)
        X1 = self.lsblock(input)
        x = input + self.drop_path(self.op(self.norm(X1)))
        if self.mlp_branch:
            x = x + self.drop_path(self.mlp(self.norm2(x)))  # FFN
        return x


class CrackVSSBlock(VSSBlock):
    """VSSBlock with crack-guided pre-softplus delta modulation enabled."""

    def __init__(self, in_channels=0, hidden_dim=0, drop_path=0, **kwargs):
        super().__init__(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            drop_path=drop_path,
            crack_guided_delta=True,
            **kwargs,
        )


class CrackXSSBlock(XSSBlock):
    """XSSBlock with crack-guided pre-softplus delta modulation enabled."""

    def __init__(self, in_channels=0, hidden_dim=0, n=1, **kwargs):
        super().__init__(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            n=n,
            crack_guided_delta=True,
            **kwargs,
        )


class CrackVSSBlockV2(VSSBlock):
    """VSSBlock with bounded, centered crack-guided delta modulation."""

    def __init__(self, in_channels=0, hidden_dim=0, drop_path=0, **kwargs):
        super().__init__(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            drop_path=drop_path,
            crack_guided_delta=True,
            crack_delta_centered=True,
            **kwargs,
        )


class CrackXSSBlockV2(XSSBlock):
    """XSSBlock with bounded, centered crack-guided delta modulation."""

    def __init__(self, in_channels=0, hidden_dim=0, n=1, **kwargs):
        super().__init__(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            n=n,
            crack_guided_delta=True,
            crack_delta_centered=True,
            **kwargs,
        )


class OrientationVSSBlock(VSSBlock):
    """VSSBlock with content-adaptive horizontal/vertical scan fusion."""

    def __init__(self, in_channels=0, hidden_dim=0, drop_path=0, **kwargs):
        super().__init__(in_channels, hidden_dim, drop_path, orientation_scan=True, **kwargs)


class DiagonalOrientationVSSBlock(VSSBlock):
    """VSSBlock with horizontal, vertical, and two diagonal scan families."""

    def __init__(self, in_channels=0, hidden_dim=0, drop_path=0, **kwargs):
        super().__init__(in_channels, hidden_dim, drop_path, orientation_scan=True, diagonal_scan=True, **kwargs)


class CrackWriteVSSBlock(VSSBlock):
    """VSSBlock that preserves delta and strengthens crack-probable input writes through B."""

    def __init__(self, in_channels=0, hidden_dim=0, drop_path=0, **kwargs):
        super().__init__(in_channels, hidden_dim, drop_path, crack_guided_write=True, **kwargs)


class CenteredCrackWriteVSSBlock(VSSBlock):
    """VSSBlock using B * (1 + beta * (2g - 1)) to remove constant-gate scaling."""

    def __init__(self, in_channels=0, hidden_dim=0, drop_path=0, **kwargs):
        super().__init__(
            in_channels,
            hidden_dim,
            drop_path,
            crack_guided_write=True,
            write_guidance_centered=True,
            write_beta_init=0.05,
            **kwargs,
        )


class _LastCrackWriteVSSStage(nn.Sequential):
    """Reproduce a repeated VSS stage while modifying only its final block."""

    final_block = CrackWriteVSSBlock

    def __init__(self, in_channels=0, hidden_dim=0, n=1, drop_path=0, **kwargs):
        blocks = []
        for index in range(n):
            block_type = self.final_block if index == n - 1 else VSSBlock
            blocks.append(
                block_type(
                    in_channels if index == 0 else hidden_dim,
                    hidden_dim,
                    drop_path,
                    **kwargs,
                )
            )
        super().__init__(*blocks)


class LastCrackWriteVSSStage(_LastCrackWriteVSSStage):
    """Repeated VSS stage with the original probability write gate only in the last block."""


class LastCenteredCrackWriteVSSStage(_LastCrackWriteVSSStage):
    """Repeated VSS stage with a centered write gate only in the last block."""

    final_block = CenteredCrackWriteVSSBlock


class CrackMemoryVSSBlock(VSSBlock):
    """VSSBlock combining bounded centered delta retention with probability-guided input writes."""

    def __init__(self, in_channels=0, hidden_dim=0, drop_path=0, **kwargs):
        super().__init__(
            in_channels,
            hidden_dim,
            drop_path,
            crack_guided_delta=True,
            crack_delta_centered=True,
            crack_guided_write=True,
            **kwargs,
        )


class CrackStructureVSSBlock(VSSBlock):
    """Joint H/V orientation-aware scan and probability-guided memory block."""

    def __init__(self, in_channels=0, hidden_dim=0, drop_path=0, **kwargs):
        super().__init__(
            in_channels,
            hidden_dim,
            drop_path,
            crack_guided_write=True,
            orientation_scan=True,
            **kwargs,
        )


class UnifiedCrackAwareVSSBlock(VSSBlock):
    """One crack-aware block with a shared probability/orientation head and learnable effect gates."""

    def __init__(self, in_channels=0, hidden_dim=0, drop_path=0, write_gate_init=0.05,
                 scan_gate_init=0.05, write_gate_max=0.5, scan_gate_max=1.0,
                 orientation_temperature=2.0, nonnegative_gates=False,
                 enable_write=True, enable_scan=True, orientation_family_logits=False, **kwargs):
        super().__init__(
            in_channels,
            hidden_dim,
            drop_path,
            crack_guided_write=True,
            write_guidance_centered=True,
            write_beta_init=write_gate_init,
            write_beta_max=write_gate_max,
            orientation_scan=True,
            orientation_gate_init=scan_gate_init,
            orientation_gate_max=scan_gate_max,
            orientation_temperature=orientation_temperature,
            unified_crack_guidance=True,
            nonnegative_gates=nonnegative_gates,
            unified_enable_write=enable_write,
            unified_enable_scan=enable_scan,
            orientation_family_logits=orientation_family_logits,
            **kwargs,
        )


class LastUnifiedCrackAwareVSSStage(nn.Sequential):
    """Repeated VSS stage whose final block is the fixed unified crack-aware block."""

    def __init__(self, in_channels=0, hidden_dim=0, n=1, write_gate_init=0.05,
                 scan_gate_init=0.05, write_gate_max=0.5, scan_gate_max=1.0,
                 orientation_temperature=2.0, nonnegative_gates=False,
                 enable_write=True, enable_scan=True, orientation_family_logits=False, **kwargs):
        blocks = []
        for index in range(n):
            input_dim = in_channels if index == 0 else hidden_dim
            if index == n - 1:
                block = UnifiedCrackAwareVSSBlock(
                    input_dim, hidden_dim, 0, write_gate_init, scan_gate_init,
                    write_gate_max, scan_gate_max, orientation_temperature,
                    nonnegative_gates, enable_write, enable_scan, orientation_family_logits, **kwargs
                )
            else:
                block = VSSBlock(input_dim, hidden_dim, **kwargs)
            blocks.append(block)
        super().__init__(*blocks)


class EfficientCrackAlignedState(nn.Module):
    """Partial-channel crack-aligned SSM with a bounded residual router.

    One shared crack edge controls directional fusion, state decay and state
    writing inside SS2D. Only ``state_ratio`` channels enter the SSM, keeping
    the block practical at high-resolution YOLO stages.
    """

    def __init__(self, channels, state_ratio=0.25, route_init=0.01, route_max=0.5,
                 d_state=8, ssm_ratio=1.0, edge_transition_init=0.05,
                 edge_transition_max=0.5, edge_write_init=0.05,
                 edge_write_max=0.25, scan_gate_init=0.05, scan_gate_max=0.5,
                 orientation_temperature=1.0, enable_transition=True,
                 enable_write=True, enable_fusion=True, structure_kernel=1,
                 structure_init_std=0.0, direction_mix=1.0):
        super().__init__()
        self.channels = int(channels)
        self.state_ratio = float(state_ratio)
        state_channels = max(8, int(round(channels * state_ratio / 8.0)) * 8)
        self.state_channels = min(channels, state_channels)
        if not 0.0 < route_init < route_max:
            raise ValueError("route_init must be between 0 and route_max")
        route_ratio = route_init / route_max
        self.route_raw = nn.Parameter(torch.tensor(math.log(route_ratio / (1.0 - route_ratio))))
        self.route_raw._no_weight_decay = True
        self.route_max = float(route_max)
        self.norm = LayerNorm2d(self.state_channels)
        self.state = SS2D(
            d_model=self.state_channels,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            ssm_rank_ratio=ssm_ratio,
            d_conv=3,
            dropout=0.0,
            orientation_scan=True,
            orientation_gate_init=scan_gate_init if enable_fusion else None,
            orientation_gate_max=scan_gate_max,
            orientation_temperature=orientation_temperature,
            orientation_family_logits=True,
            nonnegative_gates=True,
            crack_aligned_edges=True,
            edge_transition_init=edge_transition_init,
            edge_transition_max=edge_transition_max,
            edge_write_init=edge_write_init,
            edge_write_max=edge_write_max,
            edge_enable_transition=enable_transition,
            edge_enable_write=enable_write,
            edge_enable_fusion=enable_fusion,
            structure_kernel=structure_kernel,
            structure_init_std=structure_init_std,
            direction_mix=direction_mix,
            forward_type="v2noz",
        )

    def effective_route(self):
        return self.route_max * self.route_raw.sigmoid()

    def forward(self, x):
        state_input = x[:, :self.state_channels]
        state_delta = self.state(self.norm(state_input))
        # The structure probability supplies sample/spatial adaptation, while
        # the bounded scalar lets each YOLO stage learn how much SSM it needs.
        probability = self.state.last_guidance
        spatial_route = 0.5 + probability
        updated = state_input + self.effective_route() * spatial_route * state_delta
        return torch.cat((updated, x[:, self.state_channels:]), dim=1)


class _AdaptiveCASPUnit(nn.Module):
    """Local YOLO bottleneck followed by an efficient residual CASP branch."""

    def __init__(self, channels, c3k=False, shortcut=True, state_ratio=0.25, route_init=0.01,
                 route_max=0.5, d_state=8, ssm_ratio=1.0,
                 edge_transition_init=0.05, edge_transition_max=0.5,
                 edge_write_init=0.05, edge_write_max=0.25,
                 scan_gate_init=0.05, scan_gate_max=0.5,
                 orientation_temperature=1.0, enable_transition=True,
                 enable_write=True, enable_fusion=True, structure_kernel=1,
                 structure_init_std=0.0, direction_mix=1.0):
        super().__init__()
        self.local = C3k(channels, channels, 2, shortcut) if c3k else Bottleneck(
            channels, channels, shortcut, 1, k=((3, 3), (3, 3)), e=1.0
        )
        self.casp = EfficientCrackAlignedState(
            channels, state_ratio, route_init, route_max, d_state, ssm_ratio,
            edge_transition_init, edge_transition_max, edge_write_init,
            edge_write_max, scan_gate_init, scan_gate_max,
            orientation_temperature, enable_transition, enable_write, enable_fusion,
            structure_kernel, structure_init_std, direction_mix
        )

    def forward(self, x):
        return self.casp(self.local(x))


class AdaptiveC3k2CASP(nn.Module):
    """Drop-in, efficient YOLO11 C3k2 replacement with one adaptive CASP unit.

    The CSP topology and local operations remain available at every stage. Only
    the last internal repeat contains a partial-channel state branch, so the same
    module can be deployed throughout the backbone and neck without forcing
    shallow features to rely on global propagation.
    """

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, state_ratio=0.25,
                 route_init=0.01, route_max=0.5, d_state=8, ssm_ratio=1.0,
                 edge_transition_init=0.05, edge_transition_max=0.5,
                 edge_write_init=0.05, edge_write_max=0.25,
                 scan_gate_init=0.05, scan_gate_max=0.5,
                 orientation_temperature=1.0, enable_transition=True,
                 enable_write=True, enable_fusion=True, structure_kernel=1,
                 structure_init_std=0.0, direction_mix=1.0, shortcut=True):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        units = []
        for index in range(n):
            if index == n - 1:
                unit = _AdaptiveCASPUnit(
                    self.c, c3k, shortcut, state_ratio, route_init, route_max, d_state,
                    ssm_ratio, edge_transition_init, edge_transition_max,
                    edge_write_init, edge_write_max, scan_gate_init,
                    scan_gate_max, orientation_temperature, enable_transition,
                    enable_write, enable_fusion, structure_kernel,
                    structure_init_std, direction_mix
                )
            else:
                unit = C3k(self.c, self.c, 2, shortcut) if c3k else Bottleneck(
                    self.c, self.c, shortcut, 1, k=((3, 3), (3, 3)), e=1.0
                )
            units.append(unit)
        self.m = nn.ModuleList(units)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(module(y[-1]) for module in self.m)
        return self.cv2(torch.cat(y, dim=1))


class AdaptiveC2fCASP(AdaptiveC3k2CASP):
    """YOLOv8 C2f-compatible wrapper around the same efficient CASP core."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5,
                 state_ratio=0.25, route_init=0.01, route_max=0.5,
                 d_state=8, ssm_ratio=1.0, edge_transition_init=0.05,
                 edge_transition_max=0.5, edge_write_init=0.05,
                 edge_write_max=0.25, scan_gate_init=0.05,
                 scan_gate_max=0.5, orientation_temperature=1.0,
                 enable_transition=True, enable_write=True, enable_fusion=True,
                 structure_kernel=1, structure_init_std=0.0, direction_mix=1.0):
        # g is accepted for a C2f-compatible YAML signature; the current local
        # bottleneck remains group=1 to keep the CASP wrapper lightweight.
        del g
        super().__init__(
            c1, c2, n, False, e, state_ratio, route_init, route_max, d_state,
            ssm_ratio, edge_transition_init, edge_transition_max,
            edge_write_init, edge_write_max, scan_gate_init, scan_gate_max,
            orientation_temperature, enable_transition, enable_write,
            enable_fusion, structure_kernel, structure_init_std, direction_mix,
            shortcut
        )


class CrackPathSelectiveSSM(nn.Module):
    """Bidirectional selective recurrence over short, already ordered crack paths.

    Crack paths contain only a few tokens but are packed into a very large
    effective batch (image_batch * path_count). The CUDA image-scan kernel was
    designed for the opposite regime and produced non-finite backward values in
    AMP training without making the forward loss NaN. A direct FP32 recurrence is
    both cheaper for these short paths and preserves the same input-dependent
    Delta/B/C state-space update.
    """

    def __init__(self, channels, d_state=8, memory_init=0.05, memory_max=0.5,
                 transition_init=0.05, transition_max=0.5,
                 write_init=0.05, write_max=0.25):
        super().__init__()
        self.channels = int(channels)
        self.d_state = int(d_state)
        self.dt_rank = max(1, math.ceil(channels / 16))
        self.K = 2

        projections = [nn.Linear(channels, self.dt_rank + 2 * self.d_state, bias=False) for _ in range(self.K)]
        self.x_proj_weight = nn.Parameter(torch.stack([layer.weight for layer in projections], dim=0))
        dt_projections = [SS2D.dt_init(self.dt_rank, channels) for _ in range(self.K)]
        self.dt_projs_weight = nn.Parameter(torch.stack([layer.weight for layer in dt_projections], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([layer.bias for layer in dt_projections], dim=0))
        self.A_logs = SS2D.A_log_init(self.d_state, channels, copies=self.K, merge=True)
        self.Ds = SS2D.D_init(channels, copies=self.K, merge=True)
        self.norm = nn.LayerNorm(channels)
        self.out_proj = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        def bounded_raw(initial, maximum, name):
            if not 0.0 < initial < maximum:
                raise ValueError(f"{name} must be between 0 and {maximum}")
            ratio = initial / maximum
            return math.log(ratio / (1.0 - ratio))

        self.memory_raw = nn.Parameter(torch.tensor(bounded_raw(memory_init, memory_max, "memory_init")))
        self.transition_raw = nn.Parameter(
            torch.tensor(bounded_raw(transition_init, transition_max, "transition_init"))
        )
        self.write_raw = nn.Parameter(torch.tensor(bounded_raw(write_init, write_max, "write_init")))
        for parameter in (self.memory_raw, self.transition_raw, self.write_raw):
            parameter._no_weight_decay = True
        self.memory_max = float(memory_max)
        self.transition_max = float(transition_max)
        self.write_max = float(write_max)

    def effective_memory(self):
        return self.memory_max * self.memory_raw.sigmoid()

    def effective_transition(self):
        return self.transition_max * self.transition_raw.sigmoid()

    def effective_write(self):
        return self.write_max * self.write_raw.sigmoid()

    def _selective_recurrence(self, xs, dts, Bs, Cs):
        """Parallel FP32 selective SSM for short NKCL crack paths.

        For h_t = a_t * h_(t-1) + u_t and h_(-1) = 0, the complete state
        sequence is p_t * cumsum(u_t / p_t), where p_t = cumprod(a_t).
        This is algebraically identical to the token loop but launches a small
        fixed set of GPU kernels instead of one autograd graph per path step.
        Crack paths are deliberately short, so their cumulative decay remains
        safely away from FP32 underflow.
        """
        path_batch, scans, channels, length = xs.shape
        state_size = Bs.shape[2]
        A = -torch.exp(self.A_logs.float()).view(scans, channels, state_size)
        D = self.Ds.float().view(scans, channels)
        dt_bias = self.dt_projs_bias.float().view(scans, channels)

        x32, dt32, B32, C32 = xs.float(), dts.float(), Bs.float(), Cs.float()
        delta = F.softplus(dt32 + dt_bias[None, ..., None])
        decay = torch.exp(delta[:, :, :, None, :] * A[None, ..., None])
        write = (
            delta[:, :, :, None, :]
            * B32[:, :, None, :, :]
            * x32[:, :, :, None, :]
        )
        decay_prefix = torch.cumprod(decay, dim=-1)
        # The clamp is inactive in the intended short-path regime and protects
        # deliberately extreme parameter sweeps from division by zero.
        state_sequence = decay_prefix * torch.cumsum(
            write / decay_prefix.clamp_min(1e-20), dim=-1
        )
        read = (state_sequence * C32[:, :, None, :, :]).sum(dim=3)
        return read + D[None, ..., None] * x32

    def forward(self, x, probability, predecessor, valid_mask):
        """Scan N paths with tensors shaped x=NCL and guidance=N1L."""
        path_batch, channels, length = x.shape
        if channels != self.channels:
            raise ValueError(f"expected {self.channels} path channels, got {channels}")
        reverse_predecessor = torch.cat(
            (predecessor.new_zeros((path_batch, 1, 1)), predecessor[..., 1:].flip(-1)), dim=-1
        )
        xs = torch.stack((x, x.flip(-1)), dim=1)
        ps = torch.stack((probability, probability.flip(-1)), dim=1)
        cs = torch.stack((predecessor, reverse_predecessor), dim=1)
        masks = torch.stack((valid_mask, valid_mask.flip(-1)), dim=1)
        xs = xs * masks.to(xs.dtype)

        projected = torch.einsum("nkcl,kdc->nkdl", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(projected, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("nkrl,kcr->nkcl", dts, self.dt_projs_weight)
        dts = dts + self.effective_memory().to(dts.dtype) * (0.5 - ps.to(dts.dtype))
        dts = dts + self.effective_transition().to(dts.dtype) * (1.0 - cs.to(dts.dtype))
        Bs = Bs * (1.0 + self.effective_write().to(Bs.dtype) * (2.0 * ps.to(Bs.dtype) - 1.0))

        output_dtype = xs.dtype
        ys = self._selective_recurrence(xs, dts, Bs, Cs).to(output_dtype)
        merged = 0.5 * (ys[:, 0] + ys[:, 1].flip(-1))
        merged = self.norm(merged.transpose(1, 2)).transpose(1, 2).contiguous()
        return self.out_proj(merged) * valid_mask.to(merged.dtype)


class SparseCrackPathState(nn.Module):
    """Sparse, image-adaptive Mamba propagation along predicted crack curves.

    A light structure head selects sparse seeds and predicts an undirected tangent
    plus four neighbor-connectivity families.  From each seed, two vectorized
    traces follow the locally most compatible 8-neighbor token.  The resulting
    variable image-specific curves are packed as fixed short paths, scanned by a
    bidirectional selective SSM, and scattered back to the 2-D feature map.
    """

    def __init__(self, channels, state_ratio=0.25, seed_ratio=0.02, max_paths=128,
                 path_steps=4, path_min_conf=0.05, route_init=0.02, route_max=0.5,
                 d_state=8, memory_init=0.05, memory_max=0.5,
                 transition_init=0.05, transition_max=0.5,
                 write_init=0.05, write_max=0.25,
                 structure_kernel=3, structure_init_std=0.01):
        super().__init__()
        self.channels = int(channels)
        self.state_ratio = float(state_ratio)
        self.seed_ratio = float(seed_ratio)
        self.max_paths = int(max_paths)
        self.path_steps = int(path_steps)
        self.path_min_conf = float(path_min_conf)
        if not 0.0 < self.seed_ratio <= 1.0:
            raise ValueError("seed_ratio must be in (0, 1]")
        if self.max_paths < 1 or self.path_steps < 1:
            raise ValueError("max_paths and path_steps must be positive")
        if not 0.0 <= self.path_min_conf < 1.0:
            raise ValueError("path_min_conf must be in [0, 1)")
        if not 0.0 < route_init < route_max:
            raise ValueError("route_init must be between 0 and route_max")

        state_channels = max(8, int(round(channels * state_ratio / 8.0)) * 8)
        self.state_channels = min(channels, state_channels)
        self.state_in = nn.Conv2d(channels, self.state_channels, kernel_size=1, bias=False)
        self.norm = LayerNorm2d(self.state_channels)
        self.structure_head = nn.Sequential(
            nn.Conv2d(
                self.state_channels, self.state_channels, kernel_size=structure_kernel,
                padding=structure_kernel // 2, groups=self.state_channels, bias=False
            ),
            nn.GELU(),
            nn.Conv2d(self.state_channels, 7, kernel_size=1, bias=True),
        )
        nn.init.normal_(self.structure_head[-1].weight, mean=0.0, std=structure_init_std)
        nn.init.zeros_(self.structure_head[-1].bias)
        self.path_ssm = CrackPathSelectiveSSM(
            self.state_channels, d_state, memory_init, memory_max,
            transition_init, transition_max, write_init, write_max
        )
        self.state_out = nn.Conv2d(self.state_channels, channels, kernel_size=1, bias=False)
        # ReZero-style safe start: the new block is exactly the local YOLO C3k2
        # path at initialization. The projection learns first, then gradually
        # exposes the crack-path state without corrupting early backbone features.
        nn.init.zeros_(self.state_out.weight)
        route_ratio = route_init / route_max
        self.route_raw = nn.Parameter(torch.tensor(math.log(route_ratio / (1.0 - route_ratio))))
        self.route_raw._no_weight_decay = True
        self.route_max = float(route_max)

        offsets = torch.tensor(
            ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)),
            dtype=torch.long,
        )
        direction_vectors = offsets.float()
        direction_vectors[1::2] *= 2 ** -0.5
        self.register_buffer("neighbor_offsets", offsets, persistent=False)
        self.register_buffer("neighbor_vectors", direction_vectors, persistent=False)
        self.register_buffer("neighbor_families", torch.tensor((0, 2, 1, 3, 0, 2, 1, 3)), persistent=False)

        self.semantic_structure = True
        self.orientation_family_logits = False
        self.last_guidance = None
        self.last_orientation = None
        self.last_connectivity = None
        self.last_path_indices = None
        self.last_path_mask = None
        self.visual_guidance = None
        self.visual_orientation = None
        self.visual_connectivity = None

    def __getstate__(self):
        state = self.__dict__.copy()
        for name in (
            "last_guidance", "last_orientation", "last_connectivity", "last_path_indices",
            "last_path_mask", "visual_guidance", "visual_orientation", "visual_connectivity"
        ):
            state.pop(name, None)
        return state

    def effective_route(self):
        return self.route_max * self.route_raw.sigmoid()

    @staticmethod
    def _gather_flat(field, indices):
        return field.flatten(2).gather(2, indices[:, None].expand(-1, field.shape[1], -1))

    def _trace(self, probability, orientation, connectivity, seeds, sign):
        batch, _, height, width = probability.shape
        paths = seeds.shape[1]
        current = seeds
        current_row, current_col = current.div(width, rounding_mode="floor"), current.remainder(width)
        seed_orientation = self._gather_flat(orientation, current)
        theta = 0.5 * torch.atan2(seed_orientation[:, 1], seed_orientation[:, 0])
        heading = torch.stack((torch.sin(theta), torch.cos(theta)), dim=-1) * float(sign)
        active = torch.ones((batch, paths), dtype=torch.bool, device=probability.device)
        indices, edges, masks = [current], [probability.new_zeros((batch, paths))], [active]

        for _ in range(self.path_steps):
            candidate_row = current_row[..., None] + self.neighbor_offsets[:, 0]
            candidate_col = current_col[..., None] + self.neighbor_offsets[:, 1]
            valid = (
                (candidate_row >= 0) & (candidate_row < height)
                & (candidate_col >= 0) & (candidate_col < width)
            )
            safe_row = candidate_row.clamp(0, height - 1)
            safe_col = candidate_col.clamp(0, width - 1)
            candidate = safe_row * width + safe_col

            flat_candidate = candidate.reshape(batch, -1)
            neighbor_probability = self._gather_flat(probability, flat_candidate)[:, 0].view(batch, paths, 8)
            neighbor_orientation = self._gather_flat(orientation, flat_candidate).view(batch, 2, paths, 8)
            neighbor_connectivity = self._gather_flat(connectivity, flat_candidate).view(batch, 4, paths, 8)
            current_orientation = self._gather_flat(orientation, current)
            current_connectivity = self._gather_flat(connectivity, current)

            local_theta = 0.5 * torch.atan2(current_orientation[:, 1], current_orientation[:, 0])
            local_tangent = torch.stack((torch.sin(local_theta), torch.cos(local_theta)), dim=-1)
            local_tangent = torch.where(
                (local_tangent * heading).sum(-1, keepdim=True) < 0, -local_tangent, local_tangent
            )
            directions = self.neighbor_vectors.to(probability.dtype)
            current_alignment = (local_tangent[:, :, None] * directions).sum(-1).clamp_min(0.0)
            heading_alignment = (heading[:, :, None] * directions).sum(-1).clamp_min(0.0)

            neighbor_theta = 0.5 * torch.atan2(neighbor_orientation[:, 1], neighbor_orientation[:, 0])
            neighbor_tangent = torch.stack((torch.sin(neighbor_theta), torch.cos(neighbor_theta)), dim=-1)
            neighbor_alignment = (neighbor_tangent * directions[None, None]).sum(-1).abs()
            family = self.neighbor_families
            current_family = current_connectivity[:, family].permute(0, 2, 1)
            family_index = family.view(1, 1, 1, 8).expand(batch, 1, paths, 8)
            neighbor_family = neighbor_connectivity.gather(1, family_index).squeeze(1)
            connection = torch.sqrt((current_family * neighbor_family).clamp_min(1e-6))
            score = neighbor_probability * connection * current_alignment * neighbor_alignment
            score = score * (0.5 + 0.5 * heading_alignment)
            score = score.masked_fill(~valid, -1.0)
            best_score, best_direction = score.max(dim=-1)
            next_index = candidate.gather(-1, best_direction[..., None]).squeeze(-1)
            step_active = active & (best_score >= self.path_min_conf)
            current = torch.where(step_active, next_index, current)
            current_row, current_col = current.div(width, rounding_mode="floor"), current.remainder(width)
            chosen_heading = directions[best_direction]
            heading = torch.where(step_active[..., None], chosen_heading, heading)
            active = step_active
            indices.append(current)
            edges.append(best_score.clamp(0.0, 1.0) * active.to(best_score.dtype))
            masks.append(active)
        return torch.stack(indices, dim=-1), torch.stack(edges, dim=-1), torch.stack(masks, dim=-1)

    def _build_paths(self, feature, probability, orientation, connectivity):
        batch, _, height, width = feature.shape
        path_count = min(self.max_paths, max(1, int(round(height * width * self.seed_ratio))))
        seeds = probability.detach().flatten(1).topk(path_count, dim=1).indices
        backward_idx, backward_edge, backward_mask = self._trace(
            probability, orientation, connectivity, seeds, -1
        )
        forward_idx, forward_edge, forward_mask = self._trace(
            probability, orientation, connectivity, seeds, 1
        )
        reverse_transition = torch.cat(
            (backward_edge.new_zeros((*backward_edge.shape[:2], 1)), backward_edge[..., 1:].flip(-1)), dim=-1
        )
        indices = torch.cat((backward_idx[..., 1:].flip(-1), forward_idx), dim=-1)
        predecessor = torch.cat((reverse_transition, forward_edge[..., 1:]), dim=-1)
        valid_mask = torch.cat((backward_mask[..., 1:].flip(-1), forward_mask), dim=-1)
        flat_indices = indices.reshape(batch, -1)
        path_feature = self._gather_flat(feature, flat_indices).view(
            batch, feature.shape[1], path_count, -1
        ).permute(0, 2, 1, 3).reshape(batch * path_count, feature.shape[1], -1)
        path_probability = self._gather_flat(probability, flat_indices).view(batch * path_count, 1, -1)
        return (
            path_feature,
            path_probability,
            predecessor.reshape(batch * path_count, 1, -1),
            valid_mask.reshape(batch * path_count, 1, -1),
            indices,
        )

    def forward(self, x):
        compact = self.norm(self.state_in(x))
        structure = self.structure_head(compact)
        probability = structure[:, :1].sigmoid()
        orientation = structure[:, 1:3].tanh()
        connectivity = structure[:, 3:].sigmoid()
        self.last_guidance, self.last_orientation, self.last_connectivity = probability, orientation, connectivity
        self.visual_guidance = probability.detach()
        self.visual_orientation = orientation.detach()
        self.visual_connectivity = connectivity.detach()

        # Top-k, argmax direction selection and thresholded continuation form a
        # hard routing policy. Differentiating through its atan2/sqrt scores is
        # neither meaningful (the selected indices are discrete) nor AMP-safe:
        # near-zero FP16 orientation vectors make atan2 derivatives singular and
        # can yield 0 * Inf = NaN even behind a zero-initialized residual. The
        # structure head is trained explicitly by probability/tangent/connectivity
        # losses, while detection gradients still train gathered path features,
        # the selective recurrence and the residual projection.
        path_feature, path_probability, predecessor, valid_mask, indices = self._build_paths(
            compact, probability.detach(), orientation.detach(), connectivity.detach()
        )
        path_output = self.path_ssm(path_feature, path_probability, predecessor, valid_mask)
        batch, _, height, width = compact.shape
        path_count, path_length = indices.shape[1:]
        values = path_output.view(batch, path_count, self.state_channels, path_length).permute(0, 2, 1, 3)
        mask = valid_mask.view(batch, path_count, 1, path_length).permute(0, 2, 1, 3).to(values.dtype)
        flat_index = indices.reshape(batch, 1, -1).expand(-1, self.state_channels, -1)
        accumulated = compact.new_zeros((batch, self.state_channels, height * width))
        counts = compact.new_zeros((batch, 1, height * width))
        accumulated.scatter_add_(2, flat_index, (values * mask).reshape(batch, self.state_channels, -1))
        counts.scatter_add_(2, indices.reshape(batch, 1, -1), mask.reshape(batch, 1, -1))
        sparse_delta = accumulated / counts.clamp_min(1.0)
        sparse_delta = sparse_delta.view(batch, self.state_channels, height, width)
        sparse_mask = (counts > 0).view(batch, 1, height, width).to(sparse_delta.dtype)
        self.last_path_indices = indices.detach()
        self.last_path_mask = valid_mask.view(batch, path_count, path_length).detach()
        return x + self.effective_route() * sparse_mask * self.state_out(sparse_delta)


class _AdaptiveCrackPathUnit(nn.Module):
    def __init__(self, channels, c3k=False, shortcut=True, **path_kwargs):
        super().__init__()
        self.local = C3k(channels, channels, 2, shortcut) if c3k else Bottleneck(
            channels, channels, shortcut, 1, k=((3, 3), (3, 3)), e=1.0
        )
        self.path = SparseCrackPathState(channels, **path_kwargs)

    def forward(self, x):
        return self.path(self.local(x))


class AdaptiveC3k2CrackPath(nn.Module):
    """C3k2-compatible final sparse adaptive crack-path Mamba block."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, state_ratio=0.25,
                 seed_ratio=0.02, max_paths=128, path_steps=4, path_min_conf=0.05,
                 route_init=0.02, route_max=0.5, d_state=8,
                 memory_init=0.05, memory_max=0.5,
                 transition_init=0.05, transition_max=0.5,
                 write_init=0.05, write_max=0.25,
                 structure_kernel=3, structure_init_std=0.01, shortcut=True):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        path_kwargs = dict(
            state_ratio=state_ratio, seed_ratio=seed_ratio, max_paths=max_paths,
            path_steps=path_steps, path_min_conf=path_min_conf,
            route_init=route_init, route_max=route_max, d_state=d_state,
            memory_init=memory_init, memory_max=memory_max,
            transition_init=transition_init, transition_max=transition_max,
            write_init=write_init, write_max=write_max,
            structure_kernel=structure_kernel, structure_init_std=structure_init_std,
        )
        units = []
        for index in range(n):
            if index == n - 1:
                unit = _AdaptiveCrackPathUnit(self.c, c3k, shortcut, **path_kwargs)
            else:
                unit = C3k(self.c, self.c, 2, shortcut) if c3k else Bottleneck(
                    self.c, self.c, shortcut, 1, k=((3, 3), (3, 3)), e=1.0
                )
            units.append(unit)
        self.m = nn.ModuleList(units)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(module(y[-1]) for module in self.m)
        return self.cv2(torch.cat(y, dim=1))


class SimpleStem(nn.Module):
    def __init__(self, inp, embed_dim, ks=3):
        super().__init__()
        self.hidden_dims = embed_dim // 2
        self.conv = nn.Sequential(
            nn.Conv2d(inp, self.hidden_dims, kernel_size=ks, stride=2, padding=autopad(ks, d=1), bias=False),
            nn.BatchNorm2d(self.hidden_dims),
            nn.GELU(),
            nn.Conv2d(self.hidden_dims, embed_dim, kernel_size=ks, stride=2, padding=autopad(ks, d=1), bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.conv(x)


class CrackDetailStemLite(nn.Module):
    """Light stem that preserves sub-pixel details with high-pass gating and space-to-depth."""

    def __init__(self, inp, embed_dim, ks=3):
        super().__init__()
        mid = max(embed_dim // 2, 8)
        self.first = nn.Sequential(
            nn.Conv2d(inp, mid, kernel_size=ks, stride=2, padding=autopad(ks), bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(),
        )
        self.local = nn.Sequential(
            nn.Conv2d(mid, mid, 3, 1, 1, groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(),
        )
        gate_hidden = max(mid // 4, 8)
        self.detail_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid, gate_hidden, 1),
            nn.SiLU(),
            nn.Conv2d(gate_hidden, mid, 1),
            nn.Sigmoid(),
        )
        self.project = nn.Sequential(
            nn.Conv2d(mid * 4, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(),
        )

    def forward(self, x):
        x = self.first(x)
        local = self.local(x)
        detail = x - F.avg_pool2d(x, 3, 1, 1)
        fused = local + self.detail_gate(local) * detail
        return self.project(F.pixel_unshuffle(fused, 2))


class CrackDetailStemDirectional(nn.Module):
    """Dual-branch stem using cheap horizontal/vertical differences to retain thin line evidence."""

    def __init__(self, inp, embed_dim, ks=3):
        super().__init__()
        mid = max(embed_dim // 2, 8)
        self.first = nn.Sequential(
            nn.Conv2d(inp, mid, kernel_size=ks, stride=2, padding=autopad(ks), bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(),
        )
        self.local = nn.Sequential(
            nn.Conv2d(mid, mid, 3, 2, 1, groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(),
            nn.Conv2d(mid, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(),
        )
        self.detail_project = nn.Sequential(
            nn.Conv2d(mid * 2, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(),
        )
        self.spatial_gate = nn.Sequential(nn.Conv2d(embed_dim, 1, 1), nn.Sigmoid())
        self.detail_scale = nn.Parameter(torch.tensor(math.atanh(0.1)))

    def forward(self, x):
        x = self.first(x)
        local = self.local(x)
        dx = F.pad((x[..., 1:] - x[..., :-1]).abs(), (0, 1, 0, 0))
        dy = F.pad((x[..., 1:, :] - x[..., :-1, :]).abs(), (0, 0, 0, 1))
        detail = F.avg_pool2d(torch.cat((dx, dy), dim=1), 2, 2)
        detail = self.detail_project(detail)
        return local + self.detail_scale.tanh() * self.spatial_gate(detail) * detail


class VisionClueMerge(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.hidden = int(dim * 4)

        self.pw_linear = nn.Sequential(
            nn.Conv2d(self.hidden, out_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_dim),
            nn.SiLU()
        )

    def forward(self, x):
        y = torch.cat([
            x[..., ::2, ::2],
            x[..., 1::2, ::2],
            x[..., ::2, 1::2],
            x[..., 1::2, 1::2]
        ], dim=1)
        return self.pw_linear(y)


class CrackMergeLite(nn.Module):
    """Space-to-depth merge augmented with a cheap anti-aliased local-context residual."""

    def __init__(self, dim, out_dim):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim * 4, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.SiLU(),
        )
        self.local = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(),
            nn.Conv2d(dim, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
        )
        self.local_scale = nn.Parameter(torch.tensor(math.atanh(0.1)))

    def forward(self, x):
        main = self.main(F.pixel_unshuffle(x, 2))
        local = self.local(F.avg_pool2d(x, 2, 2))
        return main + self.local_scale.tanh() * local


class CrackMergeDirectional(nn.Module):
    """Downsample by retaining four sub-pixels and explicit H/V inter-subpixel differences."""

    def __init__(self, dim, out_dim):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim * 4, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.SiLU(),
        )
        self.detail = nn.Sequential(
            nn.Conv2d(dim * 2, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.SiLU(),
        )
        self.gate = nn.Sequential(nn.Conv2d(out_dim, 1, 1), nn.Sigmoid())
        self.detail_scale = nn.Parameter(torch.tensor(math.atanh(0.1)))

    def forward(self, x):
        q00 = x[..., ::2, ::2]
        q10 = x[..., 1::2, ::2]
        q01 = x[..., ::2, 1::2]
        q11 = x[..., 1::2, 1::2]
        main = self.main(torch.cat((q00, q10, q01, q11), dim=1))
        horizontal = (q00 - q01).abs() + (q10 - q11).abs()
        vertical = (q00 - q10).abs() + (q01 - q11).abs()
        detail = self.detail(torch.cat((horizontal, vertical), dim=1))
        return main + self.detail_scale.tanh() * self.gate(detail) * detail
