"""Readable reference implementation of the G01 core innovation.

This file is documentation code: it isolates the algorithm used by G01 from the
many historical experiments in ``mamba_yolo.py``. The production/training
implementation remains:

    ultralytics/nn/modules/mamba_yolo.py

The four classes below reproduce the G01 path construction, crack-aware state
update, sparse scatter, and C3k2-compatible host block. They are intentionally
renamed so importing this file cannot silently replace the registered training
classes.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.block import Bottleneck, C3k
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.mamba_yolo import SS2D
from ultralytics.nn.modules.transformer import LayerNorm2d


G01_DEFAULTS = {
    "state_ratio": 0.25,
    "seed_ratio": 0.02,
    "max_paths": 128,
    "path_steps": 3,
    "path_min_conf": 0.05,
    "route_init": 0.02,
    "route_max": 0.50,
    "d_state": 8,
    "memory_init": 0.05,
    "memory_max": 0.50,
    "transition_init": 0.05,
    "transition_max": 0.50,
    "write_init": 0.05,
    "write_max": 0.25,
}


def _bounded_logit(initial: float, maximum: float, name: str) -> float:
    """Parameterize a positive scalar as maximum * sigmoid(raw)."""
    if not 0.0 < initial < maximum:
        raise ValueError(f"{name} must be between 0 and {maximum}")
    ratio = initial / maximum
    return math.log(ratio / (1.0 - ratio))


class G01CrackPathSelectiveSSM(nn.Module):
    """Bidirectional selective state update on ordered short crack paths."""

    def __init__(
        self,
        channels: int,
        d_state: int = 8,
        memory_init: float = 0.05,
        memory_max: float = 0.50,
        transition_init: float = 0.05,
        transition_max: float = 0.50,
        write_init: float = 0.05,
        write_max: float = 0.25,
    ):
        super().__init__()
        self.channels = int(channels)
        self.d_state = int(d_state)
        self.dt_rank = max(1, math.ceil(channels / 16))
        self.K = 2  # forward and reverse recurrence

        x_projections = [
            nn.Linear(channels, self.dt_rank + 2 * self.d_state, bias=False)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([layer.weight for layer in x_projections]))
        dt_projections = [SS2D.dt_init(self.dt_rank, channels) for _ in range(self.K)]
        self.dt_projs_weight = nn.Parameter(torch.stack([layer.weight for layer in dt_projections]))
        self.dt_projs_bias = nn.Parameter(torch.stack([layer.bias for layer in dt_projections]))
        self.A_logs = SS2D.A_log_init(self.d_state, channels, copies=self.K, merge=True)
        self.Ds = SS2D.D_init(channels, copies=self.K, merge=True)
        self.norm = nn.LayerNorm(channels)
        self.out_proj = nn.Conv1d(channels, channels, 1, bias=False)

        self.memory_raw = nn.Parameter(torch.tensor(_bounded_logit(memory_init, memory_max, "memory")))
        self.transition_raw = nn.Parameter(
            torch.tensor(_bounded_logit(transition_init, transition_max, "transition"))
        )
        self.write_raw = nn.Parameter(torch.tensor(_bounded_logit(write_init, write_max, "write")))
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

    def _recurrence(self, xs, dts, Bs, Cs):
        """Parallel FP32 form of h_t = exp(delta_t A)h_(t-1) + delta_t B_t x_t."""
        _, scans, channels, _ = xs.shape
        state_size = Bs.shape[2]
        A = -torch.exp(self.A_logs.float()).view(scans, channels, state_size)
        D = self.Ds.float().view(scans, channels)
        dt_bias = self.dt_projs_bias.float().view(scans, channels)

        x32, dt32, B32, C32 = xs.float(), dts.float(), Bs.float(), Cs.float()
        delta = F.softplus(dt32 + dt_bias[None, ..., None])
        decay = torch.exp(delta[:, :, :, None, :] * A[None, ..., None])
        write = delta[:, :, :, None, :] * B32[:, :, None, :, :] * x32[:, :, :, None, :]
        decay_prefix = torch.cumprod(decay, dim=-1)
        states = decay_prefix * torch.cumsum(write / decay_prefix.clamp_min(1e-20), dim=-1)
        read = (states * C32[:, :, None, :, :]).sum(dim=3)
        return read + D[None, ..., None] * x32

    def forward(self, x, probability, predecessor_confidence, valid_mask):
        """Args use NCL paths; probability/confidence/mask use N1L."""
        path_batch, channels, _ = x.shape
        if channels != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {channels}")

        reverse_confidence = torch.cat(
            (
                predecessor_confidence.new_zeros((path_batch, 1, 1)),
                predecessor_confidence[..., 1:].flip(-1),
            ),
            dim=-1,
        )
        xs = torch.stack((x, x.flip(-1)), dim=1)
        ps = torch.stack((probability, probability.flip(-1)), dim=1)
        ss = torch.stack((predecessor_confidence, reverse_confidence), dim=1)
        masks = torch.stack((valid_mask, valid_mask.flip(-1)), dim=1)
        xs = xs * masks.to(xs.dtype)

        projected = torch.einsum("nkcl,kdc->nkdl", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(projected, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("nkrl,kcr->nkcl", dts, self.dt_projs_weight)

        # The same structural cues used to order tokens now manage state memory.
        dts = dts + self.effective_memory().to(dts.dtype) * (0.5 - ps.to(dts.dtype))
        dts = dts + self.effective_transition().to(dts.dtype) * (1.0 - ss.to(dts.dtype))
        Bs = Bs * (1.0 + self.effective_write().to(Bs.dtype) * (2.0 * ps.to(Bs.dtype) - 1.0))

        ys = self._recurrence(xs, dts, Bs, Cs).to(xs.dtype)
        merged = 0.5 * (ys[:, 0] + ys[:, 1].flip(-1))
        merged = self.norm(merged.transpose(1, 2)).transpose(1, 2).contiguous()
        return self.out_proj(merged) * valid_mask.to(merged.dtype)


class G01SparseCrackPathState(nn.Module):
    """Predict structure, trace sparse curves, scan them, and scatter states to 2-D."""

    def __init__(
        self,
        channels: int,
        state_ratio: float = 0.25,
        seed_ratio: float = 0.02,
        max_paths: int = 128,
        path_steps: int = 3,
        path_min_conf: float = 0.05,
        route_init: float = 0.02,
        route_max: float = 0.50,
        d_state: int = 8,
        memory_init: float = 0.05,
        memory_max: float = 0.50,
        transition_init: float = 0.05,
        transition_max: float = 0.50,
        write_init: float = 0.05,
        write_max: float = 0.25,
        structure_kernel: int = 3,
        structure_init_std: float = 0.01,
    ):
        super().__init__()
        self.channels = int(channels)
        self.state_ratio = float(state_ratio)
        self.seed_ratio = float(seed_ratio)
        self.max_paths = int(max_paths)
        self.path_steps = int(path_steps)
        self.path_min_conf = float(path_min_conf)
        state_channels = max(8, int(round(channels * state_ratio / 8.0)) * 8)
        self.state_channels = min(channels, state_channels)

        self.state_in = nn.Conv2d(channels, self.state_channels, 1, bias=False)
        self.norm = LayerNorm2d(self.state_channels)
        self.structure_head = nn.Sequential(
            nn.Conv2d(
                self.state_channels,
                self.state_channels,
                structure_kernel,
                padding=structure_kernel // 2,
                groups=self.state_channels,
                bias=False,
            ),
            nn.GELU(),
            nn.Conv2d(self.state_channels, 7, 1, bias=True),
        )
        nn.init.normal_(self.structure_head[-1].weight, 0.0, structure_init_std)
        nn.init.zeros_(self.structure_head[-1].bias)

        self.path_ssm = G01CrackPathSelectiveSSM(
            self.state_channels,
            d_state,
            memory_init,
            memory_max,
            transition_init,
            transition_max,
            write_init,
            write_max,
        )
        self.state_out = nn.Conv2d(self.state_channels, channels, 1, bias=False)
        nn.init.zeros_(self.state_out.weight)
        self.route_raw = nn.Parameter(torch.tensor(_bounded_logit(route_init, route_max, "route")))
        self.route_raw._no_weight_decay = True
        self.route_max = float(route_max)

        offsets = torch.tensor(
            ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)),
            dtype=torch.long,
        )
        vectors = offsets.float()
        vectors[1::2] *= 2 ** -0.5
        self.register_buffer("neighbor_offsets", offsets, persistent=False)
        self.register_buffer("neighbor_vectors", vectors, persistent=False)
        self.register_buffer("neighbor_families", torch.tensor((0, 2, 1, 3, 0, 2, 1, 3)), persistent=False)

        # Live tensors are consumed by the auxiliary segmentation losses.
        self.semantic_structure = True
        self.orientation_family_logits = False
        self.last_guidance = None
        self.last_orientation = None
        self.last_connectivity = None
        self.last_path_indices = None
        self.last_path_mask = None

    def effective_route(self):
        return self.route_max * self.route_raw.sigmoid()

    @staticmethod
    def _gather_flat(field, indices):
        return field.flatten(2).gather(2, indices[:, None].expand(-1, field.shape[1], -1))

    def _trace(self, probability, orientation, connectivity, seeds, sign):
        """Greedily trace one side of every seed through its 8-neighborhood."""
        batch, _, height, width = probability.shape
        paths = seeds.shape[1]
        current = seeds
        row, col = current.div(width, rounding_mode="floor"), current.remainder(width)
        seed_orientation = self._gather_flat(orientation, current)
        theta = 0.5 * torch.atan2(seed_orientation[:, 1], seed_orientation[:, 0])
        heading = torch.stack((torch.sin(theta), torch.cos(theta)), dim=-1) * float(sign)
        active = torch.ones((batch, paths), dtype=torch.bool, device=probability.device)
        indices = [current]
        edge_scores = [probability.new_zeros((batch, paths))]
        masks = [active]

        for _ in range(self.path_steps):
            candidate_row = row[..., None] + self.neighbor_offsets[:, 0]
            candidate_col = col[..., None] + self.neighbor_offsets[:, 1]
            valid = (
                (candidate_row >= 0)
                & (candidate_row < height)
                & (candidate_col >= 0)
                & (candidate_col < width)
            )
            safe_row = candidate_row.clamp(0, height - 1)
            safe_col = candidate_col.clamp(0, width - 1)
            candidate = safe_row * width + safe_col
            flat_candidate = candidate.reshape(batch, -1)

            neighbor_p = self._gather_flat(probability, flat_candidate)[:, 0].view(batch, paths, 8)
            neighbor_o = self._gather_flat(orientation, flat_candidate).view(batch, 2, paths, 8)
            neighbor_c = self._gather_flat(connectivity, flat_candidate).view(batch, 4, paths, 8)
            current_o = self._gather_flat(orientation, current)
            current_c = self._gather_flat(connectivity, current)
            directions = self.neighbor_vectors.to(probability.dtype)

            local_theta = 0.5 * torch.atan2(current_o[:, 1], current_o[:, 0])
            local_tangent = torch.stack((torch.sin(local_theta), torch.cos(local_theta)), dim=-1)
            local_tangent = torch.where(
                (local_tangent * heading).sum(-1, keepdim=True) < 0,
                -local_tangent,
                local_tangent,
            )
            current_alignment = (local_tangent[:, :, None] * directions).sum(-1).clamp_min(0.0)
            heading_alignment = (heading[:, :, None] * directions).sum(-1).clamp_min(0.0)

            neighbor_theta = 0.5 * torch.atan2(neighbor_o[:, 1], neighbor_o[:, 0])
            neighbor_tangent = torch.stack((torch.sin(neighbor_theta), torch.cos(neighbor_theta)), dim=-1)
            neighbor_alignment = (neighbor_tangent * directions[None, None]).sum(-1).abs()
            family = self.neighbor_families
            current_family = current_c[:, family].permute(0, 2, 1)
            family_index = family.view(1, 1, 1, 8).expand(batch, 1, paths, 8)
            neighbor_family = neighbor_c.gather(1, family_index).squeeze(1)
            connection = torch.sqrt((current_family * neighbor_family).clamp_min(1e-6))

            score = neighbor_p * connection
            score = score * current_alignment
            score = score * neighbor_alignment
            score = score * (0.5 + 0.5 * heading_alignment)
            score = score.masked_fill(~valid, -1.0)
            best_score, best_direction = score.max(dim=-1)
            next_index = candidate.gather(-1, best_direction[..., None]).squeeze(-1)
            step_active = active & (best_score >= self.path_min_conf)
            current = torch.where(step_active, next_index, current)
            row, col = current.div(width, rounding_mode="floor"), current.remainder(width)
            heading = torch.where(step_active[..., None], directions[best_direction], heading)
            active = step_active
            indices.append(current)
            edge_scores.append(best_score.clamp(0.0, 1.0) * active.to(best_score.dtype))
            masks.append(active)

        return torch.stack(indices, -1), torch.stack(edge_scores, -1), torch.stack(masks, -1)

    def _build_paths(self, feature, probability, orientation, connectivity):
        batch, channels, height, width = feature.shape
        path_count = min(self.max_paths, max(1, round(height * width * self.seed_ratio)))
        seeds = probability.detach().flatten(1).topk(path_count, dim=1).indices
        backward_idx, backward_edge, backward_mask = self._trace(
            probability, orientation, connectivity, seeds, -1
        )
        forward_idx, forward_edge, forward_mask = self._trace(
            probability, orientation, connectivity, seeds, 1
        )
        reverse_edge = torch.cat(
            (backward_edge.new_zeros((*backward_edge.shape[:2], 1)), backward_edge[..., 1:].flip(-1)),
            dim=-1,
        )
        indices = torch.cat((backward_idx[..., 1:].flip(-1), forward_idx), dim=-1)
        predecessor = torch.cat((reverse_edge, forward_edge[..., 1:]), dim=-1)
        valid_mask = torch.cat((backward_mask[..., 1:].flip(-1), forward_mask), dim=-1)
        flat_indices = indices.reshape(batch, -1)
        path_feature = self._gather_flat(feature, flat_indices).view(
            batch, channels, path_count, -1
        ).permute(0, 2, 1, 3).reshape(batch * path_count, channels, -1)
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
        orientation = structure[:, 1:3].tanh()       # cos(2theta), sin(2theta)
        connectivity = structure[:, 3:].sigmoid()   # H, V, main diag, anti diag
        self.last_guidance = probability
        self.last_orientation = orientation
        self.last_connectivity = connectivity

        # Routing is discrete; structure tensors are supervised by auxiliary losses.
        path_feature, path_probability, predecessor, valid_mask, indices = self._build_paths(
            compact,
            probability.detach(),
            orientation.detach(),
            connectivity.detach(),
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
        sparse_delta = (accumulated / counts.clamp_min(1.0)).view(
            batch, self.state_channels, height, width
        )
        sparse_mask = (counts > 0).view(batch, 1, height, width).to(sparse_delta.dtype)
        self.last_path_indices = indices.detach()
        self.last_path_mask = valid_mask.view(batch, path_count, path_length).detach()
        return x + self.effective_route() * sparse_mask * self.state_out(sparse_delta)


class _G01AdaptiveUnit(nn.Module):
    """Original local unit followed by one sparse crack-path state route."""

    def __init__(self, channels, c3k=False, shortcut=True, **path_kwargs):
        super().__init__()
        self.local = C3k(channels, channels, 2, shortcut) if c3k else Bottleneck(
            channels, channels, shortcut, 1, k=((3, 3), (3, 3)), e=1.0
        )
        self.path = G01SparseCrackPathState(channels, **path_kwargs)

    def forward(self, x):
        return self.path(self.local(x))


class G01AdaptiveC3k2CrackPath(nn.Module):
    """C3k2-compatible G01 block: only its final internal unit gains path state."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, shortcut=True, **path_kwargs):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        units = []
        for index in range(n):
            if index == n - 1:
                unit = _G01AdaptiveUnit(self.c, c3k, shortcut, **path_kwargs)
            else:
                unit = C3k(self.c, self.c, 2, shortcut) if c3k else Bottleneck(
                    self.c, self.c, shortcut, 1, k=((3, 3), (3, 3)), e=1.0
                )
            units.append(unit)
        self.m = nn.ModuleList(units)

    def forward(self, x):
        branches = list(self.cv1(x).chunk(2, dim=1))
        branches.extend(unit(branches[-1]) for unit in self.m)
        return self.cv2(torch.cat(branches, dim=1))


__all__ = (
    "G01_DEFAULTS",
    "G01CrackPathSelectiveSSM",
    "G01SparseCrackPathState",
    "G01AdaptiveC3k2CrackPath",
)
