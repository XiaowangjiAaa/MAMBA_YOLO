import torch
import math
from functools import partial
from typing import Callable, Any

import torch.nn as nn
from einops import rearrange, repeat
from timm.models.layers import DropPath

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"
try:
    import selective_scan_cuda_core
    import selective_scan_cuda_oflex
    import selective_scan_cuda_ndstate
    import selective_scan_cuda_nrow
    import selective_scan_cuda
except:
    pass

try:
    "sscore acts the same as mamba_ssm"
    import selective_scan_cuda_core
except Exception as e:
    print(e, flush=True)
    "you should install mamba_ssm to use this"
    SSMODE = "mamba_ssm"
    import selective_scan_cuda
    # from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref


class LayerNorm2d(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps, elementwise_affine)

    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c').contiguous()
        x = self.norm(x)
        x = rearrange(x, 'b h w c -> b c h w').contiguous()
        return x


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


# Cross Scan
class CrossScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 4, C, H * W))
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        return y.view(B, -1, H, W)


class CrossMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 4, C, L))
        xs[:, 0] = x
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        xs = xs.view(B, 4, C, H, W)
        return xs, None, None


_SCAN_ORDER_CACHE = {}


def _diagonal_order(height, width, anti=False):
    """Return a flattened order that keeps pixels from each diagonal contiguous."""
    order = []
    if anti:
        keys = range(height + width - 1)
        for key in keys:
            diagonal = [(row, key - row) for row in range(height) if 0 <= key - row < width]
            order.extend(row * width + col for row, col in diagonal)
    else:
        keys = range(-(width - 1), height)
        for key in keys:
            diagonal = [(row, row - key) for row in range(height) if 0 <= row - key < width]
            order.extend(row * width + col for row, col in diagonal)
    return order


def get_scan_orders(height, width, mode, device):
    """Build scan and inverse-scan permutations for oriented selective scan."""
    cache_key = (height, width, mode, str(device))
    if cache_key in _SCAN_ORDER_CACHE:
        return _SCAN_ORDER_CACHE[cache_key]

    row = list(range(height * width))
    col = [r * width + c for c in range(width) for r in range(height)]
    forward = [row, col]
    family_ids = [0, 1]
    if mode == "oriented_hvd":
        forward.extend((_diagonal_order(height, width), _diagonal_order(height, width, anti=True)))
        family_ids.extend((2, 3))
    orders = forward + [list(reversed(order)) for order in forward]
    family_ids = family_ids + family_ids
    orders = torch.tensor(orders, dtype=torch.long, device=device)
    inverse = torch.empty_like(orders)
    positions = torch.arange(height * width, device=device).expand_as(orders)
    inverse.scatter_(1, orders, positions)
    family_ids = torch.tensor(family_ids, dtype=torch.long, device=device)
    _SCAN_ORDER_CACHE[cache_key] = (orders, inverse, family_ids)
    return orders, inverse, family_ids


def ordered_scan(x, orders):
    """Convert BCHW features to BKCL sequences using arbitrary spatial permutations."""
    batch, channels, height, width = x.shape
    flat = x.flatten(2)
    return flat[:, None].expand(-1, orders.shape[0], -1, -1).gather(
        -1, orders[None, :, None].expand(batch, -1, channels, -1)
    )


def ordered_merge(ys, inverse_orders, family_ids, direction_weights=None):
    """Align scan outputs to image coordinates and optionally fuse direction families per pixel."""
    batch, scans, channels, height, width = ys.shape
    aligned = ys.flatten(3).gather(
        -1, inverse_orders[None, :, None].expand(batch, -1, channels, -1)
    )
    if direction_weights is not None:
        weights = direction_weights.flatten(2)[:, family_ids]
        aligned = aligned * weights[:, :, None]
    return aligned.sum(dim=1)


# cross selective scan ===============================
class SelectiveScanCore(torch.autograd.Function):
    # comment all checks if inside cross_selective_scan
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1,
                oflex=True):
        # all in float
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None and D.stride(-1) != 1:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if B.dim() == 3:
            B = B.unsqueeze(dim=1)
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = C.unsqueeze(dim=1)
            ctx.squeeze_C = True
        ctx.delta_softplus = delta_softplus
        ctx.backnrows = backnrows
        out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


def cross_selective_scan(
        x: torch.Tensor = None,
        x_proj_weight: torch.Tensor = None,
        x_proj_bias: torch.Tensor = None,
        dt_projs_weight: torch.Tensor = None,
        dt_projs_bias: torch.Tensor = None,
        A_logs: torch.Tensor = None,
        Ds: torch.Tensor = None,
        out_norm: torch.nn.Module = None,
        out_norm_shape="v0",
        nrows=-1,  # for SelectiveScanNRow
        backnrows=-1,  # for SelectiveScanNRow
        delta_softplus=True,
        to_dtype=True,
        force_fp32=False,  # False if ssoflex
        ssoflex=True,
        SelectiveScan=None,
        scan_mode_type='default',
        delta_guidance: torch.Tensor = None,
        delta_alpha: torch.Tensor = None,
        delta_guidance_center: float = 1.0,
        write_guidance: torch.Tensor = None,
        write_beta: torch.Tensor = None,
        transition_guidance: torch.Tensor = None,
        transition_alpha: torch.Tensor = None,
        scan_mode: str = "cross",
        direction_weights: torch.Tensor = None,
):
    # out_norm: whatever fits (B, L, C); LayerNorm; Sigmoid; Softmax(dim=1);...

    B, D, H, W = x.shape
    D, N = A_logs.shape
    K, D, R = dt_projs_weight.shape
    L = H * W

    def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
        return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)

    oriented_scan = scan_mode in {"oriented_hv", "oriented_hvd"}
    if oriented_scan:
        orders, inverse_orders, family_ids = get_scan_orders(H, W, scan_mode, x.device)
        if orders.shape[0] != K:
            raise ValueError(f"scan mode {scan_mode} needs K={orders.shape[0]}, but the module was built with K={K}")
        xs = ordered_scan(x, orders)
    else:
        orders = inverse_orders = family_ids = None
        xs = CrossScan.apply(x)

    def guidance_scan(guidance, name):
        """Align scalar or scan-family guidance with every directional sequence."""
        if guidance is None:
            return None
        if guidance.ndim != 4 or guidance.shape[0] != B or guidance.shape[2:] != (H, W):
            raise ValueError(
                f"{name} must have shape (B, C, H, W) with B/H/W={(B, H, W)}, got {tuple(guidance.shape)}"
            )
        if guidance.shape[1] == 1:
            return ordered_scan(guidance, orders) if oriented_scan else CrossScan.apply(guidance)
        if not oriented_scan:
            raise ValueError(f"multi-family {name} requires an oriented scan mode")
        family_count = int(family_ids.max().item()) + 1
        if guidance.shape[1] != family_count:
            raise ValueError(f"{name} needs {family_count} scan-family channels, got {guidance.shape[1]}")
        selected = guidance[:, family_ids].flatten(2)
        selected = selected.gather(-1, orders[None].expand(B, -1, -1))
        return selected[:, :, None]

    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)
    if delta_guidance is not None:
        if delta_alpha is None:
            raise ValueError("delta_alpha is required when delta_guidance is provided")
        if delta_guidance.shape != (B, 1, H, W):
            raise ValueError(
                f"delta_guidance must have shape {(B, 1, H, W)}, got {tuple(delta_guidance.shape)}"
            )
        # Match the guidance positions to all four Cross Scan directions before
        # broadcasting over the per-direction dt channels. The modulation is
        # applied to the pre-softplus logits:
        # V1 uses center=1.0: alpha * (1 - g).
        # V2 uses center=0.5: alpha * (0.5 - g), so likely-crack
        # locations decrease delta while likely-background locations increase it.
        delta_scan = guidance_scan(delta_guidance, "delta_guidance").to(dtype=dts.dtype)
        dts = dts + delta_alpha.to(dtype=dts.dtype) * (delta_guidance_center - delta_scan)
    if transition_guidance is not None:
        if transition_alpha is None:
            raise ValueError("transition_alpha is required when transition_guidance is provided")
        edge_scan = guidance_scan(transition_guidance, "transition_guidance").to(dtype=dts.dtype)
        # Low crack-edge confidence increases the positive delta logit, causing
        # faster decay before irrelevant background state can cross the edge.
        dts = dts + transition_alpha.to(dtype=dts.dtype) * (1.0 - edge_scan)
    if write_guidance is not None:
        if write_beta is None:
            raise ValueError("write_beta is required when write_guidance is provided")
        write_scan = guidance_scan(write_guidance, "write_guidance")
        Bs = Bs * (1.0 + write_beta.to(dtype=Bs.dtype) * write_scan.to(dtype=Bs.dtype))
    xs = xs.view(B, -1, L)
    dts = dts.contiguous().view(B, -1, L)
    # HiPPO matrix
    As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
    Bs = Bs.contiguous()
    Cs = Cs.contiguous()
    Ds = Ds.to(torch.float)  # (K * c)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)

    if force_fp32:
        xs = xs.to(torch.float)
        dts = dts.to(torch.float)
        Bs = Bs.to(torch.float)
        Cs = Cs.to(torch.float)

    ys: torch.Tensor = selective_scan(
        xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
    ).view(B, K, -1, H, W)

    if oriented_scan:
        y = ordered_merge(ys, inverse_orders, family_ids, direction_weights)
    else:
        y: torch.Tensor = CrossMerge.apply(ys)

    if out_norm_shape in ["v1"]:  # (B, C, H, W)
        y = out_norm(y.view(B, -1, H, W)).permute(0, 2, 3, 1)  # (B, H, W, C)
    else:  # (B, L, C)
        y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
        y = out_norm(y).view(B, H, W, -1)

    return (y.to(x.dtype) if to_dtype else y)
