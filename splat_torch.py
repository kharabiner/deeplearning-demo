"""
splat_torch.py — PyTorch EWA Gaussian splat (gsplat CUDA 미사용 시 폴백)

Windows + Python 3.11 등 gsplat 사전 빌드 휠/cu124 JIT가 불가한 환경용.
SHARP extrinsics/intrinsics 규약과 동일하게 world→camera 투영 후 2-pass splat.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

from sharp.utils.gaussians import Gaussians3D, compose_covariance_matrices


def _linear_to_srgb(c: torch.Tensor) -> torch.Tensor:
    c = c.clamp(0.0, 1.0)
    return torch.where(
        c <= 0.0031308,
        12.92 * c,
        1.055 * c.clamp(min=1e-8) ** (1 / 2.4) - 0.055,
    )


@torch.no_grad()
def render_gaussians(
    gaussians: Gaussians3D,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    image_width: int,
    image_height: int,
    *,
    device: Optional[torch.device] = None,
    size_boost: float = 1.5,
    max_radius: int = 5,
    opacity_thresh: float = 0.02,
    depth_band: float = 0.05,
    aa_blur: float = 0.5,
    coverage_thresh: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """gsplat GSplatRenderer와 동일한 (rgb uint8, alpha float32) 반환."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    means = gaussians.mean_vectors[0].to(device=device, dtype=dtype)
    scales = gaussians.singular_values[0].to(device=device, dtype=dtype) * float(size_boost)
    quats = gaussians.quaternions[0].to(device=device, dtype=dtype)
    colors = gaussians.colors[0].to(device=device, dtype=dtype)
    opac = gaussians.opacities[0].to(device=device, dtype=dtype).reshape(-1)

    H, W = int(image_height), int(image_width)
    K = intrinsics.to(device=device, dtype=dtype)
    if K.ndim == 3:
        K = K[0]
    E = extrinsics.to(device=device, dtype=dtype)
    if E.ndim == 3:
        E = E[0]

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    R = E[:3, :3]
    t = E[:3, 3]
    pts = means @ R.T + t
    X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]

    eps = 1e-6
    u = fx * X / (Z + eps) + cx
    v = fy * Y / (Z + eps) + cy

    Sig_w = compose_covariance_matrices(quats, scales)
    Sig_c = R.unsqueeze(0) @ Sig_w @ R.T.unsqueeze(0)
    invZ = 1.0 / (Z + eps)
    n = means.shape[0]
    J = torch.zeros((n, 2, 3), device=device, dtype=dtype)
    J[:, 0, 0] = fx * invZ
    J[:, 0, 2] = -fx * X * invZ * invZ
    J[:, 1, 1] = fy * invZ
    J[:, 1, 2] = -fy * Y * invZ * invZ
    Sig2 = J.bmm(Sig_c).bmm(J.transpose(1, 2))
    a = Sig2[:, 0, 0] + aa_blur
    b = Sig2[:, 0, 1]
    c = Sig2[:, 1, 1] + aa_blur
    det = (a * c - b * b).clamp(min=1e-6)
    ca, cb, cc = c / det, -b / det, a / det
    tr = a + c
    disc = torch.sqrt((tr * tr * 0.25 - det).clamp(min=0.0))
    lam_max = (tr * 0.5 + disc).clamp(min=1e-6)
    rad = (3.0 * torch.sqrt(lam_max)).clamp(0.6, float(max_radius))

    base_valid = (Z > eps) & (opac > opacity_thresh)
    ui0 = torch.round(u).long()
    vi0 = torch.round(v).long()
    pivot_z = float(torch.median(Z[base_valid]).item()) if base_valid.any() else 1.0
    band = float(depth_band) * (abs(pivot_z) + 1.0)

    npix = H * W
    inf = torch.finfo(dtype).max
    zbuf = torch.full((npix,), inf, device=device, dtype=dtype)

    rad_cap = int(max_radius)
    offsets = [
        (dy, dx)
        for dy in range(-rad_cap, rad_cap + 1)
        for dx in range(-rad_cap, rad_cap + 1)
    ]
    offsets.sort(key=lambda o: o[0] * o[0] + o[1] * o[1])

    def _entries(dy: int, dx: int):
        rr = (dx * dx + dy * dy) ** 0.5
        m = base_valid & (rad >= rr)
        if not torch.any(m):
            return None
        ui = ui0[m] + dx
        vi = vi0[m] + dy
        inb = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        if not torch.any(inb):
            return None
        idx = torch.nonzero(m, as_tuple=False).squeeze(1)[inb]
        lin = vi[inb] * W + ui[inb]
        return lin, idx

    cache = []
    for dy, dx in offsets:
        e = _entries(dy, dx)
        cache.append((e, dx, dy))
        if e is None:
            continue
        lin, idx = e
        zbuf.scatter_reduce_(0, lin, Z[idx], reduce="amin", include_self=True)

    wsum_f = torch.zeros(npix, device=device, dtype=dtype)
    csum_f = torch.zeros((npix, 3), device=device, dtype=dtype)
    wsum_b = torch.zeros(npix, device=device, dtype=dtype)
    csum_b = torch.zeros((npix, 3), device=device, dtype=dtype)
    wsum_a = torch.zeros(npix, device=device, dtype=dtype)

    for e, dx, dy in cache:
        if e is None:
            continue
        lin, idx = e
        power = -0.5 * (ca[idx] * (dx * dx) + 2.0 * cb[idx] * (dx * dy) + cc[idx] * (dy * dy))
        w = opac[idx] * torch.exp(power.clamp(min=-30.0))
        col_i = colors[idx]
        z_lin = Z[idx]
        z_front = zbuf[lin]
        front_eps = band * 0.35
        wf = torch.where(z_lin <= (z_front + front_eps), w, torch.zeros_like(w))
        wb = torch.where(z_lin > (z_front + front_eps), w, torch.zeros_like(w))
        wsum_f.index_add_(0, lin, wf)
        csum_f.index_add_(0, lin, col_i * wf[:, None])
        wsum_b.index_add_(0, lin, wb)
        csum_b.index_add_(0, lin, col_i * wb[:, None])
        wsum_a.index_add_(0, lin, w)

    written = wsum_a > float(coverage_thresh)
    col_f = torch.zeros((npix, 3), device=device, dtype=dtype)
    has_f = wsum_f > 1e-6
    col_f[has_f] = csum_f[has_f] / wsum_f[has_f].unsqueeze(1)
    col_b = torch.zeros((npix, 3), device=device, dtype=dtype)
    has_b = wsum_b > 1e-6
    col_b[has_b] = csum_b[has_b] / wsum_b[has_b].unsqueeze(1)

    alpha_v = (1.0 - torch.exp(-wsum_f / 0.35)).unsqueeze(1)
    base = torch.where(has_b.unsqueeze(1), col_b, col_f)
    col = torch.where(has_f.unsqueeze(1), alpha_v * col_f + (1.0 - alpha_v) * base, base)

    rgb = (_linear_to_srgb(col.reshape(H, W, 3)) * 255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()
    alpha = alpha_v.reshape(H, W).clamp(0, 1).cpu().numpy().astype(np.float32)
    alpha[~written.reshape(H, W).cpu().numpy()] = 0.0
    return rgb, alpha