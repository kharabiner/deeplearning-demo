"""
splat_render.py — 순수 PyTorch 3D Gaussian Splat 렌더러

SHARP(task_nvs_sharp.py)가 만든 SharpScene(3D Gaussians)을 임의 카메라 시점으로 그린다.
gsplat(CUDA 전용, Windows 빌드 필요) 대신 torch만 사용 → CUDA/MPS/CPU 어디서나 동작.

실시간 100fps를 노리지 않는다. Reframe은 각도 범위가 좁으므로
"분석(대기)" 단계에서 각도 격자를 몇 십 장 미리 렌더해 캐시하고,
드래그는 최근접 캐시를 즉시 보여준다(iOS Reframe과 동일한 UX).
따라서 렌더러는 "오프라인·정확" 위주(프레임당 수백 ms 허용).

알고리즘 (2-pass z-buffer splatting):
  1. Gaussian 중심을 카메라 이동(R·t)으로 변환 후 핀홀 투영
  2. 각 Gaussian을 화면에서 (월드 크기·거리 비례) 반경 r_px 원반으로 splat
  3. pass A: 픽셀별 최소 깊이(zbuf) 계산 (가까운 면이 이김 = 가림 처리)
  4. pass B: zbuf와 일치하는(=가장 앞) Gaussian 색을 픽셀에 기록
  5. linearRGB 합성 → sRGB 변환

좌표계/카메라 규약은 reframe_core.CameraMove 와 동일(yaw/pitch + truck/pedestal/dolly,
pivot 중심 회전)하므로 기존 UI 슬라이더를 그대로 쓸 수 있다.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

import common
from reframe_core import CameraMove, rotation_matrix
from task_nvs_sharp import SharpScene


def _rotation(yaw_deg: float, pitch_deg: float, device, dtype) -> torch.Tensor:
    return rotation_matrix(yaw_deg, pitch_deg, device, dtype)


def _linear_to_srgb(c: torch.Tensor) -> torch.Tensor:
    """linearRGB[0,1] → sRGB[0,1]. SHARP 색은 linear 이므로 표시 전 변환."""
    c = c.clamp(0.0, 1.0)
    return torch.where(c <= 0.0031308, 12.92 * c, 1.055 * c.clamp(min=1e-8) ** (1 / 2.4) - 0.055)


def _polish_disocclusion(
    rgb: np.ndarray,
    coverage: np.ndarray,
    alpha: np.ndarray,
    *,
    alpha_thresh: float = 0.32,
    dark_delta: float = 22.0,
    max_area: int = 1200,
    min_area: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """디오클루전 경계의 작은 어두운 점박이만 주변색으로 정리.

    큰 영역(벽·그림자)은 건드리지 않는다.
    """
    try:
        import cv2
    except Exception:
        return rgb, coverage

    H, W = coverage.shape
    margin = max(12, min(H, W) // 16)
    interior = np.ones((H, W), dtype=bool)
    interior[:margin, :] = interior[-margin:, :] = False
    interior[:, :margin] = interior[:, -margin:] = False

    weak = interior & coverage & (alpha < alpha_thresh)
    if weak.any():
        smooth = cv2.bilateralFilter(rgb, 9, 40, 40)
        wf = cv2.GaussianBlur(weak.astype(np.float32), (0, 0), 2.0)
        wf = np.clip(wf * 0.8, 0, 1)[..., None]
        rgb = np.clip(rgb.astype(np.float32) * (1 - wf) + smooth.astype(np.float32) * wf,
                      0, 255).astype(np.uint8)

    gray_u8 = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = gray_u8.astype(np.float32)
    med = cv2.medianBlur(gray_u8, 7).astype(np.float32)
    # 아주 작은 어두운 점만 (큰 그림자 제외)
    dark = interior & coverage & (gray + dark_delta < med)
    dark = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)

    mask = (weak | dark).astype(np.uint8)
    if mask.sum() < min_area:
        return rgb, coverage

    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    keep = np.zeros_like(mask)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            keep[labels == i] = 1
    if keep.sum() < min_area:
        return rgb, coverage

    polished = cv2.inpaint(rgb, keep * 255, 3, cv2.INPAINT_TELEA)
    cov = coverage.copy()
    cov[keep.astype(bool)] = True
    return polished, cov


def _trim_disocclusion_coverage(coverage: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """테두리(회전 시 새로 드러난 영역)만 미커버 처리.

    내부(얼굴·피사체)의 약한 알파까지 구멍으로 치면 프리뷰 블러가
    전체 사진을 흐리게 만든다 → 테두리만 제외.
    """
    H, W = coverage.shape
    margin = max(16, min(H, W) // 20)
    border = np.zeros((H, W), dtype=bool)
    border[:margin, :] = border[-margin:, :] = True
    border[:, :margin] = border[:, -margin:] = True
    bad = border & coverage & (alpha < 0.42)
    cov = coverage.copy()
    cov[bad] = False
    return cov


def _close_small_holes(rgb: np.ndarray, coverage: np.ndarray):
    """내부 구멍(occlusion 경계 점박이/디오클루전)은 주변색으로 메우고, 이미지
    테두리에 닿는 '바깥 영역'만 구멍으로 남긴다(→ 앱에서 블러/아웃페인팅).

    내부 vs 바깥은 연결요소가 이미지 경계에 닿는지로 판별.

    Returns: (rgb_filled, coverage_updated)
    """
    try:
        import cv2
    except Exception:
        return rgb, coverage

    H, W = coverage.shape
    holes = (~coverage).astype(np.uint8)
    if holes.sum() == 0:
        return rgb, coverage

    # 점박이를 솔리드로 모은 뒤 연결요소 판별(틈 사이 메움이 깔끔)
    closed = cv2.morphologyEx(holes, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)

    interior = np.zeros_like(holes)
    for i in range(1, n):
        x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        touches_border = (x == 0) or (y == 0) or (x + w >= W) or (y + h >= H)
        if not touches_border:
            interior[labels == i] = 1

    if interior.sum() == 0:
        return rgb, coverage

    fill_mask = cv2.dilate(interior, np.ones((3, 3), np.uint8), iterations=1)
    filled = cv2.inpaint(rgb, fill_mask * 255, 4, cv2.INPAINT_TELEA)
    cov = coverage.copy()
    cov[fill_mask.astype(bool)] = True
    return filled, cov


@torch.no_grad()
def render(
    scene: SharpScene,
    move: CameraMove,
    *,
    out_hw: Optional[Tuple[int, int]] = None,
    pivot_z: Optional[float] = None,
    size_boost: float = 1.6,
    max_radius: int = 5,
    opacity_thresh: float = 0.02,
    depth_band: float = 0.05,
    aa_blur: float = 0.5,
    close_holes: bool = True,
    polish: bool = False,
    supersample: int = 1,
    trim_coverage: bool = False,
    coverage_thresh: float = 1e-6,
    return_depth: bool = False,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, ...]:
    """SharpScene 을 새 시점으로 렌더 (비등방성 EWA alpha-blending splatting).

    진짜 3DGS(gsplat)와 동일 원리:
      - 각 Gaussian의 3D 공분산(쿼터니언·스케일)을 카메라로 회전 후 2D로 투영
        (퍼스펙티브 야코비안) → 화면에서 '타원' footprint(conic)을 얻음.
      - 그 타원 가우시안 가중치로 색을 누적하고, 깊이로 게이팅해 '앞면'만 섞음.
    등방성(원형) 근사와 달리, 경계/비스듬한 면의 Gaussian이 길쭉하게 늘어나
    연속적으로 덮으므로 점박이(샘플링 빈틈)가 사라지고 매끄럽다.

    Args:
        size_boost: 공분산 스케일 게인(틈 메움 ↔ 부드러움).
        max_radius: footprint 최대 반경(px) — 성능 제한.
        opacity_thresh: 이보다 투명한 Gaussian 무시.
        depth_band: 앞면 인정 상대 깊이 폭(작을수록 가림 또렷).
        aa_blur: 화면공간 안티에일리어싱 블러(2D 공분산에 더할 분산, px²).

    Returns:
        rgb: uint8 NumPy (H, W, 3) sRGB
        coverage: bool NumPy (H, W) — True=그려짐. ~coverage = 빈 곳(바깥/구멍).
    """
    device = device or common.get_device()
    if device == "mps":   # MPS scatter 일부 불안정 → CPU 폴백(오프라인 렌더라 OK)
        device = "cpu"
    dtype = torch.float32

    s = scene.to(device)
    means = s.means.to(dtype)              # (N,3)
    scales = s.scales.to(dtype) * float(size_boost)  # (N,3)
    quats = s.quats.to(dtype)              # (N,4) w-first
    colors = s.colors.to(dtype)            # (N,3) linear
    opac = s.opacities.to(dtype).reshape(-1)

    # 출력 해상도 (원본 종횡비 유지)
    if out_hw is None:
        long_side = 768
        if s.width >= s.height:
            W, H = long_side, max(1, round(long_side * s.height / s.width))
        else:
            H, W = long_side, max(1, round(long_side * s.width / s.height))
    else:
        H, W = int(out_hw[0]), int(out_hw[1])

    ss = max(1, int(supersample))
    Ho, Wo = H, W
    if ss > 1:
        H, W = H * ss, W * ss

    f = s.f_px * (W / s.width)
    cx, cy = W * 0.5, H * 0.5
    eps = 1e-6

    # ── 카메라 이동 (reframe_core.warp_image 규약과 동일) ──
    if pivot_z is None:
        pivot_z = float(torch.median(means[:, 2]).item())
    pivot = torch.tensor([0.0, 0.0, pivot_z], device=device, dtype=dtype).view(1, 3)
    R = _rotation(move.yaw_deg, move.pitch_deg, device, dtype)
    t = torch.tensor(
        [move.truck * pivot_z, move.pedestal * pivot_z, -move.dolly * pivot_z],
        device=device, dtype=dtype,
    ).view(1, 3)

    pts = (means - pivot) @ R.T + pivot + t      # (N,3)
    X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]
    u = f * X / (Z + eps) + cx
    v = f * Y / (Z + eps) + cy

    # ── 3D 공분산 → 카메라 회전 → 2D 투영(EWA) ──
    from sharp.utils.gaussians import compose_covariance_matrices

    Sig_w = compose_covariance_matrices(quats, scales)            # (N,3,3) 월드
    Sig_c = torch.einsum("ij,njk->nik", R, Sig_w)
    Sig_c = torch.einsum("nik,lk->nil", Sig_c, R)                 # R Σ Rᵀ
    invZ = 1.0 / (Z + eps)
    N = means.shape[0]
    J = torch.zeros((N, 2, 3), device=device, dtype=dtype)        # 퍼스펙티브 야코비안
    J[:, 0, 0] = f * invZ
    J[:, 0, 2] = -f * X * invZ * invZ
    J[:, 1, 1] = f * invZ
    J[:, 1, 2] = -f * Y * invZ * invZ
    Sig2 = J.bmm(Sig_c).bmm(J.transpose(1, 2))                    # (N,2,2)
    a = Sig2[:, 0, 0] + aa_blur
    b = Sig2[:, 0, 1]
    c = Sig2[:, 1, 1] + aa_blur
    det = (a * c - b * b).clamp(min=1e-6)
    # conic = Σ2⁻¹ = [[c,-b],[-b,a]]/det
    ca, cb, cc = c / det, -b / det, a / det
    # footprint 반경 ≈ 3·√(최대 고유값)
    tr = a + c
    disc = torch.sqrt((tr * tr * 0.25 - det).clamp(min=0.0))
    lam_max = (tr * 0.5 + disc).clamp(min=1e-6)
    rad = (3.0 * torch.sqrt(lam_max)).clamp(0.6, float(max_radius * ss))

    base_valid = (Z > eps) & (opac > opacity_thresh)
    ui0 = torch.round(u).long()
    vi0 = torch.round(v).long()
    band = float(depth_band) * (abs(pivot_z) + 1.0)

    npix = H * W
    INF = torch.finfo(dtype).max
    zbuf = torch.full((npix,), INF, device=device, dtype=dtype)

    rad_cap = int(max_radius * ss)
    offsets = [(dy, dx) for dy in range(-rad_cap, rad_cap + 1)
               for dx in range(-rad_cap, rad_cap + 1)]
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

    # pass A: 픽셀별 앞면 깊이(footprint 포함)
    cache = []
    for (dy, dx) in offsets:
        e = _entries(dy, dx)
        cache.append((e, dx, dy))
        if e is None:
            continue
        lin, idx = e
        zbuf.scatter_reduce_(0, lin, Z[idx], reduce="amin", include_self=True)

    # pass B: 전경 / 배경(앞면뒤) 분리 누적
    #   front : zbuf±band 이내 → 또렷한 전경
    #   back  : zbuf+band 보다 뒤 → 디오클루전 시 순수 배경색
    # 얇은 전경 스플랫과 배경을 한꺼번에 섞으면(col_a) 경계가 어두워지므로 분리한다.
    wsum_f = torch.zeros(npix, device=device, dtype=dtype)
    csum_f = torch.zeros((npix, 3), device=device, dtype=dtype)
    wsum_b = torch.zeros(npix, device=device, dtype=dtype)
    csum_b = torch.zeros((npix, 3), device=device, dtype=dtype)
    wsum_a = torch.zeros(npix, device=device, dtype=dtype)
    zsum_a = torch.zeros(npix, device=device, dtype=dtype)
    for (e, dx, dy) in cache:
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
        zsum_a.index_add_(0, lin, w * z_lin)

    written = wsum_a > float(coverage_thresh)
    col_f = torch.zeros((npix, 3), device=device, dtype=dtype)
    has_f = wsum_f > 1e-6
    col_f[has_f] = csum_f[has_f] / wsum_f[has_f].unsqueeze(1)
    col_b = torch.zeros((npix, 3), device=device, dtype=dtype)
    has_b = wsum_b > 1e-6
    col_b[has_b] = csum_b[has_b] / wsum_b[has_b].unsqueeze(1)
    # 앞면 누적량 → 전경 신뢰도. 약하면 배경 레이어만 사용(어두운 경계 제거).
    alpha = (1.0 - torch.exp(-wsum_f / 0.35)).unsqueeze(1)
    base = torch.where(has_b.unsqueeze(1), col_b, col_f)
    col = torch.where(has_f.unsqueeze(1), alpha * col_f + (1.0 - alpha) * base, base)

    rgb = (_linear_to_srgb(col.reshape(H, W, 3)) * 255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()
    coverage = written.reshape(H, W).cpu().numpy()
    alpha_map = alpha.reshape(H, W).cpu().numpy()
    if polish:
        rgb, coverage = _polish_disocclusion(rgb, coverage, alpha_map)
    if ss > 1:
        import cv2
        rgb = cv2.resize(rgb, (Wo, Ho), interpolation=cv2.INTER_AREA)
        coverage = cv2.resize(coverage.astype(np.uint8), (Wo, Ho), interpolation=cv2.INTER_AREA) > 127
        alpha_map = cv2.resize(alpha_map.astype(np.float32), (Wo, Ho), interpolation=cv2.INTER_AREA)
    coverage = _trim_disocclusion_coverage(coverage, alpha_map) if trim_coverage else coverage
    if return_depth:
        depth_mean = torch.zeros(npix, device=device, dtype=dtype)
        depth_mean[written] = zsum_a[written] / wsum_a[written]
        depth = depth_mean.reshape(H, W).cpu().numpy()
        if close_holes:
            rgb, coverage = _close_small_holes(rgb, coverage)
        return rgb, coverage, depth, alpha_map
    if close_holes:
        rgb, coverage = _close_small_holes(rgb, coverage)
    return rgb, coverage


# ── CLI: sample 로 좌우 스윙 몇 장 렌더 (동작 확인) ───────────────────────────────
if __name__ == "__main__":
    import argparse
    import time

    import task_nvs_sharp

    ap = argparse.ArgumentParser(description="SHARP + torch splat 렌더 테스트")
    ap.add_argument("--image", type=str, default="sample.jpg")
    ap.add_argument("--yaw", type=float, default=10.0)
    ap.add_argument("--frames", type=int, default=5)
    args = ap.parse_args()

    dev = common.get_device()
    print(f"[env] {common.device_info()}")

    t0 = time.time()
    scene = task_nvs_sharp.predict(args.image, device=dev)
    print(f"[sharp] {scene.num_gaussians:,} gaussians in {time.time() - t0:.2f}s")

    stem = args.image.rsplit(".", 1)[0].split("/")[-1].split("\\")[-1]
    for i in range(args.frames):
        a = -args.yaw + (2 * args.yaw) * i / max(1, args.frames - 1)
        t1 = time.time()
        rgb, cov = render(scene, CameraMove(yaw_deg=a), device=dev)
        dt = time.time() - t1
        miss = 100.0 * (1.0 - cov.mean())
        common.save_result(rgb, f"{stem}_sharp_yaw{a:+.1f}.png")
        print(f"  yaw {a:+.1f}: {dt*1000:.0f} ms, hole {miss:.1f}%")
    print("[done] outputs/ 확인")
