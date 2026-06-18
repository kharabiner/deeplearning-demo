"""
reframe_core.py — OpenReframe 의 3D 워핑 엔진

iOS 27 "Spatial Reframing" 재현의 핵심: 단일 이미지 + 뎁스맵으로
가상 카메라를 살짝 움직였을 때의 새 시점을 합성한다.

방식 (무거운 3D 라이브러리 없이 torch/numpy 만 사용 → CUDA/CPU/MPS 어디서나 재현):
  1. 뎁스맵(inverse depth) → 정규화 disparity → 실제 z(거리) 복원
  2. 각 픽셀을 핀홀 카메라로 3D 역투영(unproject)
  3. 가상 카메라 외부 파라미터(R, t)를 적용해 점들을 이동
  4. 다시 투영(project)하고 forward-splat 으로 타깃 이미지에 그림
     - 멀리 있는 점부터 그려서 가까운 점이 덮어쓰게 함(painter's algorithm)
     - 어디에도 그려지지 않은 픽셀 = 디오클루전(가려졌다 드러난 빈 곳) = 인페인팅 대상

좌표계: OpenGL/CV 혼용을 피하기 위해 단순 핀홀 사용.
  - u(가로): 오른쪽 +, v(세로): 아래쪽 +
  - z: 카메라에서 멀어질수록 +
  - 카메라 이동 t = (truck=오른쪽, pedestal=아래, dolly=앞으로)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch


# ── 카메라 파라미터 ─────────────────────────────────────────────────────────────
@dataclass
class CameraMove:
    """
    가상 카메라의 상대 이동/회전. 전부 0이면 원본 시점.
    UI 의 드래그/슬라이더가 이 값을 채운다.

    회전은 장면 중심(pivot, 보통 median depth) 기준으로 돌아서
    "피사체를 바라보며 옆으로 도는" 느낌을 준다.
    """
    yaw_deg: float = 0.0      # 좌우로 도는 각도 (+: 오른쪽으로 돌아봄)
    pitch_deg: float = 0.0    # 위아래로 도는 각도 (+: 위에서 내려봄)
    truck: float = 0.0        # 좌우 평행이동 (+: 오른쪽) — scene 단위 비율
    pedestal: float = 0.0     # 상하 평행이동 (+: 아래)
    dolly: float = 0.0        # 전후 이동 (+: 앞으로 다가감)


def _rotation_matrix(yaw_deg: float, pitch_deg: float, device, dtype) -> torch.Tensor:
    """yaw(Y축) → pitch(X축) 순서의 3x3 회전 행렬."""
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)

    cy, sy = np.cos(yaw), np.sin(yaw)
    cx, sx = np.cos(pitch), np.sin(pitch)

    Ry = torch.tensor(
        [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]],
        device=device, dtype=dtype,
    )
    Rx = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]],
        device=device, dtype=dtype,
    )
    return Rx @ Ry


def rotation_matrix(yaw_deg: float, pitch_deg: float, device, dtype) -> torch.Tensor:
    """공개 API — splat_render 등에서 동일 yaw/pitch 규약으로 사용."""
    return _rotation_matrix(yaw_deg, pitch_deg, device, dtype)


# ── 뎁스 → disparity → z ────────────────────────────────────────────────────────
def normalize_disparity(depth: np.ndarray) -> np.ndarray:
    """
    Depth Anything V2 출력(값이 클수록 가까움 = inverse depth)을
    [0,1] disparity 로 정규화. 1=가장 가까움, 0=가장 멈.

    1~99 백분위로 클립해 아웃라이어(하늘의 0, 반사광 등)에 강건하게.
    """
    d = depth.astype(np.float32)
    lo, hi = np.percentile(d, 1), np.percentile(d, 99)
    d = np.clip((d - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    return d


def disparity_to_z(disp: torch.Tensor, z_near: float, z_far: float) -> torch.Tensor:
    """
    disparity[0,1] → 실제 거리 z.
    disparity 가 1/z 에 비례한다고 보고 역수 관계로 매핑.
      disp=1 → z=z_near (가까움),  disp=0 → z=z_far (멀음)
    """
    return z_near * z_far / (disp * z_far + (1.0 - disp) * z_near + 1e-6)


def disparity_to_z_np(disp: np.ndarray, z_near: float, z_far: float) -> np.ndarray:
    """disparity_to_z 의 numpy 버전 (빌보드 레이어 계산용)."""
    return z_near * z_far / (disp * z_far + (1.0 - disp) * z_near + 1e-6)


def median_pivot_z(disp: np.ndarray, z_near: float, z_far: float) -> float:
    """장면 중심 깊이(pivot). 배경 워핑과 빌보드 재투영이 같은 값을 써야 정합이 맞음."""
    return float(np.median(disparity_to_z_np(disp, z_near, z_far)))


# ── 카메라 헬퍼 (워핑/빌보드 공용) ──────────────────────────────────────────────
def intrinsics(W: int, H: int, fov_deg: float) -> Tuple[float, float, float]:
    """핀홀 (f, cx, cy) 반환."""
    f = 0.5 * W / np.tan(np.deg2rad(fov_deg) * 0.5)
    return f, W * 0.5, H * 0.5


def _rotation_np(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """_rotation_matrix 의 numpy 버전."""
    yaw, pitch = np.deg2rad(yaw_deg), np.deg2rad(pitch_deg)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cx, sx = np.cos(pitch), np.sin(pitch)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    return Rx @ Ry


def reproject_pixels_np(
    u: np.ndarray, v: np.ndarray, z: np.ndarray,
    move: "CameraMove", pivot_z: float, f: float, cx: float, cy: float,
):
    """
    원본 픽셀(u,v) + 거리 z 의 점들을 새 카메라 시점으로 재투영.
    빌보드(사람 평면)의 중심 좌표 이동/스케일 계산에 사용.

    Returns: (u', v', z')  각각 numpy 배열
    """
    u = np.asarray(u, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    z = np.asarray(z, dtype=np.float32)

    X = (u - cx) / f * z
    Y = (v - cy) / f * z
    P = np.stack([X, Y, z], axis=0)                       # (3, N)

    R = _rotation_np(move.yaw_deg, move.pitch_deg)
    pivot = np.array([[0.0], [0.0], [pivot_z]], dtype=np.float32)
    t = np.array(
        [[move.truck * pivot_z], [move.pedestal * pivot_z], [-move.dolly * pivot_z]],
        dtype=np.float32,
    )
    Pn = R @ (P - pivot) + pivot + t                     # (3, N)
    eps = 1e-6
    u2 = f * Pn[0] / (Pn[2] + eps) + cx
    v2 = f * Pn[1] / (Pn[2] + eps) + cy
    return u2, v2, Pn[2]


# ── backward(연결된 늘어남) 워핑 헬퍼 ───────────────────────────────────────────
def _fill_flow(flow: torch.Tensor, written: torch.Tensor, iters: int = 64) -> torch.Tensor:
    """
    타깃 공간 displacement field 의 빈 곳을 이웃 평균으로 반복 채움(flood-fill).
    displacement 는 공간적으로 매끄러우므로 이렇게 채워도 텍스처 우글거림이 없다.

    flow: (2, H, W),  written: (H, W) bool
    """
    import torch.nn.functional as F

    f = flow.clone()
    w = written.float().unsqueeze(0).unsqueeze(0)          # (1,1,H,W)
    fx = f.unsqueeze(0)                                     # (1,2,H,W)
    kernel_sum = lambda x: F.avg_pool2d(x, 3, 1, 1) * 9.0

    for _ in range(iters):
        if w.min() > 0:
            break
        neigh_sum = kernel_sum(fx * w)                      # (1,2,H,W)
        cnt = kernel_sum(w)                                 # (1,1,H,W)
        fill = neigh_sum / (cnt + 1e-6)
        new = (w < 0.5) & (cnt > 0.5)
        fx = torch.where(new.expand_as(fx), fill, fx)
        w = torch.where(new, torch.ones_like(w), w)

    return fx.squeeze(0)                                    # (2,H,W)


def _warp_backward(img, u_src, v_src, u_proj, v_proj, Zn, valid, H, W, frame_zoom=1.0):
    """
    forward displacement 를 타깃 공간으로 z-buffer splat 후 빈 곳을 메우고,
    grid_sample 로 역방향 샘플링 → 구멍 없는 연속 워핑(고무판 늘어남).

    frame_zoom>1 이면 화면을 축소(Extend) → 원본 프레임 밖을 더 많이 노출.

    Returns:
        warped: uint8 NumPy (H, W, C)
        hole:   bool  NumPy (H, W) — splat 으로 안 채워진 디오클루전(안쪽 '늘어난' 자리).
                LDI 합성에서 여기에 배경 플레이트를 덮어 늘어남을 가린다.
        outer:  bool  NumPy (H, W) — 원본 프레임 '밖'을 샘플한 곳(바깥 영역).
                드래그 중엔 블러, 완료 시 아웃페인팅 대상.
    """
    import torch.nn.functional as F

    device = img.device
    C = img.shape[2]
    du = u_proj - u_src
    dv = v_proj - v_src

    order = torch.argsort(Zn, descending=True)
    order = order[valid[order]]
    du_o, dv_o = du[order], dv[order]
    u_o, v_o = u_proj[order], v_proj[order]

    tgt = torch.zeros((2, H * W), device=device, dtype=torch.float32)
    written = torch.zeros(H * W, device=device, dtype=torch.bool)

    # 변위를 반경 r 블록으로 splat(가까운 점이 덮어씀) → 가파른 면의 speckle 억제
    r = 2
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            ui = torch.round(u_o + dx).long()
            vi = torch.round(v_o + dy).long()
            m = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
            lin = vi[m] * W + ui[m]
            tgt[0, lin] = du_o[m]
            tgt[1, lin] = dv_o[m]
            written[lin] = True

    flow = tgt.reshape(2, H, W)
    flow = _fill_flow(flow, written.reshape(H, W))

    # backward 좌표 = 타깃격자 - displacement
    vv, uu = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    src_x = uu - flow[0]
    src_y = vv - flow[1]

    # Extend: 중심 기준 확대 샘플 → 프레임 밖이 화면 안으로 들어옴
    if frame_zoom != 1.0:
        src_x = (W - 1) * 0.5 + (src_x - (W - 1) * 0.5) * frame_zoom
        src_y = (H - 1) * 0.5 + (src_y - (H - 1) * 0.5) * frame_zoom

    # 원본 프레임 밖을 샘플하는 픽셀 = 바깥 영역(아웃페인팅 대상)
    outer = ((src_x < 0) | (src_x > W - 1) | (src_y < 0) | (src_y > H - 1))

    gx = (src_x / (W - 1)) * 2 - 1
    gy = (src_y / (H - 1)) * 2 - 1
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)      # (1,H,W,2)

    src = img.permute(2, 0, 1).unsqueeze(0)                # (1,C,H,W)
    out = F.grid_sample(src, grid, mode="bilinear",
                        padding_mode="border", align_corners=True)
    warped = out.squeeze(0).permute(1, 2, 0).clamp(0, 255).to(torch.uint8).cpu().numpy()
    hole = (~written).reshape(H, W).cpu().numpy()
    outer_np = outer.cpu().numpy()
    return warped, hole, outer_np


# ── 메인 워핑 ───────────────────────────────────────────────────────────────────
def warp_image(
    image: np.ndarray,
    disparity: np.ndarray,
    move: CameraMove,
    *,
    fov_deg: float = 55.0,
    z_near: float = 1.0,
    z_far: float = 6.0,
    splat_radius: int = 1,
    pivot_z: Optional[float] = None,
    smooth: bool = False,
    frame_zoom: float = 1.0,
    return_outer: bool = False,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, ...]:
    """
    단일 이미지를 새 가상 카메라 시점으로 워핑.

    Args:
        image: uint8 NumPy (H, W, 3)
        disparity: float32 NumPy (H, W), [0,1] (normalize_disparity 결과)
        move: 카메라 이동/회전
        fov_deg: 가상 핀홀 수평 화각 가정
        z_near, z_far: disparity → z 매핑 범위 (상대값, 패럴랙스 세기 조절)
        splat_radius: 점을 몇 픽셀 블록으로 그릴지 (1=3x3) — 크랙(점 사이 틈) 감소
        smooth: True = backward grid_sample(연결된 늘어남, 구멍 없음) — 실시간 프리뷰용,
                시간적으로 안정적이라 "우글거림" 없음.
                False = forward splat(디오클루전 구멍 발생) — 확정+인페인팅용.
        device: 연산 device (None이면 자동)

    Returns:
        warped: uint8 NumPy (H, W, 3) — 새 시점 이미지
        hole_mask: bool NumPy (H, W) — True = 디오클루전(인페인팅/배경합성 대상).
                   smooth=True 여도 늘어난(splat 안된) 자리를 hole 로 표시 → LDI 합성에 사용.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # MPS는 일부 scatter 연산이 불안정 → CPU로 폴백
    if device == "mps":
        device = "cpu"

    dtype = torch.float32
    H, W = disparity.shape

    img = torch.from_numpy(image.astype(np.float32)).to(device)          # (H,W,3)
    disp = torch.from_numpy(disparity.astype(np.float32)).to(device)     # (H,W)

    # 핀홀 내부 파라미터
    f = 0.5 * W / np.tan(np.deg2rad(fov_deg) * 0.5)
    cx, cy = W * 0.5, H * 0.5

    # 픽셀 격자 → 3D 역투영
    vv, uu = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    z = disparity_to_z(disp, z_near, z_far)                # (H,W)
    X = (uu - cx) / f * z
    Y = (vv - cy) / f * z
    pts = torch.stack([X, Y, z], dim=-1).reshape(-1, 3).T  # (3, N)

    # pivot(장면 중심 깊이) 기준 회전 + 평행이동
    if pivot_z is None:
        pivot_z = float(torch.median(z).item())
    pivot = torch.tensor([0.0, 0.0, pivot_z], device=device, dtype=dtype).view(3, 1)

    R = _rotation_matrix(move.yaw_deg, move.pitch_deg, device, dtype)
    # 평행이동은 장면 스케일(pivot_z)에 비례시켜 직관적으로
    t = torch.tensor(
        [move.truck * pivot_z, move.pedestal * pivot_z, -move.dolly * pivot_z],
        device=device, dtype=dtype,
    ).view(3, 1)

    pts_new = R @ (pts - pivot) + pivot + t                # (3, N)
    Xn, Yn, Zn = pts_new[0], pts_new[1], pts_new[2]

    # 재투영
    eps = 1e-6
    u_proj = f * Xn / (Zn + eps) + cx
    v_proj = f * Yn / (Zn + eps) + cy

    colors = img.reshape(-1, 3)                            # (N, 3)

    # 유효: 카메라 앞(Zn>0) + 화면 안
    valid = (Zn > eps) & (u_proj >= 0) & (u_proj < W) & (v_proj >= 0) & (v_proj < H)

    # ── smooth(backward) 모드: 연결된 늘어남(프리뷰 안정). hole=늘어난 자리(LDI용) ──
    if smooth:
        warped, hole, outer = _warp_backward(
            img, uu.reshape(-1), vv.reshape(-1), u_proj, v_proj, Zn, valid, H, W,
            frame_zoom=frame_zoom,
        )
        if return_outer:
            return warped, hole, outer
        return warped, hole

    # painter's algorithm: 먼 점(Zn 큰 것)부터 그려 가까운 점이 덮어쓰게
    order = torch.argsort(Zn, descending=True)
    order = order[valid[order]]

    u_o = u_proj[order]
    v_o = v_proj[order]
    c_o = colors[order]

    target = torch.zeros((H * W, 3), device=device, dtype=dtype)
    written = torch.zeros(H * W, device=device, dtype=torch.bool)

    r = splat_radius
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            uu_i = torch.round(u_o + dx).long()
            vv_i = torch.round(v_o + dy).long()
            m = (uu_i >= 0) & (uu_i < W) & (vv_i >= 0) & (vv_i < H)
            lin = vv_i[m] * W + uu_i[m]
            target[lin] = c_o[m]
            written[lin] = True

    warped = target.reshape(H, W, 3).clamp(0, 255).to(torch.uint8).cpu().numpy()
    hole_mask = (~written).reshape(H, W).cpu().numpy()
    return warped, hole_mask


def composite_photo_anchor(
    image: np.ndarray,
    disparity: np.ndarray,
    yaw: float,
    pitch: float,
    splat_rgb: np.ndarray,
    _splat_cov: np.ndarray,
    z_near: float = 1.0,
    z_far: float = 6.0,
    device: Optional[str] = None,
) -> np.ndarray:
    """원본 사진 픽셀을 depth warp로 유지하고, 디오클루전만 SHARP splat으로 채움.

    SHARP splat 전체를 쓰면 ~60만 Gaussian 재합성이라 원본 대비 선명도·색이 떨어진다.
    보이던 표면은 사진을 3D로 재투영하고, 가려졌다 드러난 곳만 splat을 쓴다.
    """
    if abs(float(yaw)) < 0.5 and abs(float(pitch)) < 0.5:
        return image
    move = CameraMove(yaw_deg=float(yaw), pitch_deg=float(pitch))
    warped, hole = warp_image(
        image, disparity, move, z_near=z_near, z_far=z_far,
        smooth=False, device=device,
    )
    if not hole.any():
        return warped
    out = warped.copy()
    out[hole] = splat_rgb[hole]
    return out


def preview_holes(hole_mask: np.ndarray) -> np.ndarray:
    """프리뷰용: 이미지 테두리와 연결된 구멍만 (내부 점박이는 블러하지 않음)."""
    try:
        import cv2
    except Exception:
        return hole_mask

    H, W = hole_mask.shape
    m = hole_mask.astype(np.uint8)
    if m.sum() == 0:
        return hole_mask
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    border = np.zeros_like(m)
    for i in range(1, n):
        x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if (x == 0) or (y == 0) or (x + w >= W) or (y + h >= H):
            border[labels == i] = 1
    return border.astype(bool)


# ── 프리뷰용 빠른 빈칸 채움 (애플의 "드래그 중 블러") ───────────────────────────
def fill_preview(warped: np.ndarray, hole_mask: np.ndarray, blur_sigma: float = 12.0) -> np.ndarray:
    """
    드래그 중 바깥(테두리) 구멍만 뿌옇게 — Telea 인페인팅은 주변까지 번져 전체가 흐려짐.
    구멍 픽셀만 블러 색으로 교체해 중앙(피사체) 선명도 유지.
    """
    if not hole_mask.any():
        return warped
    out = warped.copy()
    try:
        import cv2

        soft = cv2.GaussianBlur(warped, (0, 0), blur_sigma)
        out[hole_mask] = soft[hole_mask]
        return out
    except Exception:
        k = 11
        pad = k // 2
        padded = np.pad(out, ((pad, pad), (pad, pad), (0, 0)), mode="edge").astype(np.float32)
        acc = np.zeros_like(out, dtype=np.float32)
        for dy in range(k):
            for dx in range(k):
                acc += padded[dy:dy + out.shape[0], dx:dx + out.shape[1]]
        blurred = (acc / (k * k)).astype(np.uint8)
        out[hole_mask] = blurred[hole_mask]
        return out


def sharpen_rgb(rgb: np.ndarray, sigma: float = 0.8, amount: float = 0.5) -> np.ndarray:
    """표시용 경미한 언샤프(렌더 softening 보정)."""
    try:
        import cv2
    except Exception:
        return rgb
    f = rgb.astype(np.float32)
    blur = cv2.GaussianBlur(rgb, (0, 0), sigma)
    return np.clip(f + amount * (f - blur.astype(np.float32)), 0, 255).astype(np.uint8)


def dilate_mask(mask: np.ndarray, iterations: int = 2) -> np.ndarray:
    """인페인팅 경계를 깔끔하게 하려 hole 마스크를 살짝 팽창."""
    try:
        import cv2

        kernel = np.ones((3, 3), np.uint8)
        return cv2.dilate(mask.astype(np.uint8), kernel, iterations=iterations).astype(bool)
    except Exception:
        return mask


# ── 단독 실행: 샘플로 좌우 스윙 프레임 몇 장 저장 (빠른 동작 확인용) ────────────
if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from common import load_image, pil_to_numpy, get_device, save_result
    import task_depth_depthanythingv2 as task_depth

    parser = argparse.ArgumentParser(description="OpenReframe core — quick warp test")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--yaw", type=float, default=8.0, help="좌우 스윙 최대 각도")
    parser.add_argument("--frames", type=int, default=5)
    args = parser.parse_args()

    device = get_device()
    image = load_image(args.image)
    img_np = pil_to_numpy(image)

    print("[reframe] depth 추정 중...")
    processor, model = task_depth.load_model(device)
    depth = task_depth.run(image, processor, model, device)
    disp = normalize_disparity(depth)

    stem = Path(args.image).stem
    for i in range(args.frames):
        # -yaw → +yaw 스윙
        a = -args.yaw + (2 * args.yaw) * i / max(1, args.frames - 1)
        warped, hole = warp_image(img_np, disp, CameraMove(yaw_deg=a), device=device)
        preview = fill_preview(warped, hole)
        save_result(preview, f"{stem}_reframe_yaw{a:+.1f}.png")
    print("[reframe] 완료 → outputs/ 확인")
