"""
reframe_complete.py — 점진적 장면 완성 (가려졌던 면 생성)

SHARP는 '보이는 표면' 위주의 단일이미지 3D라, 회전 시 물체 뒤(디오클루전)에
빈곳이 생긴다. 이를 메우기 위해 몇몇 핵심 시점에서:

  1. 현재 장면을 그 시점으로 렌더 → 내부 디오클루전 구멍 + 렌더 깊이(zbuf) 획득
  2. 구멍을 2D 인페인팅으로 '생성'해 채움 (LaMa/SD)        ← 진짜 생성 단계
  3. 그 구멍 픽셀을 '렌더 깊이(=기존 장면과 동일 metric)'로 3D 역투영
     → 두 번째 SHARP를 쓰지 않아 스케일 불일치/드리프트가 없음
  4. 원본 카메라 좌표로 되돌려 새 Gaussian으로 '병합'

핵심 시점 몇 개(모서리/상하좌우)면 그 각도 범위의 디오클루전을 거의 덮는다.
바깥 프레임(테두리 연결) 구멍은 완성 대상에서 제외(= commit 아웃페인팅 담당).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch

import common
import inpaint as rinp
import splat_render
from reframe_core import CameraMove
from splat_render import _rotation
from task_nvs_sharp import SharpScene


# 기본 핵심 시점(yaw, pitch). 모서리 + 상하좌우.
DEFAULT_VIEWS = [
    (-16, 0), (16, 0), (0, -10), (0, 10),
    (-16, -10), (16, -10), (-16, 10), (16, 10),
]


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = c.astype(np.float32) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _interior_region(mask: np.ndarray) -> np.ndarray:
    """테두리(바깥 프레임)에 닿는 연결요소를 뺀, 내부 영역만 True.

    바깥 프레임 구멍은 commit 아웃페인팅 담당이므로 완성에서 제외한다.
    """
    try:
        import cv2
    except Exception:
        return mask

    H, W = mask.shape
    m = mask.astype(np.uint8)
    if m.sum() == 0:
        return m.astype(bool)
    closed = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    interior = np.zeros_like(m)
    for i in range(1, n):
        x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if not ((x == 0) or (y == 0) or (x + w >= W) or (y + h >= H)):
            interior[labels == i] = 1
    return (interior & m).astype(bool)


def _interior_holes(coverage: np.ndarray) -> np.ndarray:
    """테두리에 닿지 않는 내부 구멍만 True(= 디오클루전). 바깥 프레임은 제외."""
    return _interior_region(~coverage)


def _fill_depth_bg(depth: np.ndarray, hole: np.ndarray, iters: int = 256) -> np.ndarray:
    """구멍 깊이를 '배경(더 먼 쪽)'으로 채움 — max 팽창을 반복 전파.

    디오클루전 = 가려졌던 배경이므로, 주변에서 가장 먼 깊이를 흘려넣는다.
    """
    try:
        import cv2
    except Exception:
        return depth
    d = depth.astype(np.float32).copy()
    m = hole.copy()
    k = np.ones((3, 3), np.uint8)
    for _ in range(iters):
        if not m.any():
            break
        dil = cv2.dilate(d, k)            # 3x3 max
        fill = m & (dil > 0)
        d[fill] = dil[fill]
        m = m & ~fill
    return d


@torch.no_grad()
def complete_scene(
    scene: SharpScene,
    *,
    views: Optional[List] = None,
    backend: str = "lama",
    out_long: int = 640,
    size_boost: float = 2.2,
    alpha_thresh: float = 0.35,
    new_scale_gain: float = 1.6,
    device: Optional[str] = None,
    progress=None,
) -> SharpScene:
    """핵심 시점들에서 디오클루전을 생성·병합한 보강 장면을 반환."""
    device = device or common.get_device()
    views = views or DEFAULT_VIEWS
    dtype = torch.float32
    s = scene.to(device)

    # 출력 해상도 / 내부 파라미터
    if s.width >= s.height:
        W, H = out_long, max(1, round(out_long * s.height / s.width))
    else:
        H, W = out_long, max(1, round(out_long * s.width / s.height))
    f = s.f_px * (W / s.width)
    cx, cy = W * 0.5, H * 0.5

    pivot_z = float(torch.median(s.means[:, 2]).item())
    pivot = torch.tensor([0.0, 0.0, pivot_z], device=device, dtype=dtype)

    inp = rinp.get_inpainter(backend, device)

    add_means, add_scales, add_quats, add_colors, add_opac = [], [], [], [], []
    total = len(views)
    for vi, (yaw, pitch) in enumerate(views):
        move = CameraMove(yaw_deg=float(yaw), pitch_deg=float(pitch))
        rgb, cov, depth, alpha = splat_render.render(
            s, move, out_hw=(H, W), pivot_z=pivot_z, size_boost=size_boost,
            close_holes=False, polish=False, return_depth=True, device=device,
        )
        # 채울 대상 = 내부 구멍(완전 빈곳) + 디오클루전(덮였지만 앞면 약함=배경만 보임)
        holes = _interior_holes(cov)
        disocc = cov & (alpha < alpha_thresh)
        fill = _interior_region(holes | disocc)
        n_hole = int(fill.sum())
        if progress is not None:
            progress(0.55 + 0.4 * vi / total,
                     desc=f"빈곳 생성 {vi + 1}/{total} (yaw{yaw},pitch{pitch}) · {n_hole}px")
        if n_hole < 30:
            continue

        # 1) 색 생성(인페인팅)
        completed = inp.inpaint(common.numpy_to_pil(rgb), fill, prompt=None)
        completed_np = np.asarray(completed.resize((W, H))).astype(np.uint8)

        # 2) 깊이: 채움영역 깊이를 '주변 정상 배경'에서 매끄럽게 전파(노이즈 제거)
        #    채움 픽셀의 노이즈한 자체 깊이는 버리고, 둘레의 신뢰 깊이로 채운다.
        depth_known = depth.copy()
        depth_known[fill] = 0.0
        depth_bg = _fill_depth_bg(depth_known, fill)

        # 3) 채움 픽셀 → 3D (카메라 좌표) → 원본 좌표
        ys, xs = np.where(fill)
        z = depth_bg[ys, xs]
        ok = z > 1e-3
        ys, xs, z = ys[ok], xs[ok], z[ok]
        if len(z) == 0:
            continue

        zt = torch.from_numpy(z).to(device).to(dtype)
        xt = torch.from_numpy(xs.astype(np.float32)).to(device)
        yt = torch.from_numpy(ys.astype(np.float32)).to(device)
        Xc = (xt - cx) / f * zt
        Yc = (yt - cy) / f * zt
        p_cam = torch.stack([Xc, Yc, zt], dim=1)            # (M,3) 카메라 좌표

        R = _rotation(move.yaw_deg, move.pitch_deg, device, dtype)  # p_cam = R(p-piv)+piv (+t=0)
        p_canon = (p_cam - pivot) @ R + pivot                # R^T 적용 = @ R

        col = _srgb_to_linear(completed_np[ys, xs])          # (M,3) linear
        # 렌더 시 size_boost 가 다시 곱해지므로 미리 나눠 ~1px 로 맞춤(거품 방지)
        scale_iso = (zt / f) * (float(new_scale_gain) / float(size_boost))
        sc = scale_iso[:, None].repeat(1, 3)
        qz = torch.zeros((len(z), 4), device=device, dtype=dtype)
        qz[:, 0] = 1.0

        add_means.append(p_canon)
        add_scales.append(sc)
        add_quats.append(qz)
        add_colors.append(torch.from_numpy(col).to(device).to(dtype))
        add_opac.append(torch.ones(len(z), device=device, dtype=dtype))

    try:
        inp.unload()
    except Exception:
        pass

    if not add_means:
        return s

    means = torch.cat([s.means] + add_means, dim=0)
    scales = torch.cat([s.scales] + add_scales, dim=0)
    quats = torch.cat([s.quats] + add_quats, dim=0)
    colors = torch.cat([s.colors] + add_colors, dim=0)
    opac = torch.cat([s.opacities.reshape(-1)] + add_opac, dim=0)
    added = sum(len(a) for a in add_means)
    print(f"[complete] +{added:,} gaussians (총 {means.shape[0]:,})")

    return SharpScene(means=means, scales=scales, quats=quats, colors=colors,
                      opacities=opac, f_px=s.f_px, width=s.width, height=s.height)
