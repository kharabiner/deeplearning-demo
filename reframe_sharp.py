"""
reframe_sharp.py — SHARP 생성형 Reframe (확정 렌더)

앱(app.py) 파이프라인:
  - [분석] SHARP predict → 3DGS 1회 생성
  - [드래그] depth warp 미리보기 (선명도 유지, iOS 방식 블러)
  - [확정] render_exact → photo anchor → 인페인팅

build_grid / ReframeGrid 는 오프라인·향후용(앱 미사용). 드래그 중 즉시 GS 렌더 대신
depth warp 프리뷰를 쓴다(SHARP splat 전체가 뿌옇게 보이는 문제 회피).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

import common
import splat_render
from reframe_core import CameraMove
from task_nvs_sharp import SharpScene


@dataclass
class ReframeGrid:
    """yaw×pitch 사전 렌더 캐시."""

    yaws: np.ndarray              # (NY,)  도
    pitches: np.ndarray          # (NP,)  도
    images: List[List[np.ndarray]]   # [NP][NY] uint8 (H,W,3)
    covers: List[List[np.ndarray]]   # [NP][NY] bool  (H,W)
    pivot_z: float

    def nearest(self, yaw: float, pitch: float) -> Tuple[np.ndarray, np.ndarray]:
        yi = int(np.argmin(np.abs(self.yaws - float(yaw))))
        pi = int(np.argmin(np.abs(self.pitches - float(pitch))))
        return self.images[pi][yi], self.covers[pi][yi]


def build_grid(
    scene: SharpScene,
    *,
    yaw_max: float = 16.0,
    pitch_max: float = 10.0,
    n_yaw: int = 11,
    n_pitch: int = 5,
    out_long: int = 640,
    size_boost: float = 1.6,
    close_holes: bool = False,
    supersample: int = 1,
    device: Optional[str] = None,
    progress=None,
) -> ReframeGrid:
    """SHARP 장면을 yaw×pitch 격자로 미리 렌더."""
    device = device or common.get_device()

    yaws = np.linspace(-yaw_max, yaw_max, n_yaw)
    pitches = np.linspace(-pitch_max, pitch_max, n_pitch)

    # 출력 해상도(원본 종횡비 유지)
    if scene.width >= scene.height:
        W = out_long
        H = max(1, round(out_long * scene.height / scene.width))
    else:
        H = out_long
        W = max(1, round(out_long * scene.width / scene.height))

    import torch
    pivot_z = float(np.median(scene.means[:, 2].cpu().numpy()))

    images: List[List[np.ndarray]] = []
    covers: List[List[np.ndarray]] = []
    total = n_pitch * n_yaw
    k = 0
    for pi, p in enumerate(pitches):
        row_img, row_cov = [], []
        for yi, y in enumerate(yaws):
            rgb, cov = splat_render.render(
                scene, CameraMove(yaw_deg=float(y), pitch_deg=float(p)),
                out_hw=(H, W), pivot_z=pivot_z, size_boost=size_boost,
                trim_coverage=False, close_holes=close_holes,
                supersample=supersample, device=device,
            )
            row_img.append(rgb)
            row_cov.append(cov)
            k += 1
            if progress is not None:
                progress(0.55 + 0.4 * k / total, desc=f"Reframe 격자 렌더 {k}/{total}")
        images.append(row_img)
        covers.append(row_cov)

    return ReframeGrid(yaws=yaws, pitches=pitches, images=images, covers=covers, pivot_z=pivot_z)


def render_exact(
    scene: SharpScene,
    move: CameraMove,
    *,
    pivot_z: float,
    out_long: int = 768,
    size_boost: float = 1.6,
    supersample: int = 1,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """확정/자유이동용 즉석 렌더 (격자 밖 각도·평행이동 지원)."""
    device = device or common.get_device()
    if scene.width >= scene.height:
        W = out_long
        H = max(1, round(out_long * scene.height / scene.width))
    else:
        H = out_long
        W = max(1, round(out_long * scene.width / scene.height))
    return splat_render.render(
        scene, move, out_hw=(H, W), pivot_z=pivot_z, size_boost=size_boost,
        trim_coverage=True, close_holes=True,
        supersample=supersample, device=device,
    )
