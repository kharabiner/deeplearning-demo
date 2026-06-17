"""
reframe_sharp.py — SHARP 생성형 Reframe (격자 사전 렌더 + 최근접 조회)

iOS Reframe UX 재현:
  - [분석] 단계에서 SHARP가 3DGS를 1회 생성한 뒤, yaw×pitch 각도 격자를
    splat_render 로 미리 다 렌더해 캐시한다(= iOS의 "잠깐 대기").
  - 드래그 중에는 슬라이더 각도에 가장 가까운 캐시 프레임을 즉시 보여준다(0ms).
    → 실시간 GS 렌더러(gsplat, CUDA 전용 빌드) 없이도 매끄러운 드래그.
  - 새로 드러난 바깥/구멍은 coverage 마스크로 표시 → 드래그 중 블러, 확정 시 인페인팅.

각도 범위 밖 자유 이동(truck/pedestal/dolly)이나 정밀 확정은 render_exact 로 즉석 렌더.
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
    size_boost: float = 2.2,
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
    size_boost: float = 2.2,
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
        supersample=supersample, device=device,
    )
