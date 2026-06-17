"""
task_nvs_sharp.py — Apple SHARP 단일 이미지 → 3D Gaussian Splatting (생성형 Reframe용)

SHARP(apple/ml-sharp)는 사진 1장에서 3D Gaussian 장면을 1회 피드포워드로 예측한다.
가려졌던(안 보이던) 면도 Gaussian으로 "생성"되므로, 시점을 돌리면 옆면이 드러난다.
→ iOS Reframe의 핵심.

이 파일은 **예측만** 담당한다(렌더링은 splat_render.py가 순수 PyTorch로 처리).
공식 repo의 gsplat 렌더러(CUDA 전용)는 import하지 않는다.

좌표계: OpenCV (x→오른쪽, y→아래, z→앞). 카메라는 원점에서 +z를 바라봄.
색공간: linearRGB로 보관(알파 블렌딩은 선형 공간에서) → 최종 표시 시 sRGB 변환.

설치(예측 전용, gsplat 없이):
    pip install -e third_party/ml-sharp --no-deps
    pip install timm "numpy==1.26.4" "scipy==1.13.1" plyfile

가중치는 최초 실행 시 자동 다운로드(공개 URL, 토큰 불필요):
    ~/.cache/torch/hub/checkpoints/sharp_2572gikvuh.pt
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import common

DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"

# SHARP 내부 처리 해상도(고정). 출력 Gaussian 수 ≈ (1536/2)^2.
_INTERNAL_SHAPE = (1536, 1536)

_PREDICTOR = None  # 싱글톤 (모델 1회 로드)


@dataclass
class SharpScene:
    """SHARP가 예측한 3D Gaussian 장면 (월드/메트릭, OpenCV 좌표)."""

    means: torch.Tensor       # (N, 3)  중심
    scales: torch.Tensor      # (N, 3)  각 축 표준편차(=특이값)
    quats: torch.Tensor       # (N, 4)  회전 쿼터니언 (w, x, y, z 순서는 SHARP 규약)
    colors: torch.Tensor      # (N, 3)  linearRGB [0..1 부근]
    opacities: torch.Tensor   # (N,)    [0..1]
    f_px: float               # 원본 이미지 기준 초점거리(px)
    width: int                # 원본 이미지 너비
    height: int               # 원본 이미지 높이

    @property
    def num_gaussians(self) -> int:
        return self.means.shape[0]

    def to(self, device: str) -> "SharpScene":
        return SharpScene(
            means=self.means.to(device),
            scales=self.scales.to(device),
            quats=self.quats.to(device),
            colors=self.colors.to(device),
            opacities=self.opacities.to(device),
            f_px=self.f_px,
            width=self.width,
            height=self.height,
        )


# ── 초점거리 추정 (io.py 의존 회피: matplotlib까지 끌어오므로 직접 복제) ──────────
def _focal_px(img_pil: Image.Image) -> float:
    """EXIF 35mm 환산 초점거리 → 픽셀 초점거리. 없으면 30mm 가정."""
    f_35mm = None
    try:
        exif = img_pil.getexif().get_ifd(0x8769)
        from PIL import ExifTags

        tags = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
        f_35mm = tags.get("FocalLengthIn35mmFilm") or tags.get("FocalLenIn35mmFilm")
        if (f_35mm is None or f_35mm < 1):
            f_35mm = tags.get("FocalLength")
            if f_35mm is not None and f_35mm < 10.0:
                f_35mm = float(f_35mm) * 8.4  # crude: 비-35mm 환산
    except Exception:
        f_35mm = None

    if f_35mm is None:
        f_35mm = 30.0
    f_35mm = float(f_35mm)

    w, h = img_pil.size
    # io.convert_focallength 와 동일 공식
    return f_35mm * np.sqrt(w ** 2.0 + h ** 2.0) / np.sqrt(36 ** 2 + 24 ** 2)


# ── 모델 로드 (싱글톤) ──────────────────────────────────────────────────────────
def load_predictor(device: Optional[str] = None):
    """SHARP predictor 로드(최초 1회). 이후 캐시 재사용."""
    global _PREDICTOR
    device = device or common.get_device()
    if _PREDICTOR is not None:
        return _PREDICTOR

    # gsplat을 import하는 cli.predict / utils.gsplat 은 건드리지 않는다.
    from sharp.models import PredictorParams, create_predictor

    print(f"[sharp] downloading/loading checkpoint ... ({device})")
    state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)

    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    predictor.eval()
    predictor.to(device)
    _PREDICTOR = predictor
    print("[sharp] predictor ready.")
    return predictor


# ── 예측 ────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(
    image: Union[str, Path, Image.Image, np.ndarray],
    device: Optional[str] = None,
) -> SharpScene:
    """이미지 1장 → SharpScene(3D Gaussians).

    cli.predict.predict_image 로직을 gsplat 의존 없이 복제.
    """
    device = device or common.get_device()
    predictor = load_predictor(device)

    # 입력 정규화 → PIL(RGB)
    if isinstance(image, (str, Path)):
        img_pil = common.load_image(image)            # 자동 resize 포함
    elif isinstance(image, np.ndarray):
        img_pil = common.numpy_to_pil(image)
    else:
        img_pil = common.resize_if_needed(image.convert("RGB"))

    f_px = _focal_px(img_pil)
    img_np = common.pil_to_numpy(img_pil)             # (H, W, 3) uint8
    height, width = img_np.shape[:2]

    image_pt = torch.from_numpy(img_np.copy()).float().to(device).permute(2, 0, 1) / 255.0
    disparity_factor = torch.tensor([f_px / width], dtype=torch.float32, device=device)

    image_resized = F.interpolate(
        image_pt[None], size=(_INTERNAL_SHAPE[1], _INTERNAL_SHAPE[0]),
        mode="bilinear", align_corners=True,
    )

    gaussians_ndc = predictor(image_resized, disparity_factor)

    # NDC → 메트릭 월드 좌표
    from sharp.utils.gaussians import unproject_gaussians

    intrinsics = torch.tensor(
        [[f_px, 0, width / 2, 0],
         [0, f_px, height / 2, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]],
        dtype=torch.float32, device=device,
    )
    intrinsics_resized = intrinsics.clone()
    intrinsics_resized[0] *= _INTERNAL_SHAPE[0] / width
    intrinsics_resized[1] *= _INTERNAL_SHAPE[1] / height

    g = unproject_gaussians(
        gaussians_ndc, torch.eye(4, device=device), intrinsics_resized, _INTERNAL_SHAPE
    )

    # NamedTuple(Gaussians3D) → 평탄화하여 SharpScene 으로
    scene = SharpScene(
        means=g.mean_vectors.flatten(0, 1).contiguous(),
        scales=g.singular_values.flatten(0, 1).contiguous(),
        quats=g.quaternions.flatten(0, 1).contiguous(),
        colors=g.colors.flatten(0, 1).contiguous(),
        opacities=g.opacities.flatten(0, 1).contiguous(),
        f_px=float(f_px),
        width=int(width),
        height=int(height),
    )
    return scene


# ── CLI 스모크 테스트 ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "sample.jpg"
    dev = common.get_device()
    print(f"[env] {common.device_info()}")

    import time

    t0 = time.time()
    scene = predict(path, device=dev)
    dt = time.time() - t0

    print(f"[sharp] {path}: {scene.num_gaussians:,} gaussians in {dt:.2f}s")
    print(f"  means  {tuple(scene.means.shape)}  range "
          f"[{scene.means.min().item():.2f}, {scene.means.max().item():.2f}]")
    print(f"  z(depth) range [{scene.means[:, 2].min().item():.2f}, "
          f"{scene.means[:, 2].max().item():.2f}]  (meters)")
    print(f"  scales range [{scene.scales.min().item():.4f}, {scene.scales.max().item():.4f}]")
    print(f"  colors range [{scene.colors.min().item():.3f}, {scene.colors.max().item():.3f}] (linearRGB)")
    print(f"  opacities range [{scene.opacities.min().item():.3f}, {scene.opacities.max().item():.3f}]")
    print(f"  f_px={scene.f_px:.1f}  size={scene.width}x{scene.height}")
