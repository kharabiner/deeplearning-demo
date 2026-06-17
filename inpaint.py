"""
inpaint.py — 인페인팅 백엔드 팩토리 (remove / expand / reframe 공용)

실제 모델은 task_ 파일에 있다:
  - LaMa : task_inpaint_lama.LaMaInpainter  (프롬프트 없이 맥락 채움)
  - SD   : task_inpaint_sd.SDInpainter      (VLM 캡션 프롬프트로 생성)
여기서는 가벼운 OpenCV 폴백과 backend 선택 팩토리만 제공한다.

공통 인터페이스:
    get_inpainter(backend, device).inpaint(image_pil, hole_mask, prompt=None) -> PIL.Image
    backend: 'lama' | 'sd2' | 'opencv'
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image

from common import get_device

# SD 인페인팅의 기본 프롬프트 (VLM 캡션이 없을 때 폴백)
DEFAULT_PROMPT = "seamless natural background, photorealistic, high quality, sharp focus"


# ── OpenCV 폴백 (의존성 없이 항상 동작) ─────────────────────────────────────────
class OpenCVInpainter:
    """가장 가벼운 폴백. LaMa/SD 미설치 환경에서도 데모가 죽지 않도록."""

    def __init__(self, device: Optional[str] = None):
        self.device = device or get_device()

    def load(self):
        return self

    def unload(self):
        pass

    def inpaint(self, image: Image.Image, hole_mask: np.ndarray, prompt=None) -> Image.Image:
        import cv2

        img = np.array(image)
        mask_u8 = (hole_mask.astype(np.uint8)) * 255
        filled = cv2.inpaint(img, mask_u8, 3, cv2.INPAINT_NS)
        return Image.fromarray(filled)


# ── 팩토리 ───────────────────────────────────────────────────────────────────────
def get_inpainter(backend: str, device: Optional[str] = None):
    """backend: 'lama' | 'sd2' | 'opencv'"""
    backend = backend.lower()
    if backend == "lama":
        from task_inpaint_lama import LaMaInpainter
        return LaMaInpainter(device)
    if backend == "sd2":
        from task_inpaint_sd import SDInpainter
        return SDInpainter(device)
    if backend == "opencv":
        return OpenCVInpainter(device)
    raise ValueError(f"Unknown inpaint backend: {backend}")
