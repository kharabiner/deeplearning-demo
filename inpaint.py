"""
inpaint.py — 인페인팅 백엔드 팩토리

세 가지 백엔드를 공통 인터페이스로 제공한다 (clean up / expand / reframe 공용):
  - "sd2"    : Stable Diffusion 인페인팅 (VLM 캡션 프롬프트로 고품질 생성)
  - "lama"   : LaMa (프롬프트 없이 주변 맥락으로 빠르게 채움 — LDI 배경 플레이트용)
  - "opencv" : cv2.inpaint (Telea) — 모델 없이 가장 빠른 폴백

공통 인터페이스:
    get_inpainter(backend, device).inpaint(image_pil, hole_mask, prompt=None) -> PIL.Image
    .unload()  # GPU/메모리 해제 (8GB VRAM 대응, 순차 로드/언로드)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image

from common import get_device, free_memory, mask_to_pil, composite_hole

# SD 인페인팅의 기본 프롬프트 (VLM 캡션이 없을 때 폴백)
DEFAULT_PROMPT = "seamless natural background, photorealistic, high quality, sharp focus"


class OpenCVInpainter:
    """cv2.inpaint(Telea) 기반 폴백. 모델 다운로드 없이 항상 동작."""

    def __init__(self, device: Optional[str] = None):
        self.device = device or get_device()

    def load(self):
        return self

    def unload(self):
        return None

    def inpaint(
        self,
        image: Image.Image,
        hole_mask: np.ndarray,
        prompt: Optional[str] = None,  # 시그니처 통일용 (무시)
        radius: int = 5,
    ) -> Image.Image:
        import cv2

        rgb = np.asarray(image.convert("RGB")).astype(np.uint8)
        mask = (np.asarray(hole_mask).astype(np.uint8)) * 255
        filled = cv2.inpaint(rgb, mask, radius, cv2.INPAINT_TELEA)
        return composite_hole(image, Image.fromarray(filled), hole_mask)


# ── 백엔드 팩토리 ──────────────────────────────────────────────────────────────
def get_inpainter(backend: str = "sd2", device: Optional[str] = None):
    """backend("sd2"/"lama"/"opencv") 에 맞는 인페인터 인스턴스 반환."""
    backend = (backend or "sd2").lower()
    device = device or get_device()

    if backend == "lama":
        from task_inpaint_lama import LaMaInpainter
        return LaMaInpainter(device)
    if backend in ("opencv", "cv2"):
        return OpenCVInpainter(device)
    # 기본: SD 인페인팅 (VLM 프롬프트)
    from task_inpaint_sd import SDInpainter
    return SDInpainter(device)
