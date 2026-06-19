"""
inpaint.py — 인페인팅 백엔드 팩토리

  - "sd2"/"sdxl" : SDXL 인페인팅 (1024px — Clean Up)
  - "expand"     : DreamShaper SD1.5 inpaint 세션 캐시 (Expand [완료])
  - "sd15"       : SD 1.5 인페인팅 (512px — Reframe)
  - "lama"       : LaMa (프롬프트 없이 주변 맥락으로 빠르게 채움)
  - "opencv"     : cv2.inpaint (Telea) — 모델 없이 가장 빠른 폴백

공통 인터페이스:
    get_inpainter(backend, device).inpaint(image_pil, hole_mask, prompt=None) -> PIL.Image
    .unload()
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image

from common import get_device, free_memory, mask_to_pil, composite_hole

DEFAULT_PROMPT = "seamless natural background, photorealistic, high quality, sharp focus"

# Expand [완료] — DreamShaper SD1.5 세션 캐시 (다른 탭 전환 시 unload_expand_sd15)
_expand_inp = None
_expand_device: Optional[str] = None


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
        prompt: Optional[str] = None,
        radius: int = 5,
    ) -> Image.Image:
        import cv2

        rgb = np.asarray(image.convert("RGB")).astype(np.uint8)
        mask = (np.asarray(hole_mask).astype(np.uint8)) * 255
        filled = cv2.inpaint(rgb, mask, radius, cv2.INPAINT_TELEA)
        return composite_hole(image, Image.fromarray(filled), hole_mask)


def get_inpainter(backend: str = "sd2", device: Optional[str] = None):
    """backend 에 맞는 인페인터 인스턴스 반환."""
    backend = (backend or "sd2").lower()
    device = device or get_device()

    if backend == "lama":
        from task_inpaint_lama import LaMaInpainter
        return LaMaInpainter(device)
    if backend in ("opencv", "cv2"):
        return OpenCVInpainter(device)
    if backend in ("sd15", "sd1.5", "sd-1.5"):
        from task_inpaint_sd15 import SD15Inpainter
        return SD15Inpainter(device)
    # sd2 / sdxl → SDXL (Clean Up)
    from task_inpaint_sd import SDXLInpainter
    return SDXLInpainter(device)


def get_expand_inpainter(device: Optional[str] = None):
    """Expand [완료] DreamShaper SD1.5 싱글톤."""
    global _expand_inp, _expand_device
    device = device or get_device()
    if _expand_inp is None or _expand_device != device:
        if _expand_inp is not None:
            _expand_inp.unload()
        from task_inpaint_sd15 import ExpandSD15Inpainter
        _expand_inp = ExpandSD15Inpainter(device)
        _expand_device = device
    return _expand_inp


def preload_expand_sd15(device: Optional[str] = None):
    """Expand DreamShaper SD1.5 를 GPU 에 올림."""
    inp = get_expand_inpainter(device)
    if not inp.is_ready():
        inp.load()
    return inp


def unload_expand_sd15():
    """Expand 세션 종료 — Expand 재시작 / Clean Up / Reframe 진입 시."""
    global _expand_inp, _expand_device
    dev = _expand_device
    if _expand_inp is not None:
        _expand_inp.unload()
        _expand_inp = None
        _expand_device = None
    if dev:
        free_memory(dev)


# 하위 호환 (기존 호출부 정리 전 alias)
unload_expand_sdxl = unload_expand_sd15
