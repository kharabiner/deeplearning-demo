"""
inpaint.py — 인페인팅 백엔드 팩토리

  - "sd15" / "dreamshaper" : DreamShaper SD1.5 inpaint (Clean Up · Reframe)
  - "expand"               : Expand [완료] 세션 캐시 (동일 모델, GPU 상주)
  - "lama"                 : LaMa (Expand 미리보기)
  - "opencv"               : cv2.inpaint (Telea) — Reframe 폴백

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

# Expand [완료] — DreamShaper 세션 캐시
_expand_inp = None
_expand_device: Optional[str] = None

_SD15_BACKENDS = frozenset({"sd15", "sd1.5", "sd-1.5", "dreamshaper"})


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


def get_inpainter(backend: str = "sd15", device: Optional[str] = None):
    """backend 에 맞는 인페인터 인스턴스 반환."""
    backend = (backend or "sd15").lower()
    device = device or get_device()

    if backend == "lama":
        from task_inpaint_lama import LaMaInpainter
        return LaMaInpainter(device)
    if backend in ("opencv", "cv2"):
        return OpenCVInpainter(device)
    if backend in _SD15_BACKENDS:
        from task_inpaint_sd15 import SD15Inpainter
        return SD15Inpainter(device)
    raise ValueError(f"Unknown inpaint backend: {backend!r}")


def get_expand_inpainter(device: Optional[str] = None):
    """Expand [완료] DreamShaper 싱글톤."""
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
    """Expand DreamShaper 를 GPU 에 올림."""
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
