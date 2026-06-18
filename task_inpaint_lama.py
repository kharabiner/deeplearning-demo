"""
task_inpaint_lama.py — Image Inpainting
모델: LaMa (big-lama, via simple-lama-inpainting)

프롬프트 없이 주변 맥락만으로 빈 곳을 채운다. 빠르고 헛것을 그리지 않아
배경이 단순할 때 깔끔하다 (clean up / expand / reframe 모두에서 사용).

설치: pip install simple-lama-inpainting  (토큰 불필요, 가중치 자동 다운로드)

사용법(단독):
    python task_inpaint_lama.py --image sample.jpg   # 가운데 박스를 지워 채움 데모

공통 인터페이스:
    LaMaInpainter(device).inpaint(image_pil, hole_mask, prompt=None) -> PIL.Image
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from PIL import Image

from common import get_device, free_memory, mask_to_pil, composite_hole


class LaMaInpainter:
    """프롬프트 없는 고품질 배경 채움. 무거우므로 순차 로드/언로드(8GB VRAM 대응)."""

    def __init__(self, device: Optional[str] = None):
        self.device = device or get_device()
        self.model = None

    def load(self):
        from simple_lama_inpainting import SimpleLama

        print("[inpaint:lama] Loading big-lama")
        # SimpleLama 는 cuda 가능 시 사용, 그 외 cpu
        self.model = SimpleLama(
            device=torch.device(self.device if self.device == "cuda" else "cpu")
        )
        print("[inpaint:lama] ready")
        return self

    def unload(self):
        self.model = None
        free_memory(self.device)

    def inpaint(
        self,
        image: Image.Image,
        hole_mask: np.ndarray,
        prompt: Optional[str] = None,  # LaMa 는 prompt 무시 (시그니처 통일용)
    ) -> Image.Image:
        if self.model is None:
            self.load()

        mask = mask_to_pil(hole_mask)
        result = self.model(image, mask)
        if result.size != image.size:
            result = result.resize(image.size, Image.LANCZOS)
        return composite_hole(image, result, hole_mask)


# ── 모델 로드 헬퍼 (task_* 컨벤션) ──────────────────────────────────────────────
def load_model(device: str) -> LaMaInpainter:
    return LaMaInpainter(device).load()


def run(image: Image.Image, model: LaMaInpainter, hole_mask: np.ndarray) -> Image.Image:
    return model.inpaint(image, hole_mask)


# ── CLI: 가운데 사각형을 지워 채우는 데모 ───────────────────────────────────────
if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from common import load_image, save_result

    parser = argparse.ArgumentParser(description="LaMa inpainting 데모")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--box", type=float, default=0.25,
                        help="가운데 정사각 마스크 비율 (0~1)")
    args = parser.parse_args()

    device = get_device()
    image = load_image(args.image)
    W, H = image.size

    mask = np.zeros((H, W), dtype=bool)
    bw, bh = int(W * args.box), int(H * args.box)
    x0, y0 = (W - bw) // 2, (H - bh) // 2
    mask[y0:y0 + bh, x0:x0 + bw] = True

    inp = load_model(device)
    out = run(image, inp, mask)
    inp.unload()

    stem = Path(args.image).stem
    save_result(out, f"{stem}_lama.png")
    print("완료 → outputs/ 확인")
