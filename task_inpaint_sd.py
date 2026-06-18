"""
task_inpaint_sd.py — Generative Image Inpainting
모델: Stable Diffusion Inpainting (stable-diffusion-v1-5/stable-diffusion-inpainting)

텍스트 프롬프트로 채울 내용을 유도하는 생성형 인페인팅.
배경이 복잡하거나 새 콘텐츠가 필요할 때 유리하며, Qwen2-VL(task_vlm_qwen2vl) 캡션을
프롬프트로 받으면 VLM 기여가 결과로 드러난다 (clean up / expand / reframe 공용).

* stabilityai/stable-diffusion-2-inpainting 은 게이트로 전환 → 비게이트 SD1.5
  인페인팅 사용 (토큰 불필요, 512 네이티브).

사용법(단독):
    python task_inpaint_sd.py --image sample.jpg --prompt "wooden floor"

공통 인터페이스:
    SDInpainter(device).inpaint(image_pil, hole_mask, prompt=None) -> PIL.Image
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from PIL import Image

from common import get_device, get_dtype, free_memory, mask_to_pil, composite_hole

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-inpainting"

DEFAULT_PROMPT = "seamless natural background, photorealistic, high quality, sharp focus"
NEGATIVE_PROMPT = "blurry, distorted, artifacts, watermark, text, low quality, deformed"


class SDInpainter:
    """Stable Diffusion Inpainting. prompt 로 채울 내용을 유도."""

    def __init__(self, device: Optional[str] = None):
        self.device = device or get_device()
        self.pipe = None

    def load(self):
        from diffusers import AutoPipelineForInpainting

        print(f"[inpaint:sd] Loading {MODEL_ID}")
        dtype = get_dtype(self.device)
        self.pipe = AutoPipelineForInpainting.from_pretrained(MODEL_ID, torch_dtype=dtype)
        self.pipe = self.pipe.to(self.device)
        if self.device == "cuda":           # 8GB VRAM 안전장치
            self.pipe.enable_attention_slicing()
        print(f"[inpaint:sd] ready on {self.device} ({dtype})")
        return self

    def unload(self):
        self.pipe = None
        free_memory(self.device)

    @torch.no_grad()
    def inpaint(
        self,
        image: Image.Image,
        hole_mask: np.ndarray,
        prompt: Optional[str] = None,
        steps: int = 25,
        guidance: float = 7.5,
        seed: Optional[int] = 0,
    ) -> Image.Image:
        if self.pipe is None:
            self.load()

        prompt = prompt or DEFAULT_PROMPT
        mask = mask_to_pil(hole_mask)

        # SD 는 입력을 8의 배수로 요구 → 512 기준 리사이즈 후 원복
        W, H = image.size
        scale = 512 / max(W, H)
        rw, rh = max((int(round(W * scale)) // 8) * 8, 8), max((int(round(H * scale)) // 8) * 8, 8)

        img_r = image.resize((rw, rh), Image.LANCZOS)
        mask_r = mask.resize((rw, rh), Image.NEAREST)
        generator = (
            torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None
        )

        result = self.pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            image=img_r,
            mask_image=mask_r,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        ).images[0].resize((W, H), Image.LANCZOS)

        # 채울 곳만 합성 (원본 픽셀은 그대로)
        return composite_hole(image, result, hole_mask)


# ── 모델 로드 헬퍼 (task_* 컨벤션) ──────────────────────────────────────────────
def load_model(device: str) -> SDInpainter:
    return SDInpainter(device).load()


def run(image: Image.Image, model: SDInpainter, hole_mask: np.ndarray,
        prompt: Optional[str] = None) -> Image.Image:
    return model.inpaint(image, hole_mask, prompt=prompt)


# ── CLI: 가운데 사각형을 지워 생성 채움 데모 ────────────────────────────────────
if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from common import load_image, save_result

    parser = argparse.ArgumentParser(description="SD inpainting 데모")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--box", type=float, default=0.25)
    args = parser.parse_args()

    device = get_device()
    image = load_image(args.image)
    W, H = image.size

    mask = np.zeros((H, W), dtype=bool)
    bw, bh = int(W * args.box), int(H * args.box)
    x0, y0 = (W - bw) // 2, (H - bh) // 2
    mask[y0:y0 + bh, x0:x0 + bw] = True

    inp = load_model(device)
    out = run(image, inp, mask, prompt=args.prompt)
    inp.unload()

    stem = Path(args.image).stem
    save_result(out, f"{stem}_sd.png")
    print("완료 → outputs/ 확인")
