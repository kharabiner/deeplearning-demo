"""
task_inpaint_sd.py — Generative Image Inpainting
모델: SDXL Inpainting (diffusers/stable-diffusion-xl-1.0-inpainting-0.1)

1024px 네이티브 · 큰 마스크(아웃페인팅)에 SD1.5보다 적합.
텍스트 프롬프트로 채울 내용을 유도하는 생성형 인페인팅 (clean up / expand / reframe 공용).

8GB VRAM: fp16 + attention/vae slicing (순차 load/unload 필수).

사용법(단독):
    python task_inpaint_sd.py --image sample.jpg --prompt "wooden floor"

공통 인터페이스:
    SDXLInpainter(device).inpaint(image_pil, hole_mask, prompt=None) -> PIL.Image
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from PIL import Image

from common import get_device, get_dtype, free_memory, mask_to_pil, composite_hole

MODEL_ID = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
INPAINT_LONG = 1024   # SDXL 네이티브 해상도(긴 변) — 512 대비 아웃페인팅 품질↑

DEFAULT_PROMPT = "seamless natural background, photorealistic, high quality, sharp focus"
NEGATIVE_PROMPT = (
    "person, people, human, man, woman, child, face, body, crowd, "
    "new object, duplicate, extra limbs, "
    "blurry, distorted, artifacts, watermark, text, low quality, deformed"
)


def _align8(n: int) -> int:
    return max(8, (int(round(n)) // 8) * 8)


def _resize_for_inpaint(image: Image.Image, mask: Image.Image, long: int = INPAINT_LONG):
    """SDXL 입력: 긴 변 long, 8의 배수. (리사이즈 PIL, 원본 W,H) 반환."""
    W, H = image.size
    scale = long / max(W, H)
    rw, rh = _align8(W * scale), _align8(H * scale)
    return (
        image.resize((rw, rh), Image.LANCZOS),
        mask.resize((rw, rh), Image.NEAREST),
        W, H,
    )


class SDXLInpainter:
    """SDXL Inpainting. prompt 로 채울 내용을 유도."""

    def __init__(self, device: Optional[str] = None):
        self.device = device or get_device()
        self.pipe = None

    def load(self):
        from diffusers import StableDiffusionXLInpaintPipeline

        print(f"[inpaint:sdxl] Loading {MODEL_ID}")
        dtype = get_dtype(self.device)
        kwargs = {"torch_dtype": dtype, "use_safetensors": True}
        if self.device == "cuda":
            kwargs["variant"] = "fp16"
        self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(MODEL_ID, **kwargs)
        if self.device == "cuda":
            self.pipe.enable_attention_slicing()
            self.pipe.vae.enable_slicing()
        self.pipe = self.pipe.to(self.device)
        print(f"[inpaint:sdxl] ready on {self.device} ({dtype}) · long={INPAINT_LONG}")
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
        long: Optional[int] = None,
    ) -> Image.Image:
        if self.pipe is None:
            self.load()

        if prompt is None:
            prompt = DEFAULT_PROMPT
        inpaint_long = long if long is not None else INPAINT_LONG
        mask = mask_to_pil(hole_mask)
        img_r, mask_r, W, H = _resize_for_inpaint(image, mask, long=inpaint_long)
        print(f"[inpaint:sdxl] infer · long={inpaint_long} · steps={steps} · {img_r.size[0]}×{img_r.size[1]}")

        gen_dev = self.device if self.device in ("cuda", "mps") else "cpu"
        generator = (
            torch.Generator(device=gen_dev).manual_seed(seed) if seed is not None else None
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

        return composite_hole(image, result, hole_mask)


def load_model(device: str) -> SDXLInpainter:
    return SDXLInpainter(device).load()


def run(image: Image.Image, model: SDXLInpainter, hole_mask: np.ndarray,
        prompt: Optional[str] = None) -> Image.Image:
    return model.inpaint(image, hole_mask, prompt=prompt)


# 하위 호환
SDInpainter = SDXLInpainter


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from common import load_image, save_result

    parser = argparse.ArgumentParser(description="SDXL inpainting 데모")
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
    save_result(out, f"{stem}_sdxl.png")
    print("완료 → outputs/ 확인")
