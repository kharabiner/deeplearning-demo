"""
features/shared.py — Clean Up / Expand / Reframe 공용 글루 코드
"""

from __future__ import annotations

import gradio as gr
import numpy as np
from PIL import Image

from common import (
    get_device, free_memory, pil_to_numpy, numpy_to_pil,
)
import inpaint as rinp

DEVICE = get_device()

PREVIEW_MAX = 1024
COMMIT_LONG = 1280

HIDDEN = gr.update(visible=False)
VISIBLE = gr.update(visible=True)


# ── VLM 캡션 (SD 인페인팅 프롬프트) ─────────────────────────────────────────────
def vlm_caption(image: Image.Image) -> str:
    """SD2 인페인팅 프롬프트용 캡션. 실패 시 기본 프롬프트."""
    try:
        import task_vlm_qwen2vl as task_vlm
        vproc, vmodel = task_vlm.load_model(DEVICE)
        desc = task_vlm.run(
            image, vproc, vmodel,
            question="Describe the background scene in a short phrase for image inpainting "
                     "(scene type, colors, lighting). Keep it under 15 words.",
            max_new_tokens=48, device=DEVICE,
        )
        del vproc, vmodel
        free_memory(DEVICE)
        return f"{desc.strip()}, {rinp.DEFAULT_PROMPT}"
    except Exception as e:
        print(f"[shared] VLM 캡션 실패 → 기본 프롬프트: {e}")
        return rinp.DEFAULT_PROMPT


def vlm_caption_reframe(image: Image.Image) -> str:
    """Reframe 바깥 SD1.5용 — 장면 연장(사람·사물 포함 가능)."""
    try:
        import task_vlm_qwen2vl as task_vlm
        vproc, vmodel = task_vlm.load_model(DEVICE)
        desc = task_vlm.run(
            image, vproc, vmodel,
            question="Describe this photo scene in a short phrase for extending "
                     "the image edges seamlessly. Under 15 words.",
            max_new_tokens=48, device=DEVICE,
        )
        del vproc, vmodel
        free_memory(DEVICE)
        return (
            f"seamless extension of {desc.strip()}, same scene and lighting, "
            "photorealistic, sharp focus"
        )
    except Exception as e:
        print(f"[shared] Reframe VLM 실패 → 기본: {e}")
        return (
            "seamless natural scene extension, same lighting and perspective, "
            "photorealistic, sharp focus"
        )


# ── 인페인팅 실행 (확정 공용) ───────────────────────────────────────────────────
def inpaint_commit(
    image_pil, fill, progress, desc="인페인팅",
    prompt=None, caption_fn=None, backend="sd2", sd_guidance=7.5,
    sd_long=None, sd_steps=None, sd_negative=None,
):
    """인페인팅 실행. (결과 PIL, 상태 메시지) 반환.

    backend="lama" : 프롬프트 없이 주변 맥락으로 채움
    backend="sd2"  : SDXL (Clean Up 등)
    backend="sd15" : SD 1.5 (Expand [완료] — SDXL 대비 빠름)
    """
    if int(fill.sum()) < 30:
        return image_pil, "완료 · 채울 영역 없음"

    if backend == "lama":
        progress(0.6, desc=f"{desc} — LaMa")
        try:
            inp = rinp.get_inpainter("lama", DEVICE)
            result = inp.inpaint(image_pil, fill)
            inp.unload()
        except Exception as e:
            raise gr.Error(f"{desc} 실패(LaMa): {e}")
        return result, f"완료 · LaMa(프롬프트 없음, 주변 맥락) · 우상단 아이콘으로 다운로드"

    use_sd15 = backend in ("sd15", "sd1.5", "sd-1.5")
    label = "SD1.5" if use_sd15 else "SDXL"
    if prompt is None:
        prompt = (caption_fn or vlm_caption)(image_pil)
    progress(0.6, desc=f"{desc} — {label}")
    try:
        inp = rinp.get_inpainter("sd15" if use_sd15 else "sd2", DEVICE)
        kw = {"prompt": prompt, "guidance": sd_guidance}
        if sd_long is not None:
            kw["long"] = sd_long
        if sd_steps is not None:
            kw["steps"] = sd_steps
        if sd_negative is not None:
            kw["negative_prompt"] = sd_negative
        result = inp.inpaint(image_pil, fill, **kw)
        inp.unload()
        free_memory(DEVICE)
    except Exception as e:
        raise gr.Error(f"{desc} 실패({label}): {e}")

    prompt_label = "(없음, 이미지 맥락)" if prompt == "" else prompt[:50]
    return result, f"완료 · {label} · prompt={prompt_label} · 우상단 아이콘으로 다운로드"


# ── 합성/마스크 헬퍼 ────────────────────────────────────────────────────────────
def feather_composite(orig_pil, filled_pil, fill_mask, feather=2.5):
    """채운 영역을 원본에 부드러운 경계로 합성 → 하드 씸(딱딱한 테두리) 제거."""
    orig = pil_to_numpy(orig_pil).astype(np.float32)
    fill = np.asarray(filled_pil).astype(np.float32)
    if fill.shape != orig.shape:
        fill = pil_to_numpy(filled_pil.resize(orig_pil.size, Image.LANCZOS)).astype(np.float32)
    try:
        import cv2
        alpha = cv2.GaussianBlur(fill_mask.astype(np.float32), (0, 0), feather)
    except Exception:
        alpha = fill_mask.astype(np.float32)
    alpha = np.clip(alpha, 0.0, 1.0)[..., None]
    out = orig * (1.0 - alpha) + fill * alpha
    return numpy_to_pil(out.clip(0, 255).astype(np.uint8))


def resize_mask(mask: np.ndarray, H: int, W: int) -> np.ndarray:
    """bool 마스크를 (H, W) 로 NEAREST 리사이즈."""
    if mask.shape[:2] == (H, W):
        return mask
    return np.array(
        Image.fromarray(mask.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)
    ).astype(bool)


def dilate_mask(mask: np.ndarray, iterations: int = 2) -> np.ndarray:
    """인페인팅 경계를 깔끔하게 하려 마스크를 살짝 팽창 (cv2 없으면 그대로)."""
    try:
        import cv2
        kernel = np.ones((3, 3), np.uint8)
        return cv2.dilate(mask.astype(np.uint8), kernel, iterations=iterations).astype(bool)
    except Exception:
        return mask


def blur_holes(rgb: np.ndarray, hole_mask: np.ndarray, blur_sigma: float = 12.0) -> np.ndarray:
    """드래그 중 바깥(빈) 영역만 뿌옇게 — 중앙(피사체) 선명도는 유지."""
    if not hole_mask.any():
        return rgb
    out = rgb.copy()
    try:
        import cv2
        soft = cv2.GaussianBlur(rgb, (0, 0), blur_sigma)
        out[hole_mask] = soft[hole_mask]
        return out
    except Exception:
        k, pad = 11, 5
        padded = np.pad(out, ((pad, pad), (pad, pad), (0, 0)), mode="edge").astype(np.float32)
        acc = np.zeros_like(out, dtype=np.float32)
        for dy in range(k):
            for dx in range(k):
                acc += padded[dy:dy + out.shape[0], dx:dx + out.shape[1]]
        out[hole_mask] = (acc / (k * k)).astype(np.uint8)[hole_mask]
        return out
