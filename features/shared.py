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

# DreamShaper inpaint — Qwen2-VL: 장면 묘사만 출력 (지시문/메타 단어 금지)
_VLM_RULES = (
    "Reply with ONE short English phrase only (max 15 words). "
    "Describe visible scene content: surfaces, colors, lighting, materials. "
    "No verbs like create/generate/expand. "
    "No words: frame, border, canvas, inpainting, outpainting, continuation, photo."
)

VLM_QUESTION_CLEANUP = (
    f"{_VLM_RULES} "
    "Task: object was removed from this photo. "
    "What background (wall, floor, sky, texture) should fill that spot?"
)

FALLBACK_PROMPT_CLEANUP = (
    "same wall and floor texture, natural indoor lighting, photorealistic, sharp focus"
)

# Expand / Reframe — 고정 DreamShaper 프롬프트 (LaMa·캔버스가 맥락 제공, VLM 불필요)
# 장면 내용만 — continuation/photograph 등 작업·메타 단어는 액자·엉뚱한 배경 유발
SD15_PROMPT_EXPAND = (
    "same background and scenery, matching colors and lighting, "
    "photorealistic, highly detailed, sharp focus"
)
SD15_PROMPT_REFRAME = (
    "same room and environment, natural lighting, photorealistic, sharp focus"
)

# VLM 이 지시문을 그대로 내보내면 DreamShaper 가 액자/메타 이미지를 그림
_BAD_PROMPT_STARTS = (
    "create ", "generate ", "describe ", "write ", "output ", "expand ",
    "seamless continuation", "continuation of", "extend the", "extending ",
)
_BAD_PROMPT_SUBSTR = (
    "beyond the current", "outpainting", "inpainting", "picture frame",
    "photo frame", " frame,", " frame ", " canvas", " border", " vignette",
)


def _sanitize_dreamshaper_prompt(raw: str, fallback: str) -> str:
    """지시문/액자 유발 메타 프롬프트 → fallback."""
    prompt = raw.strip().strip('"').strip("'").split("\n")[0].strip()
    if len(prompt) < 6:
        return fallback
    low = prompt.lower()
    if any(low.startswith(s) for s in _BAD_PROMPT_STARTS):
        return fallback
    if any(s in low for s in _BAD_PROMPT_SUBSTR):
        return fallback
    return prompt


def _vlm_dreamshaper_prompt(
    image: Image.Image,
    question: str,
    fallback: str,
    *,
    feature: str,
) -> str:
    """Qwen2-VL → DreamShaper inpaint 프롬프트. 실패 시 fallback."""
    try:
        import task_vlm_qwen2vl as task_vlm
        vproc, vmodel = task_vlm.load_model(DEVICE)
        desc = task_vlm.run(
            image, vproc, vmodel,
            question=question,
            max_new_tokens=64,
            device=DEVICE,
        )
        del vproc, vmodel
        free_memory(DEVICE)
        prompt = _sanitize_dreamshaper_prompt(desc, fallback)
        if prompt == fallback:
            print(f"[shared] VLM {feature} meta/unsafe → fallback (raw: {desc[:60]!r})")
        else:
            print(f"[shared] VLM {feature} prompt: {prompt[:80]}")
        return prompt
    except Exception as e:
        print(f"[shared] VLM {feature} 실패 → fallback: {e}")
        return fallback


def vlm_caption_clean_up(image: Image.Image) -> str:
    """Clean Up [완료] — DreamShaper용 프롬프트 (Qwen2-VL 생성)."""
    return _vlm_dreamshaper_prompt(
        image, VLM_QUESTION_CLEANUP, FALLBACK_PROMPT_CLEANUP, feature="Clean Up",
    )


# 하위 호환
vlm_caption = vlm_caption_clean_up


# ── 인페인팅 실행 (확정 공용) ───────────────────────────────────────────────────
def inpaint_commit(
    image_pil, fill, progress, desc="인페인팅",
    prompt=None, caption_fn=None, backend="sd15", sd_guidance=7.5,
    sd_long=None, sd_steps=None, sd_negative=None,
):
    """인페인팅 실행. (결과 PIL, 상태 메시지) 반환.

    backend="lama"  : 프롬프트 없이 주변 맥락으로 채움
    backend="sd15"  : DreamShaper SD1.5 inpaint
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

    label = "DreamShaper"
    if prompt is None:
        prompt = (caption_fn or vlm_caption_clean_up)(image_pil)
    progress(0.6, desc=f"{desc} — {label}")
    try:
        inp = rinp.get_inpainter("sd15", DEVICE)
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
