"""
app.py — OpenEdit 인터랙티브 데모 (Gradio)

iOS 사진 편집 3기능을 오픈 파운데이션 모델로 재현:
  - Remove  : (준비 중) 객체 검출/세그 + 인페인팅으로 지우기
  - Expand  : 프레임 축소 → 바깥 영역 아웃페인팅(확장)
  - Reframe : SHARP(단일이미지→3D Gaussian)로 시점 변경. 가려졌던 면을 생성.

UX (애플과 동일):
  좌측에 사진 업로드 → 우측 기능 버튼(Reframe) 클릭 → 잠깐 대기(3D 생성+격자 렌더)
  → 좌우/상하 회전 슬라이더로 즉시 미리보기(바깥은 블러) → [완료] 인페인팅 → 다운로드.

VRAM 8GB(RTX 3070 Ti) 대응 / device: CUDA·MPS(맥)·CPU.

실행:
    python app.py
    python app.py --share
"""

from __future__ import annotations

import argparse

import gradio as gr
import numpy as np
from PIL import Image

try:  # 아이폰 사진 등 AVIF 입력 지원 (선택적)
    import pillow_avif  # noqa: F401
except Exception:
    pass

from common import get_device, free_memory, pil_to_numpy, numpy_to_pil, resize_if_needed
import reframe_core as core
import inpaint as rinp
import reframe_ldi as rldi

DEVICE = get_device()

# SHARP 생성형 Reframe 사용 가능 여부 (가중치/패키지 설치 시)
try:
    import reframe_sharp
    import task_nvs_sharp
    SHARP_OK = True
except Exception as _e:
    SHARP_OK = False
    print(f"[app] SHARP 비활성(설치 필요): {_e}")

PREVIEW_MAX = 640          # 미리보기/격자 렌더 긴 변
COMMIT_LONG = 768          # 확정 렌더 긴 변
FALLBACK_PARALLAX = 6.0    # SHARP 미사용 시 깊이 워핑 패럴랙스(고정)


def core_depth_vis(disp: np.ndarray) -> np.ndarray:
    return (np.clip(disp, 0, 1) * 255).astype(np.uint8)[..., None].repeat(3, axis=-1)


# ── 깊이 워핑 폴백 렌더 (SHARP 불가 시) ──────────────────────────────────────────
def _render_depth(img_np, disp, plate, move, smooth, frame_zoom=1.0):
    if plate is not None:
        return rldi.render_ldi(
            plate, move, z_near=1.0, z_far=FALLBACK_PARALLAX,
            frame_zoom=float(frame_zoom), smooth=smooth, device=DEVICE,
        )
    if smooth:
        out, inner, outer = core.warp_image(
            img_np, disp, move, z_near=1.0, z_far=FALLBACK_PARALLAX,
            smooth=True, frame_zoom=float(frame_zoom), return_outer=True, device=DEVICE,
        )
        return out, outer, inner
    out, inner = core.warp_image(
        img_np, disp, move, z_near=1.0, z_far=FALLBACK_PARALLAX, smooth=False, device=DEVICE,
    )
    return out, np.zeros(inner.shape, dtype=bool), inner


def _vlm_caption(image: Image.Image) -> str:
    """SD 인페인팅 프롬프트용 캡션(선택). 실패 시 기본 프롬프트."""
    try:
        import task_vlm
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
        print(f"[app] VLM 캡션 실패 → 기본 프롬프트: {e}")
        return rinp.DEFAULT_PROMPT


def _depth_disp(image: Image.Image):
    import task_depth
    proc, model = task_depth.load_model(DEVICE)
    depth = task_depth.run(image, proc, model, DEVICE)
    del proc, model
    free_memory(DEVICE)
    return core.normalize_disparity(depth)


# ── Reframe: 분석(대기) ─────────────────────────────────────────────────────────
def reframe_analyze(image: Image.Image, do_complete=False, backend="lama",
                    progress=gr.Progress()):
    if image is None:
        raise gr.Error("먼저 왼쪽에 사진을 업로드하세요.")
    orig = image.convert("RGB")
    small = resize_if_needed(orig, max_size=PREVIEW_MAX)
    img_np = pil_to_numpy(small)

    scene = grid = disp = plate = None
    if SHARP_OK:
        try:
            progress(0.25, desc="SHARP — 사진에서 3D 생성 중")
            scene = task_nvs_sharp.predict(orig, device=DEVICE)
            if do_complete:
                try:
                    import reframe_complete
                    progress(0.45, desc="가려진 면 생성(점진적 완성) — 시간이 걸립니다")
                    scene = reframe_complete.complete_scene(
                        scene, backend=(backend if backend != "opencv" else "lama"),
                        out_long=PREVIEW_MAX, device=DEVICE, progress=progress,
                    )
                    free_memory(DEVICE)
                except Exception as e:
                    print(f"[app] 완성 단계 실패(무시): {e}")
            progress(0.5, desc="시점 각도 미리 렌더 중")
            grid = reframe_sharp.build_grid(scene, out_long=PREVIEW_MAX,
                                            device=DEVICE, progress=progress)
            free_memory(DEVICE)
        except Exception as e:
            scene = grid = None
            print(f"[app] SHARP 실패 → 깊이 워핑 폴백: {e}")

    if grid is None:  # 폴백
        progress(0.4, desc="깊이 추정")
        disp = _depth_disp(small)
        try:
            plate = rldi.build_plate(small, disp, DEVICE)
        except Exception:
            plate = None

    # 초기 미리보기 = 정면(0,0)
    canvas0 = _reframe_view(scene, grid, disp, plate, img_np, 0.0, 0.0)
    mode = (f"SHARP 생성형 ({scene.num_gaussians:,} gaussians)"
            if grid is not None else "깊이 워핑(폴백)")
    status = f"준비 완료 · {mode} · 좌우/상하 회전을 드래그하세요."
    return (scene, grid, disp, plate, img_np, canvas0, status,
            gr.update(visible=True), gr.update(visible=False))


# ── Reframe: 실시간 미리보기 ────────────────────────────────────────────────────
def _reframe_view(scene, grid, disp, plate, img_np, yaw, pitch):
    if grid is not None and scene is not None:
        rgb, cov = grid.nearest(yaw, pitch)
        hole = ~cov
        return core.fill_preview(rgb, hole) if hole.any() else rgb
    if img_np is None:
        return None
    move = core.CameraMove(yaw_deg=float(yaw), pitch_deg=float(pitch))
    out, outer, inner = _render_depth(img_np, disp, plate, move, smooth=True)
    blur = outer if plate is not None else (outer | inner)
    return core.fill_preview(out, blur) if blur.any() else out


def reframe_view(scene, grid, disp, plate, img_np, yaw, pitch):
    if img_np is None:
        return None
    return _reframe_view(scene, grid, disp, plate, img_np, yaw, pitch)


# ── Reframe: 완료 & 인페인팅 ────────────────────────────────────────────────────
def reframe_commit(scene, grid, disp, plate, img_np, backend, yaw, pitch,
                   progress=gr.Progress()):
    if img_np is None:
        raise gr.Error("먼저 Reframe를 실행하세요.")
    move = core.CameraMove(yaw_deg=float(yaw), pitch_deg=float(pitch))

    if scene is not None:
        progress(0.3, desc="확정 시점 렌더")
        out, cov = reframe_sharp.render_exact(
            scene, move, pivot_z=grid.pivot_z if grid else None,
            out_long=COMMIT_LONG, device=DEVICE,
        )
        fill = core.dilate_mask(~cov, iterations=2)
    else:
        progress(0.2, desc="새 시점 워핑")
        out, outer, inner = _render_depth(img_np, disp, plate, move,
                                          smooth=plate is not None)
        fill = core.dilate_mask(outer | inner, iterations=3)

    warped_pil = numpy_to_pil(out)
    if int(fill.sum()) < 30:
        return warped_pil, f"완료 · 채울 영역 없음 · device={DEVICE}"

    prompt = None
    if backend == "sd2":
        progress(0.45, desc="VLM 캡션 생성(프롬프트)")
        prompt = _vlm_caption(numpy_to_pil(img_np))

    progress(0.6, desc=f"인페인팅 — {backend}")
    try:
        inp = rinp.get_inpainter(backend, DEVICE)
        result = inp.inpaint(warped_pil, fill, prompt=prompt)
        inp.unload()
    except Exception as e:
        raise gr.Error(f"인페인팅 실패({backend}): {e}")
    used = prompt if prompt else "(LaMa/OpenCV)"
    return result, f"완료 · backend={backend} · prompt={used[:50]} · 우상단 아이콘으로 다운로드"


# ── Expand: 분석 ────────────────────────────────────────────────────────────────
def expand_analyze(image: Image.Image, progress=gr.Progress()):
    if image is None:
        raise gr.Error("먼저 왼쪽에 사진을 업로드하세요.")
    small = resize_if_needed(image.convert("RGB"), max_size=PREVIEW_MAX)
    img_np = pil_to_numpy(small)
    progress(0.4, desc="깊이 추정")
    disp = _depth_disp(small)
    plate = None
    try:
        plate = rldi.build_plate(small, disp, DEVICE)
    except Exception:
        plate = None
    canvas0 = _expand_view(disp, plate, img_np, 1.0)
    return (None, None, disp, plate, img_np, canvas0,
            "확장: 슬라이더로 프레임을 줄이면 바깥이 블러로 보이고, 완료 시 채워집니다.",
            gr.update(visible=False), gr.update(visible=True))


def _expand_view(disp, plate, img_np, extend):
    move = core.CameraMove()
    out, outer, inner = _render_depth(img_np, disp, plate, move, smooth=True,
                                      frame_zoom=float(extend))
    blur = outer if plate is not None else (outer | inner)
    return core.fill_preview(out, blur) if blur.any() else out


def expand_view(disp, plate, img_np, extend):
    if img_np is None:
        return None
    return _expand_view(disp, plate, img_np, extend)


def expand_commit(disp, plate, img_np, backend, extend, progress=gr.Progress()):
    if img_np is None:
        raise gr.Error("먼저 Expand를 실행하세요.")
    move = core.CameraMove()
    progress(0.2, desc="프레임 확장")
    out, outer, inner = _render_depth(img_np, disp, plate, move, smooth=True,
                                      frame_zoom=float(extend))
    fill = core.dilate_mask(outer | inner, iterations=3)
    warped_pil = numpy_to_pil(out)
    if int(fill.sum()) < 30:
        return warped_pil, "완료 · 채울 영역 없음"
    prompt = _vlm_caption(numpy_to_pil(img_np)) if backend == "sd2" else None
    progress(0.6, desc=f"아웃페인팅 — {backend}")
    try:
        inp = rinp.get_inpainter(backend, DEVICE)
        result = inp.inpaint(warped_pil, fill, prompt=prompt)
        inp.unload()
    except Exception as e:
        raise gr.Error(f"아웃페인팅 실패({backend}): {e}")
    return result, f"완료 · backend={backend} · 우상단 아이콘으로 다운로드"


# ── Remove: 준비 중 ─────────────────────────────────────────────────────────────
def remove_clicked():
    return ("Remove(지우기)는 준비 중 — Grounding DINO + SAM2 + LaMa/SD로 객체 제거 예정.",
            gr.update(visible=False), gr.update(visible=False))


# ── UI ───────────────────────────────────────────────────────────────────────────
def build_ui():
    with gr.Blocks(title="OpenEdit — iOS 사진편집 오픈모델 재현") as demo:
        gr.Markdown(
            "# OpenEdit\n"
            "**iOS 사진편집(Reframe · Expand · Remove)을 오픈 파운데이션 모델로 재현** — "
            "SHARP(단일이미지→3D) · Depth Anything V2 · SAM2 · Grounding DINO · Qwen2-VL · LaMa/SD"
        )

        st_scene = gr.State()    # SHARP 3DGS 장면
        st_grid = gr.State()     # SHARP yaw×pitch 격자 캐시
        st_disp = gr.State()     # disparity (폴백/Expand)
        st_plate = gr.State()    # LDI 플레이트 (폴백/Expand)
        st_img = gr.State()      # 원본(리사이즈) uint8 np

        with gr.Row():
            # 좌: 사진 (업로드 = 미리보기 = 결과 = 다운로드)
            with gr.Column(scale=3):
                canvas = gr.Image(
                    type="pil", label="사진", height=560,
                    sources=["upload"], interactive=True,
                )

            # 우: 기능 버튼 + 컨트롤
            with gr.Column(scale=1):
                btn_remove = gr.Button("Remove · 지우기", size="lg")
                btn_expand = gr.Button("Expand · 확장", size="lg")
                btn_reframe = gr.Button("Reframe · 시점", size="lg", variant="primary")

                backend = gr.Radio(
                    ["lama", "sd2", "opencv"], value="lama",
                    label="인페인팅 백엔드 (sd2 = VLM 프롬프트)",
                )
                complete_chk = gr.Checkbox(
                    value=False,
                    label="정밀 완성(실험적): 가려진 면 미리 생성 — 느림, 거품 가능",
                )

                # Reframe 컨트롤 (분석 후 표시)
                with gr.Group(visible=False) as grp_reframe:
                    gr.Markdown("**Reframe** — 한 발짝 옆에서 찍은 듯 시점 변경")
                    yaw = gr.Slider(-16, 16, 0, step=0.5, label="좌우 회전 yaw°")
                    pitch = gr.Slider(-10, 10, 0, step=0.5, label="상하 회전 pitch°")
                    btn_reframe_commit = gr.Button("완료 (인페인팅)", variant="primary")

                # Expand 컨트롤 (분석 후 표시)
                with gr.Group(visible=False) as grp_expand:
                    gr.Markdown("**Expand** — 프레임을 줄여 바깥을 채움")
                    extend = gr.Slider(1.0, 1.6, 1.0, step=0.02, label="프레임 축소 정도")
                    btn_expand_commit = gr.Button("완료 (아웃페인팅)", variant="primary")

                status = gr.Markdown("사진을 업로드하고 기능 버튼을 누르세요.")

        # ── 배선 ──
        reframe_states = [st_scene, st_grid, st_disp, st_plate, st_img]

        btn_reframe.click(
            reframe_analyze,
            inputs=[canvas, complete_chk, backend],
            outputs=[*reframe_states, canvas, status, grp_reframe, grp_expand],
        )
        for s in [yaw, pitch]:
            s.change(
                reframe_view,
                inputs=[*reframe_states, yaw, pitch],
                outputs=canvas, show_progress="hidden",
            )
        btn_reframe_commit.click(
            reframe_commit,
            inputs=[*reframe_states, backend, yaw, pitch],
            outputs=[canvas, status],
        )

        btn_expand.click(
            expand_analyze,
            inputs=[canvas],
            outputs=[*reframe_states, canvas, status, grp_reframe, grp_expand],
        )
        extend.change(
            expand_view,
            inputs=[st_disp, st_plate, st_img, extend],
            outputs=canvas, show_progress="hidden",
        )
        btn_expand_commit.click(
            expand_commit,
            inputs=[st_disp, st_plate, st_img, backend, extend],
            outputs=[canvas, status],
        )

        btn_remove.click(
            remove_clicked, inputs=None,
            outputs=[status, grp_reframe, grp_expand],
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenEdit Gradio app")
    parser.add_argument("--share", action="store_true", help="외부 공유 링크 생성")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    print(f"[app] device = {DEVICE}")
    build_ui().launch(share=args.share, server_port=args.port)
