"""
ui.py — OpenEdit Gradio UI (Clean Up · Expand · Reframe)
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from features import clean_up, expand, reframe
from features.reframe import (
    YAW_IDX_MIN, YAW_IDX_MAX, YAW_STEP,
    PITCH_IDX_MIN, PITCH_IDX_MAX, PITCH_STEP,
    YAW_RANGE_DEG, PITCH_RANGE_DEG,
    ANGLE_STEP,
)


_ASSETS = Path(__file__).resolve().parent / "assets"
UI_CSS = _ASSETS / "ui.css"


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="OpenEdit — iOS 사진편집") as demo:
        gr.Markdown("# OpenEdit")

        st_scene = gr.State()
        st_disp = gr.State()
        st_plate = gr.State()
        st_img = gr.State()
        st_mask = gr.State()
        st_mode = gr.State(value=None)

        edit_states = [st_scene, st_disp, st_plate, st_img, st_mask, st_mode]

        with gr.Row():
            with gr.Column(scale=3):
                canvas = gr.Image(
                    type="pil", label="사진", height=560,
                    sources=["upload"], interactive=True,
                )
                clean_up_editor = gr.ImageEditor(
                    label="Clean Up — 브러시로 객체 표시",
                    height=560,
                    sources=["upload"],
                    type="numpy",
                    brush=gr.Brush(colors=["#FF0000"], color_mode="fixed"),
                    eraser=False,
                    layers=True,
                    visible=False,
                )
                with gr.Row(elem_classes=["tool-bar"], elem_id="openedit-toolbar"):
                    btn_clean_up = gr.Button(
                        "Clean Up", size="lg",
                        elem_id="tool-clean-up",
                        elem_classes=["tool-btn", "tool-clean-up"],
                    )
                    btn_expand = gr.Button(
                        "Extend", size="lg",
                        elem_id="tool-expand",
                        elem_classes=["tool-btn", "tool-expand"],
                    )
                    btn_reframe = gr.Button(
                        "Reframe", size="lg",
                        elem_id="tool-reframe",
                        elem_classes=["tool-btn", "tool-reframe"],
                    )

            with gr.Column(scale=1):

                with gr.Group(visible=False) as grp_clean_up:
                    gr.Markdown("**Clean Up** — 브러시/텍스트로 객체 선택 후 제거")
                    clean_up_prompt = gr.Textbox(
                        label="텍스트 검색 (선택)",
                        placeholder="person . bottle . cup .",
                    )
                    with gr.Row():
                        btn_clean_up_detect = gr.Button("텍스트로 검색")
                        btn_clean_up_clear = gr.Button("선택 초기화")
                    btn_clean_up_commit = gr.Button("완료 (제거)", variant="primary")

                with gr.Group(visible=False) as grp_expand:
                    gr.Markdown("**Expand** — LaMa 미리보기 → [완료] DreamShaper SD1.5")
                    extend = gr.Slider(
                        1.0, 1.6, 1.0, step=0.02,
                        label="프레임 축소 (1.0=원본, 1.6=최대)",
                    )
                    btn_expand_commit = gr.Button("완료 (SD 생성)", variant="primary")

                with gr.Group(visible=False) as grp_reframe:
                    gr.Markdown("**Reframe** — yaw 좌우 · pitch 상하")
                    gr.Markdown(
                        f"슬라이더 **수치** (각도 아님) · **1당 {ANGLE_STEP:g}°** · "
                        f"좌우 **{YAW_IDX_MIN}~{YAW_IDX_MAX}** (최대 ±{YAW_RANGE_DEG:g}°) · "
                        f"상하 **{PITCH_IDX_MIN}~{PITCH_IDX_MAX}** (최대 ±{PITCH_RANGE_DEG:g}°) · "
                        f"{(YAW_IDX_MAX - YAW_IDX_MIN + 1) * (PITCH_IDX_MAX - PITCH_IDX_MIN + 1)}프레임"
                    )
                    yaw = gr.Slider(
                        minimum=YAW_IDX_MIN, maximum=YAW_IDX_MAX, value=0, step=YAW_STEP,
                        label=f"좌우 ({YAW_IDX_MIN}~{YAW_IDX_MAX}, ×{ANGLE_STEP:g}°)",
                    )
                    pitch = gr.Slider(
                        minimum=PITCH_IDX_MIN, maximum=PITCH_IDX_MAX, value=0, step=PITCH_STEP,
                        label=f"상하 ({PITCH_IDX_MIN}~{PITCH_IDX_MAX}, ×{ANGLE_STEP:g}°)",
                    )
                    btn_reframe_commit = gr.Button("완료 — DreamShaper 바깥 생성", variant="primary")

                status = gr.Markdown("사진을 업로드한 뒤 기능 버튼을 누르세요.")

        canvas_vis = [canvas, clean_up_editor]
        mode_outs = [status, grp_reframe, grp_expand, grp_clean_up]
        angle_outs = [yaw, pitch]
        analyze_outs = [*edit_states, *canvas_vis, *mode_outs, *angle_outs]

        btn_clean_up.click(clean_up.clean_up_prepare, inputs=[canvas], outputs=analyze_outs)
        btn_expand.click(expand.expand_analyze, inputs=[canvas], outputs=analyze_outs)
        btn_reframe.click(reframe.reframe_analyze, inputs=[canvas], outputs=analyze_outs)

        clean_up_editor.input(
            clean_up.clean_up_brush,
            inputs=[st_mode, st_img, st_mask, clean_up_editor],
            outputs=[clean_up_editor, st_mask, status],
            trigger_mode="always_last",
            show_progress="minimal",
        )
        btn_clean_up_detect.click(
            clean_up.clean_up_detect,
            inputs=[st_img, clean_up_prompt],
            outputs=[clean_up_editor, st_mask, status],
        )
        btn_clean_up_clear.click(
            clean_up.clean_up_clear,
            inputs=[st_img],
            outputs=[clean_up_editor, st_mask, status],
        )
        btn_clean_up_commit.click(
            clean_up.clean_up_commit,
            inputs=[st_img, st_mask, clean_up_editor],
            outputs=[canvas, clean_up_editor, st_img, status],
        )

        extend.change(
            expand.expand_view,
            inputs=[st_disp, st_plate, st_img, extend],
            outputs=canvas, show_progress="hidden",
        )
        btn_expand_commit.click(
            expand.expand_commit,
            inputs=[st_disp, st_plate, st_img, extend],
            outputs=[canvas, status],
        )

        reframe_in = [st_scene, st_disp, st_plate, st_img, yaw, pitch]
        for ctrl in (yaw, pitch):
            ctrl.change(
                reframe.reframe_view,
                inputs=reframe_in,
                outputs=canvas, show_progress="hidden",
            )
        btn_reframe_commit.click(
            reframe.reframe_commit,
            inputs=[st_scene, st_disp, st_img, yaw, pitch],
            outputs=[canvas, status],
            show_progress="full",
        )

    return demo
