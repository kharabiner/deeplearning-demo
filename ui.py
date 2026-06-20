"""
ui.py — OpenEdit Gradio UI (Clean Up · Expand · Reframe)
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from common import free_memory, numpy_to_pil
from features import clean_up, expand, reframe
from features.reframe import (
    YAW_IDX_MIN, YAW_IDX_MAX, YAW_STEP,
    PITCH_IDX_MIN, PITCH_IDX_MAX, PITCH_STEP,
)
from features.shared import DEVICE


_ASSETS = Path(__file__).resolve().parent / "assets"
UI_CSS = _ASSETS / "ui.css"

HIDE = gr.update(visible="hidden")
SHOW = gr.update(visible=True)
_MODE_GROUPS_HIDE = (HIDE, HIDE, HIDE)


def _layout_vis(update):
    """Group/Row용 visible=False → hidden 변환."""
    if isinstance(update, dict) and update.get("visible") is False:
        return HIDE
    if isinstance(update, dict) and update.get("visible") is True:
        return SHOW
    return update


def _prepare_with_toolbar_hidden(fn):
    """모드 진입 시 메인 3버튼 숨김."""
    def wrapped(image, progress=gr.Progress()):
        result = list(fn(image, progress))
        del result[8]  # status 메시지 제거
        result.insert(8, HIDE)  # grp_toolbar
        for i in (9, 10, 11):
            result[i] = _layout_vis(result[i])
        return tuple(result)
    return wrapped


def _exit_mode(img_np):
    """모드 취소 — 메인 3버튼 복귀."""
    clean_up.unload_sam()
    import inpaint as rinp
    rinp.unload_expand_sd15()
    free_memory(DEVICE)
    canvas_val = numpy_to_pil(img_np) if img_np is not None else None
    return (
        None, None, None, gr.skip(), None, None,
        gr.update(value=canvas_val, visible=True),
        HIDE,
        SHOW, *_MODE_GROUPS_HIDE,
        gr.update(value=0), gr.update(value=0),
    )


def _finish_clean_up(*args, **kwargs):
    canvas_up, editor_up, img, _status = clean_up.clean_up_commit(*args, **kwargs)
    return canvas_up, editor_up, img, SHOW, *_MODE_GROUPS_HIDE


def _finish_expand(*args, **kwargs):
    result, _status = expand.expand_commit(*args, **kwargs)
    return result, SHOW, *_MODE_GROUPS_HIDE


def _finish_reframe(*args, **kwargs):
    result, _status = reframe.reframe_commit(*args, **kwargs)
    return result, SHOW, *_MODE_GROUPS_HIDE


def _brush_out(mode, img_np, mask, editor_value, progress=gr.Progress()):
    editor, new_mask, _status = clean_up.clean_up_brush(
        mode, img_np, mask, editor_value, progress,
    )
    return editor, new_mask


def _clear_out(img_np):
    editor, mask, _status = clean_up.clean_up_clear(img_np)
    return editor, mask


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="OpenEdit — iOS 사진편집") as demo:
        st_scene = gr.State()
        st_disp = gr.State()
        st_plate = gr.State()
        st_img = gr.State()
        st_mask = gr.State()
        st_mode = gr.State(value=None)

        edit_states = [st_scene, st_disp, st_plate, st_img, st_mask, st_mode]

        with gr.Column(elem_id="openedit-main"):
            canvas = gr.Image(
                type="pil", label="사진", height=560,
                sources=["upload"], interactive=True,
            )
            clean_up_editor = gr.ImageEditor(
                label="Clean Up — 브러시로 객체 표시",
                height=560,
                sources=["upload"],
                type="numpy",
                brush=gr.Brush(colors=["#FF0000"], color_mode="fixed", default_size=8),
                eraser=False,
                layers=True,
                visible=False,
            )

            with gr.Group(visible=True, elem_id="toolbar-wrap", elem_classes=["toolbar-wrap"]) as grp_toolbar:
                with gr.Row(
                    elem_classes=["tool-bar"],
                    elem_id="openedit-toolbar",
                    show_progress=False,
                ):
                    btn_clean_up = gr.Button(
                        "Clean Up", size="lg", scale=0,
                        elem_id="tool-clean-up",
                        elem_classes=["tool-btn", "tool-clean-up"],
                    )
                    btn_expand = gr.Button(
                        "Extend", size="lg", scale=0,
                        elem_id="tool-expand",
                        elem_classes=["tool-btn", "tool-expand"],
                    )
                    btn_reframe = gr.Button(
                        "Reframe", size="lg", scale=0,
                        elem_id="tool-reframe",
                        elem_classes=["tool-btn", "tool-reframe"],
                    )

            with gr.Group(visible="hidden", elem_id="panel-clean-up") as grp_clean_up:
                with gr.Row(elem_id="clean-up-actions"):
                    btn_clean_up_clear = gr.Button(
                        "Clear", elem_classes=["mode-btn", "mode-btn-clear"],
                    )
                    btn_clean_up_commit = gr.Button(
                        "Done", elem_classes=["mode-btn", "mode-btn-done"],
                    )
                    btn_clean_up_cancel = gr.Button(
                        "Back", elem_classes=["mode-btn", "mode-btn-back"],
                    )

            with gr.Group(visible="hidden", elem_id="panel-expand") as grp_expand:
                with gr.Row(elem_id="expand-layout", equal_height=True):
                    with gr.Column(scale=4, elem_id="expand-sliders-col"):
                        extend = gr.Slider(
                            1.0, 1.6, 1.0, step=0.02,
                            label="Scale",
                            elem_id="slider-expand",
                            elem_classes=["mode-slider"],
                            container=False,
                            buttons=[],
                            min_width=0,
                        )
                    with gr.Column(scale=1, min_width=100, elem_id="expand-actions-col"):
                        with gr.Row(elem_id="expand-actions"):
                            btn_expand_commit = gr.Button(
                                "Done", elem_classes=["mode-btn", "mode-btn-done"],
                            )
                            btn_expand_cancel = gr.Button(
                                "Back", elem_classes=["mode-btn", "mode-btn-back"],
                            )

            with gr.Group(visible="hidden", elem_id="panel-reframe") as grp_reframe:
                with gr.Row(elem_id="reframe-layout", equal_height=True):
                    with gr.Column(scale=4, elem_id="reframe-sliders-col"):
                        yaw = gr.Slider(
                            minimum=YAW_IDX_MIN, maximum=YAW_IDX_MAX, value=0, step=YAW_STEP,
                            label="Horizontal",
                            elem_id="slider-yaw",
                            elem_classes=["mode-slider"],
                            container=False,
                            buttons=[],
                            min_width=0,
                        )
                        pitch = gr.Slider(
                            minimum=PITCH_IDX_MIN, maximum=PITCH_IDX_MAX, value=0, step=PITCH_STEP,
                            label="Vertical",
                            elem_id="slider-pitch",
                            elem_classes=["mode-slider"],
                            container=False,
                            buttons=[],
                            min_width=0,
                        )
                    with gr.Column(scale=1, min_width=100, elem_id="reframe-actions-col"):
                        with gr.Row(elem_id="reframe-actions"):
                            btn_reframe_commit = gr.Button(
                                "Done", elem_classes=["mode-btn", "mode-btn-done"],
                            )
                            btn_reframe_cancel = gr.Button(
                                "Back", elem_classes=["mode-btn", "mode-btn-back"],
                            )

        canvas_vis = [canvas, clean_up_editor]
        mode_outs = [grp_toolbar, grp_reframe, grp_expand, grp_clean_up]
        angle_outs = [yaw, pitch]
        analyze_outs = [*edit_states, *canvas_vis, *mode_outs, *angle_outs]
        exit_outs = analyze_outs

        btn_clean_up.click(
            _prepare_with_toolbar_hidden(clean_up.clean_up_prepare),
            inputs=[canvas], outputs=analyze_outs, show_progress="full",
        )
        btn_expand.click(
            _prepare_with_toolbar_hidden(expand.expand_analyze),
            inputs=[canvas], outputs=analyze_outs, show_progress="full",
        )
        btn_reframe.click(
            _prepare_with_toolbar_hidden(reframe.reframe_analyze),
            inputs=[canvas], outputs=analyze_outs, show_progress="full",
        )

        for btn in (btn_clean_up_cancel, btn_expand_cancel, btn_reframe_cancel):
            btn.click(_exit_mode, inputs=[st_img], outputs=exit_outs)

        clean_up_editor.input(
            _brush_out,
            inputs=[st_mode, st_img, st_mask, clean_up_editor],
            outputs=[clean_up_editor, st_mask],
            trigger_mode="always_last",
            show_progress="minimal",
        )
        btn_clean_up_clear.click(
            _clear_out,
            inputs=[st_img],
            outputs=[clean_up_editor, st_mask],
        )
        btn_clean_up_commit.click(
            _finish_clean_up,
            inputs=[st_img, st_mask, clean_up_editor],
            outputs=[
                canvas, clean_up_editor, st_img,
                grp_toolbar, grp_reframe, grp_expand, grp_clean_up,
            ],
            show_progress="full",
        )

        extend.change(
            expand.expand_view,
            inputs=[st_disp, st_plate, st_img, extend],
            outputs=canvas, show_progress="hidden",
        )
        btn_expand_commit.click(
            _finish_expand,
            inputs=[st_disp, st_plate, st_img, extend],
            outputs=[canvas, grp_toolbar, grp_reframe, grp_expand, grp_clean_up],
            show_progress="full",
        )

        reframe_in = [st_scene, st_disp, st_plate, st_img, yaw, pitch]
        for ctrl in (yaw, pitch):
            ctrl.change(
                reframe.reframe_view,
                inputs=reframe_in,
                outputs=canvas, show_progress="hidden",
            )
        btn_reframe_commit.click(
            _finish_reframe,
            inputs=[st_scene, st_disp, st_img, yaw, pitch],
            outputs=[canvas, grp_toolbar, grp_reframe, grp_expand, grp_clean_up],
            show_progress="full",
        )

    return demo
