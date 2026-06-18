"""
ui.py — OpenEdit 프론트엔드 화면 구성 (Gradio)

iOS 27 사진편집 3기능(Clean Up · Expand · Reframe)의 화면 레이아웃과
버튼/슬라이더 이벤트 와이어링만 담당한다. 실제 기능 로직은 features/* 에 있다.

UX (애플과 동일):
  좌측에 사진 업로드 → 우측 기능 버튼 클릭 → 잠깐 대기
  → 슬라이더/브러시로 즉시 미리보기(빈 곳은 블러) → [완료] SD2 인페인팅 → 다운로드.
"""

from __future__ import annotations

import gradio as gr

from features import clean_up, expand, reframe


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="OpenEdit — iOS 사진편집") as demo:
        gr.Markdown(
            "# OpenEdit\n"
            "**iOS 27 사진편집 (Clean Up · Expand · Reframe)** — "
            "SHARP · Depth Anything V2 · SAM2 · Grounding DINO · Qwen2-VL · SD2"
        )

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

            with gr.Column(scale=1):
                btn_clean_up = gr.Button("Clean Up · 지우기", size="lg")
                btn_expand = gr.Button("Expand · 확장", size="lg")
                btn_reframe = gr.Button("Reframe · 시점 변경", size="lg", variant="primary")

                gr.Markdown("**인페인팅: SD2 전용** (VLM 프롬프트)")

                with gr.Group(visible=False) as grp_clean_up:
                    gr.Markdown("**Clean Up** — 지우고 싶은 객체를 브러시로 문질러서 선택")
                    clean_up_prompt = gr.Textbox(
                        label="텍스트 검색 (선택)",
                        placeholder="person . bottle . cup .",
                    )
                    with gr.Row():
                        btn_clean_up_detect = gr.Button("텍스트로 검색")
                        btn_clean_up_clear = gr.Button("선택 초기화")
                    btn_clean_up_commit = gr.Button("완료 (제거)", variant="primary")

                with gr.Group(visible=False) as grp_expand:
                    gr.Markdown(
                        "**Expand** — LaMa 미리보기 배경 → 슬라이더 조절 → "
                        "[완료] SD 고품질 생성"
                    )
                    extend = gr.Slider(
                        1.0, 1.6, 1.0, step=0.02,
                        label="프레임 축소 (1.0=원본, 1.6=최대 확장)",
                    )
                    btn_expand_commit = gr.Button("완료 (SD 생성)", variant="primary")

                with gr.Group(visible=False) as grp_reframe:
                    gr.Markdown("**Reframe** — 마치 사진을 찍기 전 각도를 바꾸는 것처럼 시점 변경 (SHARP 단일 루트)")
                    gr.Markdown("💡 슬라이더를 움직이면 **SHARP 미리보기**(사전 렌더 격자)로 즉시 확인 · 바깥은 블러. [완료]는 정확한 각도로 재렌더 + SD2 생성")
                    yaw = gr.Slider(-16, 16, 0, step=0.5, label="좌우 회전 yaw° (소폭 권장: ±12° 이내)")
                    pitch = gr.Slider(-10, 10, 0, step=0.5, label="상하 회전 pitch° (소폭 권장: ±8° 이내)")
                    btn_reframe_commit = gr.Button("완료 — SD2로 바깥 영역 생성 (로딩 시간 소요)", variant="primary")

                status = gr.Markdown("사진을 업로드한 뒤 기능 버튼을 누르세요.")

        canvas_vis = [canvas, clean_up_editor]
        mode_outs = [status, grp_reframe, grp_expand, grp_clean_up]
        analyze_outs = [*edit_states, *canvas_vis, *mode_outs]

        # ── 기능 버튼 → 분석/준비 ──
        btn_clean_up.click(clean_up.clean_up_prepare, inputs=[canvas], outputs=analyze_outs)
        btn_expand.click(expand.expand_analyze, inputs=[canvas], outputs=analyze_outs)
        btn_reframe.click(reframe.reframe_analyze, inputs=[canvas], outputs=analyze_outs)

        # ── Clean Up ──
        clean_up_editor.change(
            clean_up.clean_up_brush,
            inputs=[st_mode, st_img, st_mask, clean_up_editor],
            outputs=[clean_up_editor, st_mask, status],
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
            inputs=[st_img, st_mask],
            outputs=[canvas, clean_up_editor, st_img, status],
        )

        # ── Expand ──
        # 매 틱은 가벼운 작업(블러 배경 재사용 + BILINEAR 축소)이라 .change 로도 부드럽다.
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

        # ── Reframe ──
        reframe_view_in = [st_scene, st_disp, st_plate, st_img, yaw, pitch]
        for s in [yaw, pitch]:
            s.change(
                reframe.reframe_view,
                inputs=reframe_view_in,
                outputs=canvas, show_progress="hidden",
            )
        btn_reframe_commit.click(
            reframe.reframe_commit,
            inputs=[st_scene, st_disp, st_img, yaw, pitch],
            outputs=[canvas, status],
        )

    return demo
