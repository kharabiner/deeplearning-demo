"""
app.py — OpenEdit (Clean Up · Expand · Reframe)
"""

from __future__ import annotations

import argparse

# Gradio 6.x still uses HTTP_422_UNPROCESSABLE_ENTITY; Starlette 1.x deprecates it
# via __getattr__. Bind the alias before Gradio loads (avoid hasattr — it triggers the warn).
import starlette.status as _starlette_status

_starlette_status.__dict__["HTTP_422_UNPROCESSABLE_ENTITY"] = (
    _starlette_status.HTTP_422_UNPROCESSABLE_CONTENT
)

try:
    import pillow_avif  # noqa: F401
except Exception:
    pass

from features.shared import DEVICE
from ui import build_ui


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenEdit Gradio app")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    print(f"[app] device = {DEVICE} · OpenEdit (Clean Up · Expand · Reframe)")
    port = args.port if args.port != 7860 else None
    build_ui().launch(share=args.share, server_port=port)
