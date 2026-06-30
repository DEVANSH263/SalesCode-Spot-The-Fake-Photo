"""
Spot the Fake Photo — live camera demo (Gradio).

Run locally:
    python app.py
Then open http://127.0.0.1:7860 and allow camera access.

Deploy free on Hugging Face Spaces:
    See DEPLOY.md for step-by-step instructions.
"""

import os
import time
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
from PIL import Image

from predict import predict

_DIR = Path(__file__).parent


def classify(img: Image.Image):
    if img is None:
        return "Waiting for a photo…", None

    tmp_path = _DIR / "_live_frame.jpg"
    img.convert("RGB").save(tmp_path, quality=90)

    t0 = time.perf_counter()
    score = predict(str(tmp_path))
    ms = (time.perf_counter() - t0) * 1000

    is_screen = score >= 0.5
    label = "📱 SCREEN (recapture)" if is_screen else "✅ REAL photo"
    color = "#d9363e" if is_screen else "#2e8b57"

    html = f"""
    <div style="text-align:center; font-family:sans-serif;">
        <div style="font-size:28px; font-weight:700; color:{color};">{label}</div>
        <div style="font-size:16px; margin-top:6px;">
            fraud score = <b>{score:.3f}</b> &nbsp;|&nbsp; {ms:.0f} ms
        </div>
    </div>
    """
    return html


with gr.Blocks(title="Spot the Fake Photo") as demo:
    gr.Markdown(
        """
        # 📷 Spot the Fake Photo
        Point your camera at something real, or at another screen showing a photo.
        The model scores every frame: **0 = real photo, 1 = screen recapture**.
        """
    )
    with gr.Row():
        cam = gr.Image(sources=["webcam"], streaming=True, label="Camera")
        out = gr.HTML(label="Result")

    cam.stream(classify, inputs=cam, outputs=out, time_limit=30, stream_every=0.5)

if __name__ == "__main__":
    demo.launch()
