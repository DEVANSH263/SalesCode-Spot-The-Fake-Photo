"""
Spot the Fake Photo — live camera demo (Gradio).

Run locally:
    python app.py
Then open http://127.0.0.1:7860, click the camera icon once to grant
permission, and the live score updates automatically — no further clicks.

Deploy free on Hugging Face Spaces:
    See DEPLOY.md for step-by-step instructions.
"""

import os
import time
import traceback
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import numpy as np
import gradio as gr
from PIL import Image

from predict import predict

_DIR = Path(__file__).parent

# Force the rear ("environment") camera by default on phones — Gradio has no
# built-in control for this, so we patch getUserMedia in the page <head>.
# Users can still switch back to the front camera using their browser's own
# camera-switch control if their browser exposes one.
_PREFER_BACK_CAMERA_JS = """
<script>
const _origGetUserMedia = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
navigator.mediaDevices.getUserMedia = (constraints) => {
    if (constraints && constraints.video && typeof constraints.video === "object" && !constraints.video.facingMode) {
        constraints.video.facingMode = { ideal: "environment" };
    }
    return _origGetUserMedia(constraints);
};
</script>
"""


def classify(img):
    """Run one frame through the model. Always returns a dict for gr.Label
    and a short status string — never raises, so the stream never stalls.

    Gradio's webcam stream can hand us either a numpy array or a PIL Image
    depending on version/config, so we normalise to PIL first."""
    if img is None:
        return {"Waiting for camera…": 1.0}, "Point the camera at something."

    try:
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)

        tmp_path = _DIR / "_live_frame.jpg"
        img.convert("RGB").save(tmp_path, quality=90)

        t0 = time.perf_counter()
        score = float(predict(str(tmp_path)))
        ms = (time.perf_counter() - t0) * 1000

        probs = {"📱 Screen recapture": score, "✅ Real photo": 1.0 - score}
        status = f"Fraud score: {score:.3f}  •  {ms:.0f} ms"
        return probs, status

    except Exception as exc:                      # noqa: BLE001
        traceback.print_exc()
        return {"⚠️ Error": 1.0}, f"Could not process frame: {exc}"


with gr.Blocks(title="Spot the Fake Photo") as demo:
    gr.Markdown(
        """
        # 📷 Spot the Fake Photo
        Click the camera box **once** to grant access — after that, every
        frame is scored automatically, live, with no further clicks.

        Point it at something real, or at another screen/printout showing a
        photo, and watch the prediction update.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            cam = gr.Image(
                sources=["webcam"],
                streaming=True,
                label="Camera (click once to start)",
                type="numpy",
            )
        with gr.Column(scale=1):
            result = gr.Label(label="Live prediction", num_top_classes=2)
            status = gr.Markdown("Waiting for camera…")

    # stream_every controls how often a new frame is sent (seconds).
    # time_limit is set high so the stream keeps running continuously
    # instead of stopping and asking the user to click "Record" again.
    cam.stream(
        classify,
        inputs=cam,
        outputs=[result, status],
        stream_every=0.5,
        time_limit=3600,
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="emerald"), head=_PREFER_BACK_CAMERA_JS)
