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

# Default to the rear ("environment") camera on phones — most people testing
# real vs. screen photos hold the phone up and look at the back camera feed.
# Gradio's native WebcamOptions passes this straight to the browser's
# getUserMedia() call, no custom JS needed.
_WEBCAM_OPTS = gr.WebcamOptions(
    mirror=False,
    constraints={"video": {"facingMode": {"ideal": "environment"}}},
)

# Mobile-friendly layout: stack the camera above the result (instead of
# side-by-side columns that get squeezed on narrow screens), and let the
# camera preview fill most of the viewport height.
_CSS = """
.gradio-container { max-width: 720px !important; margin: auto; }
#camera-box { width: 100% !important; }
#camera-box video, #camera-box img { width: 100% !important; object-fit: cover; }
@media (max-width: 640px) {
    #camera-box video, #camera-box img { height: 70vh !important; }
}
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
        Your browser will show a small **"Record"** button below — that's
        just a one-time camera-permission click (every site that uses a
        camera, e.g. Zoom or Google Meet, requires this by browser security
        rules — it is **not** ongoing recording, nothing is saved).
        After that single click, every frame is scored automatically and
        live, with **no further clicks needed**. On phones this opens your
        **back camera** by default.
        """
    )

    cam = gr.Image(
        sources=["webcam"],
        streaming=True,
        label="Camera — click \"Record\" once for permission, then it's fully live",
        type="numpy",
        webcam_options=_WEBCAM_OPTS,
        elem_id="camera-box",
    )
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
    demo.launch(theme=gr.themes.Soft(primary_hue="emerald"), css=_CSS)
