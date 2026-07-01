"""Spot the Fake Photo — Screen vs Real Photo Detector.

Usage
-----
    python predict.py some_image.jpg

Prints ONE number from 0 to 1:
    0 = real photo
    1 = photo of a screen (recapture / fraud)

Model priority (loaded once, cached):
    1. model_nn.pt  – fine-tuned EfficientNet-B0 (more accurate, ~70ms)
    2. model.pkl    – RF + SVM ensemble on classical+CNN features (~110ms)
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image   # noqa: F401  (used inside features.py as fallback)

_DIR = Path(__file__).parent

# ── Lazy-loaded model handles ─────────────────────────────────────────────────
_nn_model      = None   # EfficientNet-B0 state dict wrapper
_feature_model = None   # sklearn VotingClassifier (model.pkl)


# ── Neural network path ───────────────────────────────────────────────────────

def _load_nn():
    global _nn_model
    if _nn_model is not None:
        return _nn_model
    pt_path = _DIR / "model_nn.pt"
    if not pt_path.exists():
        return None
    try:
        import torch
        from torchvision import models, transforms
        from torchvision.models import EfficientNet_B0_Weights
        import torch.nn as nn

        ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
        net  = models.efficientnet_b0(weights=None)
        net.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(1280, 1),
        )
        net.load_state_dict(ckpt["state_dict"])
        net.eval()

        tfm = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
        _nn_model = (net, tfm)
    except Exception as exc:
        print(f"[predict] NN model load failed ({exc}); falling back to pkl",
              file=sys.stderr)
        _nn_model = None
    return _nn_model


def _predict_nn(image_path: str) -> float:
    handle = _load_nn()
    if handle is None:
        return None
    net, tfm = handle
    import torch
    img = Image.open(image_path).convert("RGB")
    tensor = tfm(img).unsqueeze(0)
    with torch.no_grad():
        logit = net(tensor).squeeze()
    return float(torch.sigmoid(logit).item())


# ── Feature-based path ────────────────────────────────────────────────────────

def _load_feature_model():
    global _feature_model
    if _feature_model is None:
        import pickle
        with open(_DIR / "model.pkl", "rb") as fh:
            _feature_model = pickle.load(fh)
    return _feature_model


def _predict_feature(image_path: str) -> float:
    sys.path.insert(0, str(_DIR))
    from features import extract_features
    feat  = extract_features(image_path)
    feat  = np.nan_to_num(feat, nan=0.0, posinf=1e6, neginf=-1e6).reshape(1, -1)
    model = _load_feature_model()
    return float(model.predict_proba(feat)[0][1])


# ── Public API ────────────────────────────────────────────────────────────────

def predict(image_path: str) -> float:
    """
    Return a fraud score in [0, 1].

    Parameters
    ----------
    image_path : str
        Path to the image file to classify.

    Returns
    -------
    float
        0  → real photo
        1  → photo of a screen (recapture)
    """
    # Try fine-tuned NN first (more accurate)
    score = _predict_nn(image_path)
    if score is not None:
        return score
    # Fall back to classical+CNN feature ensemble
    return _predict_feature(image_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>", file=sys.stderr)
        sys.exit(1)
    # Full-precision plain decimal (e.g. 0.9999986886978149) instead of
    # scientific notation (e.g. 3.2e-05), so the score is never displayed
    # with an exponent while keeping the full floating-point precision.
    score = predict(sys.argv[1])
    text = f"{score:.20f}".rstrip("0")
    if text.endswith("."):
        text += "0"
    print(text)
