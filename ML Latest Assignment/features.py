"""
Screen-vs-Real photo feature extraction.

Multi-signal approach:
  1.  FFT spectrum           – screens have characteristic frequency signatures
  2.  High-frequency noise   – camera sensor noise vs screen-captured noise
  3.  Color / saturation     – backlit screen colours vs natural reflected light
  4.  LBP texture histogram  – screen pixel grid regularity vs natural texture
  5.  DCT block artifacts    – double-JPEG compression (screen → camera)
  6.  Local texture stats    – uniformity patterns from screen vs real scenes
  7.  1-D periodicity        – row/column regularity from screen grid lines
  8.  Gradient orientation   – screens have mostly 0°/90° edges (UI/bezel)
  9.  Dark-border detection  – phone/laptop bezels create a dark frame
  10. Brightness bimodality  – dark bezel + lit screen content
  11. MobileNetV2 CNN feats  – 128-dim semantic embedding (ImageNet pretrained)

All classical analysis uses a fixed ANALYSIS_SIZE×ANALYSIS_SIZE resize.
CNN features use 224×224 (standard ImageNet input).
"""

import numpy as np
import cv2
from scipy import stats
from skimage.feature import local_binary_pattern

ANALYSIS_SIZE = 256       # resize target for classical analysis
LBP_POINTS    = 16
LBP_RADIUS    = 2
LBP_BINS      = LBP_POINTS + 2   # 18 bins for uniform LBP with P=16

# ── CNN feature extractor (loaded lazily) ────────────────────────────────────
_cnn_model   = None
_cnn_transform = None

def _get_cnn():
    """Lazy-load a MobileNetV2 feature extractor (CPU, eval mode)."""
    global _cnn_model, _cnn_transform
    if _cnn_model is not None:
        return _cnn_model, _cnn_transform
    try:
        import torch
        import torchvision.models as tvm
        from torchvision import transforms

        weights = tvm.MobileNet_V2_Weights.IMAGENET1K_V1
        full    = tvm.mobilenet_v2(weights=weights)
        full.eval()

        # Keep everything up to (and including) the adaptive-avg-pool → 1280-d
        # Remove the final classifier; output of features[18] + pool = 1280-d
        class _Extractor(torch.nn.Module):
            def __init__(self, base):
                super().__init__()
                self.features = base.features
                self.pool     = torch.nn.AdaptiveAvgPool2d((1, 1))
            def forward(self, x):
                x = self.features(x)
                x = self.pool(x)
                return x.view(x.size(0), -1)

        _cnn_model = _Extractor(full)
        _cnn_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225]),
        ])
        return _cnn_model, _cnn_transform
    except Exception:
        return None, None


def _cnn_features(img_rgb_u8: np.ndarray) -> np.ndarray:
    """
    Return a 1280-d float32 MobileNetV2 embedding, or zeros if torch unavailable.
    img_rgb_u8 : H×W×3 uint8 RGB array.
    """
    model, transform = _get_cnn()
    if model is None:
        return np.zeros(1280, dtype=np.float32)
    import torch
    tensor = transform(img_rgb_u8).unsqueeze(0)   # (1, 3, 224, 224)
    with torch.no_grad():
        vec = model(tensor).squeeze().numpy()       # (1280,)
    return vec.astype(np.float32)


def extract_features(img_path: str) -> np.ndarray:
    """
    Extract a float32 feature vector from *img_path*.

    Returns
    -------
    np.ndarray, shape (N,), dtype float32
        Feature vector (N ≈ 57).

    Raises
    ------
    ValueError if the image cannot be loaded.
    """
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        try:                                         # fallback: PIL handles EXIF rotation
            from PIL import Image as PILImage
            pil = PILImage.open(str(img_path)).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        except Exception as exc:
            raise ValueError(f"Cannot load {img_path}: {exc}") from exc

    S = ANALYSIS_SIZE
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (S, S), interpolation=cv2.INTER_AREA)
    gray    = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)

    feats = []

    # ── 1. 2-D FFT spectrum ───────────────────────────────────────────────────
    fft_s = np.fft.fftshift(np.fft.fft2(gray))
    mag   = np.abs(fft_s)
    logm  = np.log1p(mag)

    cy, cx = S // 2, S // 2
    Y, X   = np.ogrid[:S, :S]
    R      = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    total  = mag.sum() + 1e-12

    # Radial band energy fractions (6 bands)
    for lo, hi in [(0, 8), (8, 16), (16, 32), (32, 64), (64, 96), (96, 128)]:
        feats.append(float(mag[(R >= lo) & (R < hi)].sum() / total))

    # High-to-low frequency energy ratio
    feats.append(float(mag[R >= 64].sum() / (mag[R < 16].sum() + 1e-12)))

    # Peak prominence outside DC (screen grid creates spectral peaks)
    logm_ndc = logm.copy()
    logm_ndc[cy - 4:cy + 5, cx - 4:cx + 5] = 0.0
    feats.append(float(logm_ndc.max() / (logm_ndc.mean() + 1e-12)))

    # Coefficient of variation in the high-frequency ring
    hf = mag[R >= 64]
    feats.append(float(hf.std() / (hf.mean() + 1e-12)))

    # ── 2. High-frequency noise ───────────────────────────────────────────────
    smooth = cv2.GaussianBlur(gray, (5, 5), 1.0)
    noise  = gray - smooth

    feats.append(float(noise.std()))
    feats.append(float(np.abs(noise).mean()))
    feats.append(float(stats.kurtosis(noise.ravel())))
    feats.append(float(stats.skew(noise.ravel())))

    # ACF secondary peak of middle row
    # (periodic noise from screen pixel grid → peak at pixel-pitch lag)
    row_n = noise[S // 2, :]
    acf   = np.correlate(row_n, row_n, mode="full")
    acf   = acf[len(acf) // 2:]
    acf  /= acf[0] + 1e-12
    feats.append(float(acf[3:25].max()))

    # ── 3. Color / saturation ─────────────────────────────────────────────────
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].astype(float)
    val = hsv[:, :, 2].astype(float)

    feats += [float(sat.mean()), float(sat.std()),
              float(np.percentile(sat, 75)), float(np.percentile(sat, 95)),
              float(np.mean(sat > 120))]

    feats += [float(val.mean()), float(val.std()),
              float(np.mean(val > 200))]

    # RGB channel correlations (subpixel stripe layout alters R-G-B covariance)
    r_f = img_rgb[:, :, 0].ravel().astype(float)
    g_f = img_rgb[:, :, 1].ravel().astype(float)
    b_f = img_rgb[:, :, 2].ravel().astype(float)
    feats.append(float(np.corrcoef(r_f, g_f)[0, 1]))
    feats.append(float(np.corrcoef(r_f, b_f)[0, 1]))
    feats.append(float(np.corrcoef(g_f, b_f)[0, 1]))

    # ── 4. Edge / sharpness & orientation ────────────────────────────────────
    lap  = cv2.Laplacian(gray, cv2.CV_64F)
    feats += [float(lap.var()), float(np.abs(lap).mean()),
              float(stats.kurtosis(lap.ravel()))]

    gx   = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy   = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    feats += [float(grad.mean()), float(grad.std()),
              float(np.percentile(grad, 90))]

    # Gradient orientation histogram (8 bins × 45°)
    # Screens have mostly 0°/90° edges (UI chrome, text, bezel border).
    # Natural scenes (food grids, fabric patterns) have more diagonal edges.
    angle          = np.arctan2(gy, gx) * 180.0 / np.pi   # –180 … 180
    strong_mask    = grad > np.percentile(grad, 60)        # only strong edges
    strong_angles  = angle[strong_mask]
    ang_hist, _    = np.histogram(strong_angles,
                                  bins=np.arange(-180, 181, 45))   # 8 bins
    ang_hist       = ang_hist / (ang_hist.sum() + 1e-12)
    feats         += ang_hist.tolist()                             # 8 features
    # Entropy of orientation (low = mostly horizontal/vertical = screen-like)
    feats.append(float(-np.sum(ang_hist * np.log(ang_hist + 1e-12))))
    # Horizontal+vertical fraction vs diagonal fraction
    hv_frac  = ang_hist[0] + ang_hist[1] + ang_hist[4] + ang_hist[5]
    diag_frac= ang_hist[2] + ang_hist[3] + ang_hist[6] + ang_hist[7]
    feats.append(float(hv_frac / (diag_frac + 1e-12)))

    # ── 5. DCT block-boundary artifacts (double JPEG compression) ────────────
    dh = np.abs(np.diff(gray, axis=0))
    dv = np.abs(np.diff(gray, axis=1))

    h_bnd = dh[7::8].mean()  if len(dh)       >= 8 else dh.mean()
    h_int = np.mean([dh[i::8].mean() for i in range(min(7, len(dh)))])
    feats.append(float(h_bnd / (h_int + 1e-12)))

    v_bnd = dv[:, 7::8].mean() if dv.shape[1] >= 8 else dv.mean()
    v_int = np.mean([dv[:, i::8].mean() for i in range(min(7, dv.shape[1]))])
    feats.append(float(v_bnd / (v_int + 1e-12)))

    # ── 6. Local texture statistics ───────────────────────────────────────────
    k    = 15
    lm   = cv2.blur(gray, (k, k))
    lsq  = cv2.blur(gray ** 2, (k, k))
    lstd = np.sqrt(np.maximum(lsq - lm ** 2, 0.0))
    feats += [float(lstd.mean()), float(lstd.std()),
              float(np.percentile(lstd, 10)), float(np.percentile(lstd, 90))]

    # ── 7. LBP texture histogram ──────────────────────────────────────────────
    gray_u8 = gray.clip(0, 255).astype(np.uint8)
    lbp     = local_binary_pattern(gray_u8, P=LBP_POINTS, R=LBP_RADIUS,
                                   method="uniform")
    hist, _ = np.histogram(lbp, bins=LBP_BINS,
                           range=(0, LBP_BINS), density=True)
    feats  += hist.tolist()

    # ── 8. 1-D periodicity (row / column projection FFT) ─────────────────────
    col_fft = np.abs(np.fft.rfft(gray.mean(axis=0)))
    row_fft = np.abs(np.fft.rfft(gray.mean(axis=1)))
    feats.append(float(col_fft[2:].max() / (col_fft[2:].mean() + 1e-12)))
    feats.append(float(row_fft[2:].max() / (row_fft[2:].mean() + 1e-12)))

    # ── 9. Dark-border / bezel detection ─────────────────────────────────────
    # Phone and laptop screens always have a dark bezel surrounding the content.
    # Even when the screen is nearly dark (reflections), the very edge/frame
    # is slightly different in brightness from the interior.
    bw = S // 8   # 32-pixel border strip
    border_vals = np.concatenate([
        gray[:bw, :].ravel(),
        gray[-bw:, :].ravel(),
        gray[:, :bw].ravel(),
        gray[:, -bw:].ravel(),
    ])
    center_vals = gray[bw:S - bw, bw:S - bw].ravel()
    border_mean = border_vals.mean()
    center_mean = center_vals.mean()
    feats.append(float(center_mean - border_mean))   # positive = darker border
    feats.append(float(border_mean / (center_mean + 1e-12)))
    # Std contrast: screen border tends to have sharper brightness boundary
    feats.append(float(border_vals.std()))
    feats.append(float(center_vals.std()))

    # ── 10. Brightness bimodality (dark bezel + bright content) ──────────────
    # Screens often show a bimodal brightness distribution: very dark (bezel /
    # background) + the lit screen area.
    feats.append(float(np.mean(gray < 25)))    # very-dark fraction
    feats.append(float(np.mean(gray > 220)))   # very-bright fraction
    # Combined dark+bright fraction (screens are more extreme)
    feats.append(float(np.mean(gray < 25) + np.mean(gray > 220)))
    # Entropy of 16-bin brightness histogram
    brt_hist, _ = np.histogram(gray, bins=16, range=(0, 256))
    brt_hist     = brt_hist / (brt_hist.sum() + 1e-12)
    feats.append(float(-np.sum(brt_hist * np.log(brt_hist + 1e-12))))

    # ── 11. MobileNetV2 semantic embedding ───────────────────────────────────
    # A pretrained CNN captures high-level cues (device shape, UI chrome,
    # keyboard) that classical features miss on tricky edge cases.
    cnn_vec = _cnn_features(img_rgb)    # 1280-d float32
    feats  += cnn_vec.tolist()

    return np.array(feats, dtype=np.float32)
