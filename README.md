# Spot the Fake Photo

Full brief: **ASSIGNMENT.pdf**.  In short:

**Task** — Given one image, decide if it's a **real photo** or a **photo of a screen**
(someone re-photographing a phone/laptop instead of the real thing).

**The bar:** aim for **>95% accuracy** on held-out photos.

🔴 **Live demo:** https://huggingface.co/spaces/Devansh1009/spot-the-fake-photo

---

## Quick start

```bash
# 1. Install dependencies (once)
pip install -r requirements.txt

# 2. Train (creates model_nn.pt and model.pkl)
python train_nn.py      # fine-tuned EfficientNet-B0  (~17 min, CPU)
python train.py         # classical + CNN feature ensemble  (~1 min)

# 3. Predict
python predict.py path/to/image.jpg
# Prints a number 0–1:  0 = real photo,  1 = photo of a screen
```

A live camera demo of this model is also deployed at the link above
(source lives in a separate Hugging Face Space repo, not needed here).

---

## Files

| File | Purpose |
|---|---|
| `predict.py` | **Main script** — run this on any image |
| `features.py` | Hand-crafted + MobileNetV2 feature extraction |
| `train.py` | Train the RF+SVM ensemble → `model.pkl` |
| `train_nn.py` | Fine-tune EfficientNet-B0 → `model_nn.pt` |
| `model_nn.pt` | Saved fine-tuned NN (primary; ~20 MB) |
| `model.pkl` | Saved RF+SVM ensemble (fallback; ~50 MB) |
| `note.md` | **Full solution write-up** (approach, accuracy, latency, cost) |
| `Data/real/`, `Data/screen/` | Training photos (52 real, 63 screen) |

---

## How `predict.py` chooses a model

1. If `model_nn.pt` exists → use fine-tuned **EfficientNet-B0** (~51 ms/image).
2. Otherwise → use the **RF+SVM ensemble** from `model.pkl` (~142 ms/image).

---

## Accuracy

| Model | 3-fold CV | Training |
|---|---|---|
| EfficientNet-B0 fine-tuned | **93.9 %** ± 1.3 % | 100 % |
| RF + SVM ensemble | 93.9 % ± 4.4 % | 100 % |

**Measured accuracy is 93.9%, not the 95%+ target.** See `note.md` for an
honest discussion of the residual CV errors (all ambiguous edge cases, not
clear failures) and why I'd expect — but have not measured — improvement
with more and more-varied training data.

---

## Latency & Cost

| Deployment | Latency | Cost |
|---|---|---|
| **On-device** (phone) | ~120–180 ms | **$0** |
| Cloud CPU (t3.medium) | ~51 ms | ~$0.50 / 1 000 images |
| Cloud GPU batch | ~5–10 ms | ~$0.04 / 1 000 images |

~1 day. Use whatever tools you like.
