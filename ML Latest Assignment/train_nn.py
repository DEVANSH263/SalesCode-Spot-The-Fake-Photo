"""
Fine-tune EfficientNet-B0 for screen-vs-real detection.

Strategy
--------
- Start from ImageNet-pretrained EfficientNet-B0 (5.3 M params, 14 MB weights).
- Freeze early feature blocks; only fine-tune the last 3 MBConv blocks + the
  binary classifier head (~1.5 M trainable params).
- Heavy data augmentation compensates for the small dataset (115 images).
- 3-fold cross-validation gives an honest accuracy estimate.
- Final model is trained on ALL data and saved as model_nn.pt.

Usage
-----
    python train_nn.py
    python train_nn.py --data ../../Data --out model_nn.pt --epochs 30

Output
------
    model_nn.pt   – full EfficientNet-B0 state dict (binary classifier head)
                    loaded by predict.py
"""

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights

warnings.filterwarnings("ignore")

EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# ── Transforms ────────────────────────────────────────────────────────────────

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize(260),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.2),
    # Include 90°/180°/270° rotations so the model isn't fooled by rotated screens
    transforms.RandomApply([transforms.RandomRotation((85, 95))],  p=0.15),  # ≈90°
    transforms.RandomApply([transforms.RandomRotation((175, 185))], p=0.15),  # ≈180°
    transforms.RandomRotation(20),   # small continuous rotation
    transforms.ColorJitter(brightness=0.3, contrast=0.3,
                           saturation=0.3, hue=0.08),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Dataset ───────────────────────────────────────────────────────────────────

class ScreenDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths     = list(paths)
        self.labels    = list(labels)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), torch.tensor(self.labels[idx], dtype=torch.float32)


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model():
    """EfficientNet-B0 with a binary logit head; last 3 blocks + head are trainable."""
    net = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Freeze everything
    for param in net.parameters():
        param.requires_grad = False

    # Unfreeze last 3 MBConv blocks (indices 6, 7, 8 inside features)
    for i in [6, 7, 8]:
        for param in net.features[i].parameters():
            param.requires_grad = True

    # Replace the classifier with a binary head
    net.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(1280, 1),          # raw logit; apply BCEWithLogitsLoss
    )
    for param in net.classifier.parameters():
        param.requires_grad = True

    return net


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_fold(train_paths, train_labels, val_paths, val_labels,
                   epochs: int = 30, device: str = "cpu"):
    """
    Fine-tune on one fold.  Returns (model, best_val_accuracy).
    """
    train_ds = ScreenDataset(train_paths, train_labels, TRAIN_TRANSFORM)
    val_ds   = ScreenDataset(val_paths,   val_labels,   VAL_TRANSFORM)

    # Oversample minority class in training
    n0 = sum(1 for l in train_labels if l == 0)
    n1 = sum(1 for l in train_labels if l == 1)
    w  = [1.0 / n0 if l == 0 else 1.0 / n1 for l in train_labels]
    sampler = torch.utils.data.WeightedRandomSampler(w, len(w), replacement=True)

    train_dl = DataLoader(train_ds, batch_size=8,  sampler=sampler,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=0)

    model     = build_model().to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4, weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3,
        steps_per_epoch=len(train_dl), epochs=epochs,
    )
    # Weighted BCE loss
    pos_weight = torch.tensor([n0 / (n1 + 1e-6)]).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_acc   = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        # ── train ──
        model.train()
        for imgs, labels in train_dl:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs).squeeze(1)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

        # ── validate ──
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = (torch.sigmoid(model(imgs).squeeze(1)) > 0.5).long()
                correct += (preds == labels.long()).sum().item()
                total   += len(labels)
        acc = correct / total

        if acc > best_acc:
            best_acc   = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        print(f"    epoch {epoch:02d}/{epochs}  val_acc={acc:.4f}  best={best_acc:.4f}")

    model.load_state_dict(best_state)
    return model, best_acc


# ── Data loading ──────────────────────────────────────────────────────────────

def load_paths_labels(data_dir: Path):
    paths, labels = [], []
    for label, folder in [(0, "real"), (1, "screen")]:
        fpath = data_dir / folder
        for f in sorted(fpath.iterdir()):
            if f.suffix.lower() in EXTS:
                paths.append(str(f))
                labels.append(label)
    return paths, labels


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default=str(Path(__file__).parent.parent / "Data"))
    parser.add_argument("--out",    default=str(Path(__file__).parent / "model_nn.pt"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--folds",  type=int, default=3,
                        help="CV folds (3 recommended for small datasets)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    paths, labels = load_paths_labels(Path(args.data))
    labels_arr = np.array(labels)
    print(f"Dataset: {len(paths)} images  (real={int((labels_arr==0).sum())}, "
          f"screen={int((labels_arr==1).sum())})")

    # ── Cross-validation ──────────────────────────────────────────────────────
    cv      = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    cv_accs = []
    print(f"\n--- {args.folds}-fold cross-validation ({args.epochs} epochs each) ---")

    for fold_idx, (tr_idx, val_idx) in enumerate(cv.split(paths, labels), 1):
        tr_paths   = [paths[i]  for i in tr_idx]
        val_paths  = [paths[i]  for i in val_idx]
        tr_labels  = [labels[i] for i in tr_idx]
        val_labels = [labels[i] for i in val_idx]

        print(f"\n  Fold {fold_idx}/{args.folds}  "
              f"(train={len(tr_paths)}, val={len(val_paths)})")
        t0 = time.perf_counter()
        _, acc = train_one_fold(tr_paths, tr_labels, val_paths, val_labels,
                                epochs=args.epochs, device=device)
        elapsed = (time.perf_counter() - t0) / 60
        print(f"  Fold {fold_idx} done  best_val_acc={acc:.4f}  ({elapsed:.1f} min)")
        cv_accs.append(acc)

    print(f"\nCV accuracy: {np.mean(cv_accs):.4f} +/- {np.std(cv_accs):.4f}  "
          f"per-fold={np.round(cv_accs, 3).tolist()}")

    # ── Final training on ALL data ────────────────────────────────────────────
    print(f"\n--- Final model: train on ALL {len(paths)} images ({args.epochs} epochs) ---")
    t0    = time.perf_counter()
    model, _ = train_one_fold(paths, labels, paths, labels,
                               epochs=args.epochs, device=device)
    elapsed   = (time.perf_counter() - t0) / 60

    # Save state dict + architecture tag
    out_path = Path(args.out)
    torch.save({"state_dict": model.state_dict(), "arch": "efficientnet_b0"},
               str(out_path))
    print(f"Model saved -> {out_path}  ({elapsed:.1f} min)")
    print(f"\nFinal CV accuracy: {np.mean(cv_accs):.4f} ({np.mean(cv_accs)*100:.1f}%)")


if __name__ == "__main__":
    main()
