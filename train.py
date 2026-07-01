"""
Train the screen-vs-real photo detector.

Usage
-----
    python train.py                          # uses default paths
    python train.py --data Data --out model.pkl

Outputs
-------
    model.pkl  – pickled sklearn Pipeline (StandardScaler → PCA → SVC + RF ensemble)
                 loaded at prediction time by predict.py
"""

import argparse
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from features import extract_features

warnings.filterwarnings("ignore")

EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# Number of PCA components for the full feature vector.
# Must be < min(n_samples_in_fold) across all CV folds.
# For 5-fold CV on 115 samples, each training fold has ~92 samples → cap at 80.
PCA_COMPONENTS = 60


# ── Data loading ──────────────────────────────────────────────────────────────

def load_dataset(data_dir: Path):
    """
    Walk data_dir/real/ (label=0) and data_dir/screen/ (label=1),
    extract features from every image, and return (X, y).
    """
    X, y = [], []
    for label, folder in [(0, "real"), (1, "screen")]:
        fpath = data_dir / folder
        files = sorted(f for f in fpath.iterdir() if f.suffix.lower() in EXTS)
        print(f"\n  {folder}/  ({len(files)} images)")
        for f in files:
            t0 = time.perf_counter()
            try:
                feat = extract_features(str(f))
                X.append(feat)
                y.append(label)
                ms = (time.perf_counter() - t0) * 1000
                print(f"    OK  {f.name}  ({ms:.0f} ms)")
            except Exception as exc:
                print(f"    ERR {f.name}: {exc}")
    return np.array(X, dtype=np.float32), np.array(y, dtype=int)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train screen-vs-real detector")
    parser.add_argument("--data", default=str(Path(__file__).parent / "Data"),
                        help="Folder containing real/ and screen/ subfolders")
    parser.add_argument("--out",  default=str(Path(__file__).parent / "model.pkl"),
                        help="Output path for the trained model")
    args = parser.parse_args()

    data_dir = Path(args.data)
    print(f"Loading dataset from {data_dir} ...")
    X, y = load_dataset(data_dir)

    print(f"\n{'='*60}")
    print(f"Dataset :  {len(y)} images,  {X.shape[1]} features")
    print(f"  real   (0): {int((y == 0).sum())}")
    print(f"  screen (1): {int((y == 1).sum())}")
    print(f"{'='*60}")

    # Sanitise NaN / Inf
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    # ── Build models ──────────────────────────────────────────────────────────
    # Cap PCA components at min(samples-1, desired)
    n_pca = min(PCA_COMPONENTS, len(y) - 1)

    # Pipeline A: RandomForest on raw features.
    # With 1355 features, set max_features=200 so each split sees enough CNN features.
    rf = RandomForestClassifier(
        n_estimators=400, min_samples_leaf=1,
        max_features=200,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )

    # Pipeline B: StandardScaler → PCA → calibrated SVC
    svm_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=n_pca, random_state=42)),
        ("svc",    SVC(kernel="rbf", C=10, gamma="scale",
                       class_weight="balanced", probability=True)),
    ])

    # Soft-voting ensemble (equal weights)
    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("svm", svm_pipe)],
        voting="soft",
    )

    # ── Cross-validation ──────────────────────────────────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("\n--- 5-fold stratified cross-validation ---")
    for name, clf in [("RandomForest", rf),
                      ("SVM+PCA", svm_pipe),
                      ("Ensemble", ensemble)]:
        s = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
        print(f"  {name:15s}: {s.mean():.4f} +/- {s.std():.4f}  "
              f"per-fold = {np.round(s, 3).tolist()}")

    # ── Final model trained on ALL data ───────────────────────────────────────
    print("\n--- Training final ensemble on all data ---")
    ensemble.fit(X, y)

    train_preds = ensemble.predict(X)
    train_acc   = float((train_preds == y).mean())
    print(f"  Training accuracy: {train_acc:.4f}")
    print("\nClassification report (train):")
    print(classification_report(y, train_preds, target_names=["real", "screen"]))

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = Path(args.out)
    with open(out_path, "wb") as fh:
        pickle.dump(ensemble, fh)
    print(f"Model saved -> {out_path}")


if __name__ == "__main__":
    main()
