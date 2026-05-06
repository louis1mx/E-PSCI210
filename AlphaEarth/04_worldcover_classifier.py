"""
Chengdu AlphaEarth — Script 04
Purpose : Train a lightweight Random Forest classifier on AlphaEarth 2021
          embeddings using ESA WorldCover 2021 as ground-truth land-cover
          labels. Apply the trained classifier to 2025 embeddings to infer
          whether each grid cell became more vegetation-like (greening) or
          more built-up-like (paving) between 2021 and 2025.

Workflow:
  1. Load AlphaEarth embeddings: 2021 (pre-policy) and 2025 (post-policy).
  2. Load and reproject ESA WorldCover 2021 onto the same 64x64 grid.
  3. Map WorldCover classes → 3 labels: vegetation / built / other.
  4. Train Random Forest on 2021 embeddings + WorldCover labels.
     Report cross-validation accuracy and feature importance.
  5. Apply classifier to both years → per-cell class probabilities.
  6. Compute Δgreening_prob = P(vegetation|2025) − P(vegetation|2021).
     Positive → embedding moved toward vegetation (greening signal).
     Negative → embedding moved away from vegetation (paving/clearing).
  7. Save maps, metrics, and a summary figure.

Inputs (must be in AlphaEarth/data/raw/):
  alphaearth_chengdu_2021.tif
  alphaearth_chengdu_2025.tif
  worldcover_chengdu_2021.tif   ← produced by 03_export_worldcover_chengdu.py

Outputs (AlphaEarth/outputs/):
  figures/chengdu_worldcover_classifier_maps.png
  figures/chengdu_greening_probability_change.png
  tables/chengdu_worldcover_classifier_metrics.csv
  tables/chengdu_worldcover_class_transition.csv
  arrays/greening_prob_2021.npy
  arrays/greening_prob_2025.npy
  arrays/delta_greening_prob.npy
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject as warp_reproject
from rasterio.warp import Resampling as WarpResampling
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


# ── Paths ────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent
EMBED_DIR    = ROOT / "data" / "raw"
OUT_FIG_DIR  = ROOT / "outputs" / "figures"
OUT_TAB_DIR  = ROOT / "outputs" / "tables"
OUT_ARR_DIR  = ROOT / "outputs" / "arrays"
for d in (OUT_FIG_DIR, OUT_TAB_DIR, OUT_ARR_DIR):
    d.mkdir(parents=True, exist_ok=True)

BASELINE_YEAR = 2021
COMPARE_YEAR  = 2025

# ── WorldCover class mapping ─────────────────────────────────────────────────
# ESA WorldCover v200 class values
WC_CLASSES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}

# Collapse to 3 interpretable labels
def wc_to_label(val: int) -> str:
    if val in (10, 20, 30, 90, 95, 100):
        return "vegetation"
    if val == 50:
        return "built"
    return "other"   # cropland, bare, water, ice


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# ── I/O helpers ──────────────────────────────────────────────────────────────

def read_multiband(path: Path, out_shape: tuple[int, int]) -> tuple[np.ndarray, object, object]:
    with rasterio.open(path) as src:
        arr = src.read(
            out_shape=(src.count, out_shape[0], out_shape[1]),
            resampling=Resampling.bilinear,
        ).astype(np.float32)
        transform = src.transform * src.transform.scale(
            src.width / out_shape[1], src.height / out_shape[0]
        )
        nodata = src.nodata
        crs = src.crs
    if nodata is not None:
        arr[arr == nodata] = np.nan
    return arr, transform, crs


def read_worldcover(wc_path: Path, target_shape: tuple[int, int],
                    target_transform, target_crs) -> np.ndarray:
    """
    Read WorldCover GeoTIFF and reproject/resample to match the AlphaEarth
    64x64 analysis grid.  Uses nearest-neighbour resampling to preserve
    categorical class values.
    """
    with rasterio.open(wc_path) as src:
        src_arr  = src.read(1).astype(np.float32)
        src_transform = src.transform
        src_crs  = src.crs
        nodata   = src.nodata if src.nodata is not None else 0.0

    dst = np.zeros(target_shape, dtype=np.float32)
    warp_reproject(
        source=src_arr,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=target_transform,
        dst_crs=target_crs,
        resampling=WarpResampling.nearest,
        src_nodata=nodata,
        dst_nodata=0.0,
    )
    dst[dst == 0.0] = np.nan
    return dst


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Chengdu WorldCover + AlphaEarth classifier")
    print("=" * 60)

    # ── 1. Check inputs ───────────────────────────────────────────────────
    pre_path  = EMBED_DIR / f"alphaearth_chengdu_{BASELINE_YEAR}.tif"
    post_path = EMBED_DIR / f"alphaearth_chengdu_{COMPARE_YEAR}.tif"
    wc_path   = EMBED_DIR / "worldcover_chengdu_2021.tif"

    for p in (pre_path, post_path, wc_path):
        if not p.exists():
            raise FileNotFoundError(
                f"\nMissing required input: {p}\n"
                + ("  Run 03_export_worldcover_chengdu.py first and download "
                   "the GeoTIFF." if "worldcover" in p.name else
                   f"  Download AlphaEarth {p.stem.split('_')[-1]} embedding from GEE.")
            )

    # ── 2. Load embeddings ────────────────────────────────────────────────
    print(f"\nLoading AlphaEarth embeddings …")
    pre_emb,  pre_transform,  pre_crs  = read_multiband(pre_path,  (64, 64))
    post_emb, post_transform, post_crs = read_multiband(post_path, (64, 64))
    H, W = pre_emb.shape[1], pre_emb.shape[2]
    n_bands = pre_emb.shape[0]
    print(f"  Shape : {n_bands} bands × {H}×{W}")

    # ── 3. Load WorldCover ────────────────────────────────────────────────
    print("Loading ESA WorldCover 2021 …")
    wc_grid = read_worldcover(wc_path, (H, W), pre_transform, pre_crs)
    print(f"  Unique classes in grid: {np.unique(wc_grid[np.isfinite(wc_grid)]).astype(int).tolist()}")

    # ── 4. Build training data ────────────────────────────────────────────
    # Flatten: (H*W, n_bands) features, (H*W,) labels
    pre_flat  = pre_emb.reshape(n_bands, -1).T    # (N, 64)
    wc_flat   = wc_grid.ravel()                   # (N,)

    valid_mask = (
        np.isfinite(pre_flat).all(axis=1) &
        np.isfinite(wc_flat) &
        (wc_flat > 0)
    )
    X_train = pre_flat[valid_mask]
    y_raw   = wc_flat[valid_mask].astype(int)
    y_train = np.array([wc_to_label(v) for v in y_raw])

    print(f"\nTraining set: {X_train.shape[0]} valid cells")
    for lbl in ("vegetation", "built", "other"):
        n = (y_train == lbl).sum()
        print(f"  {lbl:12s}: {n} cells ({100*n/len(y_train):.1f}%)")

    if X_train.shape[0] < 10:
        raise RuntimeError(
            "Too few valid training cells. Check that the WorldCover GeoTIFF "
            "covers the Chengdu analysis region."
        )

    # ── 5. Train Random Forest ────────────────────────────────────────────
    print("\nTraining Random Forest classifier …")
    le = LabelEncoder()
    y_enc = le.fit_transform(y_train)

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf, X_train, y_enc, cv=cv, scoring="accuracy")
    print(f"  5-fold CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    rf.fit(X_train, y_enc)

    # Classification report on full training set (in-sample, for reference)
    y_pred_train = rf.predict(X_train)
    print("\n  In-sample classification report:")
    print(classification_report(y_enc, y_pred_train,
                                target_names=le.classes_,
                                digits=3))

    # Feature importance (top 10 embedding dimensions)
    importances = rf.feature_importances_
    top10_idx = np.argsort(importances)[::-1][:10]
    print("  Top-10 most informative embedding dimensions:")
    for rank, idx in enumerate(top10_idx, 1):
        print(f"    {rank:2d}. Band {idx:2d}  importance = {importances[idx]:.4f}")

    # ── 6. Apply to both years ────────────────────────────────────────────
    print("\nApplying classifier to both years …")

    def predict_proba_map(emb: np.ndarray) -> dict[str, np.ndarray]:
        """Return per-class probability maps (H, W) for each class."""
        flat = emb.reshape(n_bands, -1).T          # (N, n_bands)
        valid = np.isfinite(flat).all(axis=1)

        proba_flat = np.full((flat.shape[0], len(le.classes_)), np.nan)
        proba_flat[valid] = rf.predict_proba(flat[valid])

        maps = {}
        for i, cls in enumerate(le.classes_):
            maps[cls] = proba_flat[:, i].reshape(H, W)
        return maps

    proba_2021 = predict_proba_map(pre_emb)
    proba_2025 = predict_proba_map(post_emb)

    veg_prob_2021 = proba_2021["vegetation"]
    veg_prob_2025 = proba_2025["vegetation"]
    delta_veg     = veg_prob_2025 - veg_prob_2021   # positive = greening

    # ── 7. Transition table ───────────────────────────────────────────────
    pred_2021 = np.full((H, W), "NA", dtype=object)
    pred_2025 = np.full((H, W), "NA", dtype=object)

    post_flat = post_emb.reshape(n_bands, -1).T
    valid_both = (
        np.isfinite(pre_flat).all(axis=1) &
        np.isfinite(post_flat).all(axis=1)
    )

    pred_2021_enc = rf.predict(pre_flat[valid_both])
    pred_2025_enc = rf.predict(post_flat[valid_both])
    pred_2021.ravel()[valid_both] = le.inverse_transform(pred_2021_enc)
    pred_2025.ravel()[valid_both] = le.inverse_transform(pred_2025_enc)

    transitions = pd.crosstab(
        pd.Series(pred_2021.ravel(), name="2021"),
        pd.Series(pred_2025.ravel(), name="2025"),
    ).drop(index="NA", errors="ignore").drop(columns="NA", errors="ignore")

    print("\n  Land-cover transition table (cell counts):")
    print(transitions.to_string())

    # ── 8. Summarise delta_veg ────────────────────────────────────────────
    valid_dv = np.isfinite(delta_veg)
    n_greening = (delta_veg[valid_dv] > 0.05).sum()
    n_paving   = (delta_veg[valid_dv] < -0.05).sum()
    n_stable   = valid_dv.sum() - n_greening - n_paving
    mean_dv    = float(np.nanmean(delta_veg))

    print(f"\n  Δgreening_prob summary (threshold ±0.05):")
    print(f"    Mean Δgreening_prob      : {mean_dv:+.4f}")
    print(f"    Greening cells (>+0.05)  : {n_greening} ({100*n_greening/valid_dv.sum():.1f}%)")
    print(f"    Paving   cells (<-0.05)  : {n_paving}   ({100*n_paving/valid_dv.sum():.1f}%)")
    print(f"    Stable   cells (|Δ|≤0.05): {n_stable}  ({100*n_stable/valid_dv.sum():.1f}%)")

    # ── 9. Save arrays ────────────────────────────────────────────────────
    np.save(OUT_ARR_DIR / "greening_prob_2021.npy", veg_prob_2021)
    np.save(OUT_ARR_DIR / "greening_prob_2025.npy", veg_prob_2025)
    np.save(OUT_ARR_DIR / "delta_greening_prob.npy", delta_veg)

    # ── 10. Save tables ───────────────────────────────────────────────────
    metrics_df = pd.DataFrame({
        "metric": [
            "cv_accuracy_mean", "cv_accuracy_std",
            "mean_delta_greening_prob",
            "n_greening_cells", "n_paving_cells", "n_stable_cells",
            "pct_greening", "pct_paving",
        ],
        "value": [
            round(cv_scores.mean(), 4), round(cv_scores.std(), 4),
            round(mean_dv, 4),
            int(n_greening), int(n_paving), int(n_stable),
            round(100 * n_greening / valid_dv.sum(), 2),
            round(100 * n_paving   / valid_dv.sum(), 2),
        ],
    })
    metrics_df.to_csv(OUT_TAB_DIR / "chengdu_worldcover_classifier_metrics.csv", index=False)
    transitions.to_csv(OUT_TAB_DIR / "chengdu_worldcover_class_transition.csv")

    # ── 11. Figures ───────────────────────────────────────────────────────
    # Figure A: vegetation probability maps 2021 / 2025 / delta
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    im0 = axes[0].imshow(veg_prob_2021, cmap="YlGn", vmin=0, vmax=1)
    axes[0].set_title(f"(a) P(vegetation) — {BASELINE_YEAR}")
    plt.colorbar(im0, ax=axes[0], shrink=0.82)

    im1 = axes[1].imshow(veg_prob_2025, cmap="YlGn", vmin=0, vmax=1)
    axes[1].set_title(f"(b) P(vegetation) — {COMPARE_YEAR}")
    plt.colorbar(im1, ax=axes[1], shrink=0.82)

    lim = max(abs(np.nanmin(delta_veg)), abs(np.nanmax(delta_veg)))
    im2 = axes[2].imshow(delta_veg, cmap="RdYlGn", vmin=-lim, vmax=lim)
    axes[2].set_title(f"(c) Δgreening prob\n({BASELINE_YEAR}→{COMPARE_YEAR})")
    plt.colorbar(im2, ax=axes[2], shrink=0.82,
                 label="+ = greening   − = paving")

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(
        f"WorldCover-supervised greening probability  ·  "
        f"CV accuracy = {cv_scores.mean():.3f}",
        fontsize=11,
    )
    fig.tight_layout()
    path_a = OUT_FIG_DIR / "chengdu_worldcover_classifier_maps.png"
    fig.savefig(path_a, dpi=300)
    plt.close(fig)
    print(f"\n  Saved: {path_a}")

    # Figure B: Δgreening_prob vs AlphaEarth cosine change (if available)
    cosine_path = OUT_ARR_DIR.parent.parent.parent / \
        "outputs" / "arrays" / "convlstm_residual.npy"
    # Also try relative to project root
    project_cosine = OUT_ARR_DIR / "cosine_change_2021_2025.npy"

    if project_cosine.exists():
        cosine_change = np.load(project_cosine)
        valid_both_map = np.isfinite(delta_veg) & np.isfinite(cosine_change)
        x = cosine_change[valid_both_map]
        y = delta_veg[valid_both_map]

        fig2, ax2 = plt.subplots(figsize=(6, 5))
        sc = ax2.scatter(x, y, s=8, alpha=0.4, c=y,
                         cmap="RdYlGn", vmin=-0.3, vmax=0.3,
                         edgecolors="none", rasterized=True)
        ax2.axhline(0, color="#555", lw=0.8, ls=":")
        ax2.axhline( 0.05, color="#2CA02C", lw=0.8, ls="--", alpha=0.6)
        ax2.axhline(-0.05, color="#D62728", lw=0.8, ls="--", alpha=0.6)
        plt.colorbar(sc, ax=ax2, label="Δgreening prob (+ = greening)")
        r = float(np.corrcoef(x, y)[0, 1])
        ax2.set_xlabel(f"AlphaEarth cosine change ({BASELINE_YEAR}→{COMPARE_YEAR})")
        ax2.set_ylabel("Δgreening probability")
        ax2.set_title(
            f"AlphaEarth change vs WorldCover-inferred greening\n"
            f"r = {r:.3f}"
        )
        fig2.tight_layout()
        path_b = OUT_FIG_DIR / "chengdu_greening_probability_change.png"
        fig2.savefig(path_b, dpi=300)
        plt.close(fig2)
        print(f"  Saved: {path_b}")
        print(f"  r(cosine_change, Δgreening_prob) = {r:.3f}")

    print("\n✅  Script 04 complete.")
    print(f"   Figures : {OUT_FIG_DIR}")
    print(f"   Tables  : {OUT_TAB_DIR}")
    print(f"   Arrays  : {OUT_ARR_DIR}")


if __name__ == "__main__":
    main()
