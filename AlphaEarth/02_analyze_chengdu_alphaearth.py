"""
Chengdu AlphaEarth — Script 02
Purpose : Compute Chengdu-only AlphaEarth change detection and align it with
          the existing ConvLSTM cooling residual.

Core idea from the draft paper:
  - Compare a pre-policy AlphaEarth embedding image (default: 2021) with a
    post-policy image (default: latest available year).
  - Measure per-cell embedding change with cosine distance and Euclidean
    distance.
  - Rebuild the ConvLSTM post-policy cooling residual from the saved model and
    Landsat tiles.
  - Identify where high embedding change and high cooling coincide.

Expected local inputs:
  AlphaEarth/data/raw/alphaearth_chengdu_2021.tif
  AlphaEarth/data/raw/alphaearth_chengdu_2024.tif  (or another post-policy year)
  data/raw/gee/chengdu_lst_tiles/LST_*.tif
  outputs/models/convlstm_model.keras

Outputs:
  AlphaEarth/outputs/figures/chengdu_alphaearth_alignment_2021_2024.png
  AlphaEarth/outputs/tables/chengdu_alphaearth_summary_2021_2024.csv
  AlphaEarth/outputs/tables/chengdu_alphaearth_hotspots_2021_2024.csv
  AlphaEarth/outputs/arrays/*.npy
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import xy as transform_xy
from sklearn.decomposition import PCA
import tensorflow as tf


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent

EMBED_DIR_DEFAULT = ROOT / "data" / "raw"
OUT_FIG_DIR = ROOT / "outputs" / "figures"
OUT_TAB_DIR = ROOT / "outputs" / "tables"
OUT_ARR_DIR = ROOT / "outputs" / "arrays"
for directory in (OUT_FIG_DIR, OUT_TAB_DIR, OUT_ARR_DIR):
    directory.mkdir(parents=True, exist_ok=True)

MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "convlstm_model.keras"
TILE_DIR = PROJECT_ROOT / "data" / "raw" / "gee" / "chengdu_lst_tiles"

SEQ_LEN = 12
POLICY_YM = "2022-01"
YEAR_RE = re.compile(r"(20\d{2})")

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Chengdu AlphaEarth change detection and cooling alignment."
    )
    parser.add_argument("--embedding-dir", type=Path, default=EMBED_DIR_DEFAULT)
    parser.add_argument("--baseline-year", type=int, default=2021)
    parser.add_argument(
        "--compare-year",
        default="latest",
        help="Post-policy year to compare against. Use 'latest' or an explicit year.",
    )
    parser.add_argument(
        "--hotspot-quantile",
        type=float,
        default=0.75,
        help="Quantile threshold for high-change/high-cooling hotspot logic.",
    )
    return parser.parse_args()


def load_model(model_path: Path) -> tf.keras.Model:
    if not model_path.exists():
        raise FileNotFoundError(
            f"ConvLSTM model not found: {model_path}\n"
            "Run Version2/scripts/06_convlstm.py first."
        )
    return tf.keras.models.load_model(model_path)


def model_grid_shape(model: tf.keras.Model) -> tuple[int, int]:
    _, _, height, width, _ = model.input_shape
    return int(height), int(width)


def discover_embedding_files(embedding_dir: Path) -> dict[int, Path]:
    if not embedding_dir.exists():
        raise FileNotFoundError(f"Embedding directory not found: {embedding_dir}")

    catalog: dict[int, Path] = {}
    for path in sorted(embedding_dir.glob("alphaearth_chengdu_*.tif")):
        match = YEAR_RE.search(path.stem)
        if match:
            catalog[int(match.group(1))] = path

    if not catalog:
        raise FileNotFoundError(
            f"No embedding GeoTIFFs found in {embedding_dir}\n"
            "Expected files such as alphaearth_chengdu_2021.tif."
        )
    return catalog


def discover_singleband_year_files(directory: Path, prefix: str) -> dict[int, Path]:
    catalog: dict[int, Path] = {}
    if not directory.exists():
        return catalog

    for path in sorted(directory.glob(f"{prefix}*.tif")):
        match = YEAR_RE.search(path.stem)
        if match:
            catalog[int(match.group(1))] = path
    return catalog


def resolve_compare_year(requested: str, catalog: dict[int, Path]) -> int:
    if requested == "latest":
        return max(catalog)
    return int(requested)


def resampled_transform(src: rasterio.io.DatasetReader, out_shape: tuple[int, int]):
    out_h, out_w = out_shape
    return src.transform * src.transform.scale(src.width / out_w, src.height / out_h)


def read_multiband(path: Path, out_shape: tuple[int, int]) -> tuple[np.ndarray, object, object]:
    with rasterio.open(path) as src:
        arr = src.read(
            out_shape=(src.count, out_shape[0], out_shape[1]),
            resampling=Resampling.bilinear,
        ).astype(np.float32)
        transform = resampled_transform(src, out_shape)
        crs = src.crs
        nodata = src.nodata

    if nodata is not None:
        arr[arr == nodata] = np.nan
    return arr, transform, crs


def read_singleband(path: Path, out_shape: tuple[int, int]) -> tuple[np.ndarray, object]:
    with rasterio.open(path) as src:
        arr = src.read(
            1,
            out_shape=out_shape,
            resampling=Resampling.bilinear,
        ).astype(np.float32)
        transform = resampled_transform(src, out_shape)
        nodata = src.nodata

    if nodata is not None:
        arr[arr == nodata] = np.nan
    return arr, transform


def load_tiles(target_shape: tuple[int, int]) -> tuple[np.ndarray, list[str]]:
    tile_files = sorted(TILE_DIR.glob("LST_*.tif"))
    if not tile_files:
        raise FileNotFoundError(
            f"No Landsat tiles found in {TILE_DIR}\n"
            "Make sure Script 01 exported the monthly GeoTIFF stack."
        )

    images: list[np.ndarray] = []
    yms: list[str] = []

    for path in tile_files:
        parts = path.stem.split("_")
        ym = f"{parts[1]}-{parts[2]}"
        with rasterio.open(path) as src:
            arr = src.read(
                1,
                out_shape=target_shape,
                resampling=Resampling.bilinear,
            ).astype(np.float32)
            nodata = src.nodata

        if nodata is not None:
            arr[arr == nodata] = np.nan
        arr[arr == 0.0] = np.nan
        images.append(arr)
        yms.append(ym)

    stack = np.stack(images)

    for i in range(stack.shape[1]):
        for j in range(stack.shape[2]):
            ts = stack[:, i, j]
            mask = np.isnan(ts)
            if mask.any() and not mask.all():
                idx = np.arange(len(ts))
                stack[:, i, j] = np.interp(idx, idx[~mask], ts[~mask])

    global_mean = np.nanmean(stack)
    stack = np.where(np.isnan(stack), global_mean, stack)
    return stack, yms


def annual_lst_mean(year: int, target_shape: tuple[int, int]) -> np.ndarray:
    images, yms = load_tiles(target_shape)
    mask = np.array([ym.startswith(f"{year}-") for ym in yms], dtype=bool)
    if not mask.any():
        raise FileNotFoundError(f"No monthly LST tiles found for year {year}.")
    return np.nanmean(images[mask], axis=0)


def normalise(images: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.nanmean(images, axis=0, keepdims=True)
    sig = np.nanstd(images, axis=0, keepdims=True) + 1e-6
    return (images - mu) / sig, mu, sig


def make_sequences(images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(images) - SEQ_LEN):
        X.append(images[i : i + SEQ_LEN, :, :, np.newaxis])
        y.append(images[i + SEQ_LEN, :, :, np.newaxis])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


RESIDUAL_NPY = PROJECT_ROOT / "outputs" / "arrays" / "convlstm_residual.npy"


def compute_cooling_residual(
    model: tf.keras.Model,
) -> tuple[np.ndarray, list[str], object, object]:
    """Return residual on the Landsat tile grid, plus that grid's transform and CRS.

    Loads the pre-saved residual from Script 06 when available so that the
    values shown here are bit-for-bit identical to Figure 8 panel (c).
    """
    target_shape = model_grid_shape(model)
    tile_files = sorted(TILE_DIR.glob("LST_*.tif"))
    with rasterio.open(tile_files[0]) as src:
        lst_transform = resampled_transform(src, target_shape)
        lst_crs = src.crs

    model_mtime = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else None
    residual_mtime = RESIDUAL_NPY.stat().st_mtime if RESIDUAL_NPY.exists() else None

    if RESIDUAL_NPY.exists() and (
        model_mtime is None or residual_mtime is None or residual_mtime >= model_mtime
    ):
        print(f"  Loading saved residual from {RESIDUAL_NPY}")
        residual = np.load(RESIDUAL_NPY).astype(np.float32)
        _, yms = load_tiles(target_shape)
        target_yms = [yms[i + SEQ_LEN] for i in range(len(yms) - SEQ_LEN)]
        yms_test = [ym for ym in target_yms if ym >= POLICY_YM]
        return residual, yms_test, lst_transform, lst_crs

    if RESIDUAL_NPY.exists() and model_mtime is not None and residual_mtime is not None:
        print(
            "  Saved residual is older than the current ConvLSTM model – "
            "recomputing to match Figure 8."
        )

    # Fallback: recompute from scratch (results may differ slightly from Fig 8)
    else:
        print("  WARNING: convlstm_residual.npy not found – recomputing residual.")
    images, yms = load_tiles(target_shape)
    images_norm, mu, sig = normalise(images)
    X, y = make_sequences(images_norm)

    target_yms = [yms[i + SEQ_LEN] for i in range(len(X))]
    test_mask = np.array([ym >= POLICY_YM for ym in target_yms], dtype=bool)
    if not test_mask.any():
        raise RuntimeError("No post-policy months found for ConvLSTM residual analysis.")

    y_pred_norm = model.predict(X[test_mask], verbose=0)
    y_pred = y_pred_norm * sig[..., np.newaxis] + mu[..., np.newaxis]
    y_true = y[test_mask] * sig[..., np.newaxis] + mu[..., np.newaxis]
    residual = y_pred.mean(axis=0).squeeze() - y_true.mean(axis=0).squeeze()
    yms_test = [ym for ym, keep in zip(target_yms, test_mask) if keep]
    return residual, yms_test, lst_transform, lst_crs


def cosine_distance(pre: np.ndarray, post: np.ndarray) -> np.ndarray:
    pre_flat = pre.reshape(pre.shape[0], -1).T
    post_flat = post.reshape(post.shape[0], -1).T
    numer = np.sum(pre_flat * post_flat, axis=1)
    denom = np.linalg.norm(pre_flat, axis=1) * np.linalg.norm(post_flat, axis=1)
    cosine = np.full(pre_flat.shape[0], np.nan, dtype=np.float32)
    valid = denom > 0
    cosine[valid] = 1.0 - (numer[valid] / denom[valid])
    return cosine.reshape(pre.shape[1], pre.shape[2])


def euclidean_distance(pre: np.ndarray, post: np.ndarray) -> np.ndarray:
    delta = post - pre
    return np.sqrt(np.sum(delta * delta, axis=0))


def first_delta_component(pre: np.ndarray, post: np.ndarray) -> np.ndarray:
    delta = (post - pre).reshape(pre.shape[0], -1).T
    valid = np.isfinite(delta).all(axis=1)
    output = np.full(delta.shape[0], np.nan, dtype=np.float32)
    if valid.sum() < 2:
        return output.reshape(pre.shape[1], pre.shape[2])

    pca = PCA(n_components=1, random_state=42)
    output[valid] = pca.fit_transform(delta[valid]).ravel()
    return output.reshape(pre.shape[1], pre.shape[2])


def zscore(arr: np.ndarray) -> np.ndarray:
    valid = np.isfinite(arr)
    out = np.full_like(arr, np.nan, dtype=np.float32)
    if valid.sum() == 0:
        return out
    values = arr[valid]
    std = values.std()
    if std == 0:
        out[valid] = 0.0
    else:
        out[valid] = (values - values.mean()) / std
    return out


def finite_corr(a: np.ndarray, b: np.ndarray | None) -> float:
    if b is None:
        return np.nan
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 2:
        return np.nan
    return float(np.corrcoef(a[valid], b[valid])[0, 1])


def assign_quadrants(
    change_map: np.ndarray,
    residual_map: np.ndarray,
    quantile: float,
) -> tuple[np.ndarray, dict[str, float]]:
    valid = np.isfinite(change_map) & np.isfinite(residual_map)
    change_vals = change_map[valid]
    residual_vals = residual_map[valid]

    change_thr = float(np.quantile(change_vals, quantile))
    cooling_thr = float(np.quantile(residual_vals, quantile))

    quadrant = np.zeros(change_map.shape, dtype=np.uint8)
    high_change = change_map >= change_thr
    high_cooling = residual_map >= cooling_thr

    quadrant[valid] = 4
    quadrant[high_change & high_cooling] = 1
    quadrant[high_change & (residual_map < 0)] = 2
    quadrant[(~high_change) & high_cooling & valid] = 3

    union = np.logical_or(high_change, high_cooling) & valid
    overlap = high_change & high_cooling & valid
    stats = {
        "change_threshold": change_thr,
        "cooling_threshold": cooling_thr,
        "overlap_share_pct": float(overlap.sum() / valid.sum() * 100.0),
        "hotspot_jaccard": float(overlap.sum() / union.sum()) if union.sum() else np.nan,
    }
    return quadrant, stats


def hotspot_table(
    change_map: np.ndarray,
    residual_map: np.ndarray,
    delta_lst: np.ndarray,
    delta_ndvi: np.ndarray | None,
    quadrant: np.ndarray,
    transform,
    top_k: int = 25,
) -> pd.DataFrame:
    valid = np.isfinite(change_map) & np.isfinite(residual_map)
    combined = zscore(change_map) + zscore(residual_map)
    rows, cols = np.where(valid)
    scores = combined[valid]
    order = np.argsort(scores)[::-1][:top_k]

    records = []
    for idx in order:
        row = int(rows[idx])
        col = int(cols[idx])
        lon, lat = transform_xy(transform, row, col, offset="center")
        records.append(
            {
                "rank": len(records) + 1,
                "row": row,
                "col": col,
                "lon": float(lon),
                "lat": float(lat),
                "cosine_change": float(change_map[row, col]),
                "cooling_residual_c": float(residual_map[row, col]),
                "delta_lst_c": float(delta_lst[row, col]),
                "delta_ndvi": (
                    float(delta_ndvi[row, col])
                    if delta_ndvi is not None and np.isfinite(delta_ndvi[row, col])
                    else np.nan
                ),
                "combined_zscore": float(combined[row, col]),
                "quadrant": int(quadrant[row, col]),
            }
        )
    return pd.DataFrame.from_records(records)


def plot_alignment(
    change_map: np.ndarray,
    residual_map: np.ndarray,
    residual_raw: np.ndarray,
    delta_lst: np.ndarray,
    delta_ndvi: np.ndarray | None,
    quadrant: np.ndarray,
    pearson_r: float,
    lst_corr: float,
    ndvi_corr: float,
    baseline_year: int,
    compare_year: int,
    out_path: Path,
) -> None:
    valid = np.isfinite(change_map) & np.isfinite(residual_map)
    quad_colors = ["#000000", "#1B9E77", "#D95F02", "#7570B3", "#D9D9D9"]
    quad_cmap = mcolors.ListedColormap(quad_colors)
    quad_norm = mcolors.BoundaryNorm(np.arange(-0.5, 5.5, 1.0), quad_cmap.N)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    im0 = axes[0, 0].imshow(change_map, cmap="magma")
    axes[0, 0].set_title(
        f"(a) AlphaEarth cosine change\n{baseline_year} → {compare_year}"
    )
    plt.colorbar(im0, ax=axes[0, 0], shrink=0.82, label="Cosine distance")

    lim = np.nanmax(np.abs(residual_raw))
    im1 = axes[0, 1].imshow(residual_raw, cmap="RdBu", vmin=-lim, vmax=lim)
    axes[0, 1].set_title("(b) ConvLSTM cooling residual\n(predicted − observed °C)")
    plt.colorbar(im1, ax=axes[0, 1], shrink=0.82, label="°C")

    lst_lim = np.nanmax(np.abs(delta_lst))
    im2 = axes[0, 2].imshow(delta_lst, cmap="RdBu_r", vmin=-lst_lim, vmax=lst_lim)
    axes[0, 2].set_title(f"(c) Annual mean ΔLST\n{baseline_year} → {compare_year}")
    plt.colorbar(im2, ax=axes[0, 2], shrink=0.82, label="°C")

    axes[1, 0].imshow(quadrant, cmap=quad_cmap, norm=quad_norm)
    axes[1, 0].set_title("(d) Quadrant map")
    labels = [
        "1 = high change + high cooling",
        "2 = high change + warming",
        "3 = cooling without high change",
        "4 = background / low-signal",
    ]
    handles = [
        plt.Line2D([0], [0], marker="s", color="w", label=label, markersize=9,
                   markerfacecolor=quad_colors[i])
        for i, label in zip((1, 2, 3, 4), labels)
    ]
    legend = axes[1, 0].legend(
        handles=handles,
        loc="lower left",
        fontsize=8,
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=0.95,
    )
    for text in legend.get_texts():
        text.set_color("black")

    if delta_ndvi is not None:
        ndvi_lim = np.nanmax(np.abs(delta_ndvi))
        im3 = axes[1, 1].imshow(delta_ndvi, cmap="BrBG", vmin=-ndvi_lim, vmax=ndvi_lim)
        axes[1, 1].set_title(f"(e) Annual mean ΔNDVI\n{baseline_year} → {compare_year}")
        plt.colorbar(im3, ax=axes[1, 1], shrink=0.82, label="NDVI")
    else:
        axes[1, 1].axis("off")
        axes[1, 1].text(
            0.5,
            0.5,
            "No annual NDVI raster found.\nRun export with --with-ndvi.",
            ha="center",
            va="center",
            fontsize=11,
        )

    axes[1, 2].scatter(
        change_map[valid],
        residual_map[valid],
        s=18,
        alpha=0.7,
        c=zscore(change_map)[valid],
        cmap="viridis",
        edgecolor="none",
    )
    axes[1, 2].axhline(0, color="#444", lw=1)
    axes[1, 2].set_xlabel("AlphaEarth cosine change")
    axes[1, 2].set_ylabel("Cooling residual (°C)")
    subtitle = [f"r(change, cooling) = {pearson_r:.3f}", f"r(change, ΔLST) = {lst_corr:.3f}"]
    if np.isfinite(ndvi_corr):
        subtitle.append(f"r(change, ΔNDVI) = {ndvi_corr:.3f}")
    axes[1, 2].set_title("(f) Cell-wise alignment\n" + " · ".join(subtitle))

    for ax in (axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]):
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        "Chengdu AlphaEarth change detection aligned with cooling, ΔLST, and ΔNDVI",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    model = load_model(MODEL_PATH)
    target_shape = model_grid_shape(model)

    catalog = discover_embedding_files(args.embedding_dir)
    baseline_year = args.baseline_year
    compare_year = resolve_compare_year(args.compare_year, catalog)

    if baseline_year not in catalog:
        raise FileNotFoundError(
            f"Baseline year {baseline_year} not found in {args.embedding_dir}."
        )
    if compare_year not in catalog:
        raise FileNotFoundError(
            f"Compare year {compare_year} not found in {args.embedding_dir}."
        )
    if compare_year <= baseline_year:
        raise ValueError("compare-year must be later than baseline-year.")

    print("=" * 60)
    print("Chengdu AlphaEarth alignment analysis")
    print("=" * 60)
    print(f"Embedding dir : {args.embedding_dir}")
    print(f"Baseline year : {baseline_year}")
    print(f"Compare year  : {compare_year}")
    print(f"ConvLSTM grid : {target_shape[0]}x{target_shape[1]}")

    pre, transform, ae_crs = read_multiband(catalog[baseline_year], target_shape)
    post, _, _ = read_multiband(catalog[compare_year], target_shape)
    if pre.shape[0] != post.shape[0]:
        raise ValueError("Embedding band counts differ between baseline and compare year.")

    change_map = cosine_distance(pre, post)
    euclid_map = euclidean_distance(pre, post)
    delta_pc1 = first_delta_component(pre, post)

    residual_raw, post_policy_months, lst_transform, lst_crs = compute_cooling_residual(model)

    # Reproject ConvLSTM residual from Landsat tile grid → AlphaEarth grid
    # so that pixel (i,j) in residual_map aligns with pixel (i,j) in change_map
    from rasterio.warp import reproject as warp_reproject
    from rasterio.warp import Resampling as WarpResampling
    residual_map = np.full(target_shape, np.nan, dtype=np.float32)
    warp_reproject(
        source=residual_raw.astype(np.float32),
        destination=residual_map,
        src_transform=lst_transform,
        src_crs=lst_crs,
        dst_transform=transform,
        dst_crs=ae_crs,
        resampling=WarpResampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )

    delta_lst = annual_lst_mean(compare_year, target_shape) - annual_lst_mean(
        baseline_year, target_shape
    )
    ndvi_catalog = discover_singleband_year_files(
        args.embedding_dir, "ndvi_annual_chengdu_"
    )
    if baseline_year in ndvi_catalog and compare_year in ndvi_catalog:
        ndvi_pre, _ = read_singleband(ndvi_catalog[baseline_year], target_shape)
        ndvi_post, _ = read_singleband(ndvi_catalog[compare_year], target_shape)
        delta_ndvi = ndvi_post - ndvi_pre
    else:
        delta_ndvi = None

    valid = np.isfinite(change_map) & np.isfinite(residual_map)
    pearson_r = float(np.corrcoef(change_map[valid], residual_map[valid])[0, 1])
    lst_corr = finite_corr(change_map, delta_lst)
    ndvi_corr = finite_corr(change_map, delta_ndvi)

    quadrant, hotspot_stats = assign_quadrants(
        change_map, residual_map, args.hotspot_quantile
    )
    hotspots = hotspot_table(
        change_map, residual_map, delta_lst, delta_ndvi, quadrant, transform
    )

    stem = f"{baseline_year}_{compare_year}"
    fig_path = OUT_FIG_DIR / f"chengdu_alphaearth_alignment_{stem}.png"
    summary_path = OUT_TAB_DIR / f"chengdu_alphaearth_summary_{stem}.csv"
    hotspot_path = OUT_TAB_DIR / f"chengdu_alphaearth_hotspots_{stem}.csv"

    plot_alignment(
        change_map,
        residual_map,
        residual_raw,
        delta_lst,
        delta_ndvi,
        quadrant,
        pearson_r,
        lst_corr,
        ndvi_corr,
        baseline_year,
        compare_year,
        fig_path,
    )

    np.save(OUT_ARR_DIR / f"cosine_change_{stem}.npy", change_map)
    np.save(OUT_ARR_DIR / f"euclidean_change_{stem}.npy", euclid_map)
    np.save(OUT_ARR_DIR / f"delta_pc1_{stem}.npy", delta_pc1)
    np.save(OUT_ARR_DIR / "cooling_residual.npy", residual_map)
    np.save(OUT_ARR_DIR / f"delta_lst_{stem}.npy", delta_lst)
    if delta_ndvi is not None:
        np.save(OUT_ARR_DIR / f"delta_ndvi_{stem}.npy", delta_ndvi)
    np.save(OUT_ARR_DIR / f"quadrant_map_{stem}.npy", quadrant)

    summary = pd.DataFrame(
        [
            {
                "baseline_year": baseline_year,
                "compare_year": compare_year,
                "embedding_bands": pre.shape[0],
                "grid_height": pre.shape[1],
                "grid_width": pre.shape[2],
                "valid_cells": int(valid.sum()),
                "cosine_change_mean": float(np.nanmean(change_map)),
                "cosine_change_p90": float(np.nanquantile(change_map, 0.9)),
                "euclidean_change_mean": float(np.nanmean(euclid_map)),
                "cooling_residual_mean_c": float(np.nanmean(residual_map)),
                "cooling_residual_p90_c": float(np.nanquantile(residual_map, 0.9)),
                "pearson_r": pearson_r,
                "corr_change_delta_lst": lst_corr,
                "corr_change_delta_ndvi": ndvi_corr,
                "delta_lst_mean_c": float(np.nanmean(delta_lst)),
                "delta_lst_p90_c": float(np.nanquantile(delta_lst, 0.9)),
                "delta_ndvi_mean": float(np.nanmean(delta_ndvi)) if delta_ndvi is not None else np.nan,
                "delta_ndvi_p90": float(np.nanquantile(delta_ndvi, 0.9)) if delta_ndvi is not None else np.nan,
                "post_policy_months": len(post_policy_months),
                **hotspot_stats,
            }
        ]
    )
    summary.to_csv(summary_path, index=False)
    hotspots.to_csv(hotspot_path, index=False)

    print(f"Saved figure : {fig_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved hotspots: {hotspot_path}")
    print(f"Post-policy months used in residual map: {len(post_policy_months)}")


if __name__ == "__main__":
    main()
