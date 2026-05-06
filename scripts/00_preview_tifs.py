"""
Chengdu Urban Cooling Project — Script 00
Purpose : Preview / QC all LST GeoTIFF tiles in data/raw/gee/chengdu_lst_tiles/

Outputs (saved to outputs/tif_preview/):
  ├── tif_overview.png      — grid of every available month (thumbnails)
  ├── tif_timeseries.png    — mean LST per tile over time (line chart)
  ├── tif_seasonal.png      — mean LST by month across all years (box plot)
  ├── tif_coverage.png      — heatmap: which months are present / all-NaN
  └── single/               — one high-res map per tile (on demand)

Usage:
  # Quick overview (default — all 4 summary plots):
  python scripts/00_preview_tifs.py

  # Also save one hi-res map per tile (slow, ~2 min):
  python scripts/00_preview_tifs.py --single

  # Only a few tiles (space-separated YYYY_MM):
  python scripts/00_preview_tifs.py --tiles 2019_07 2022_08 2023_06

Dependencies:
  pip install rasterio numpy matplotlib seaborn tqdm
"""

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import numpy as np
import seaborn as sns
from tqdm import tqdm

try:
    import rasterio
    from rasterio.plot import reshape_as_image
except ImportError:
    raise SystemExit("rasterio not found — run:  pip install rasterio")

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
TIF_DIR   = ROOT / "data" / "raw" / "gee" / "chengdu_lst_tiles"
OUT_DIR   = ROOT / "outputs" / "tif_preview"
SINGLE_DIR = OUT_DIR / "single"

# ── Style ────────────────────────────────────────────────────────────────────
PALETTE   = "RdYlBu_r"          # diverging: blue=cool, red=hot
BG        = "#0f1117"
FG        = "#e8eaf0"
ACCENT    = "#7c9ee8"
GRID_CLR  = "#2a2d38"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    "#16192a",
    "axes.edgecolor":    GRID_CLR,
    "axes.labelcolor":   FG,
    "xtick.color":       FG,
    "ytick.color":       FG,
    "text.color":        FG,
    "grid.color":        GRID_CLR,
    "grid.linewidth":    0.5,
    "font.family":       "DejaVu Sans",
    "axes.titlepad":     10,
})

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_tifs(tif_dir: Path) -> list[dict]:
    """
    Read every *.tif in tif_dir.
    Returns list of dicts sorted by (year, month):
        { 'path', 'year', 'month', 'label', 'arr', 'valid_pct' }
    arr is a 2-D float32 masked-array (NaN where no data).
    """
    records = []
    tifs = sorted(tif_dir.glob("LST_*.tif"))
    if not tifs:
        raise FileNotFoundError(f"No LST_*.tif found in {tif_dir}")

    print(f"Found {len(tifs)} TIF files — loading …")
    for p in tqdm(tifs, unit="tif", ncols=70):
        parts = p.stem.split("_")          # LST_YYYY_MM
        if len(parts) != 3:
            continue
        year, month = int(parts[1]), int(parts[2])

        with rasterio.open(p) as src:
            raw = src.read(1).astype(np.float32)
            nodata = src.nodata

        # Mask nodata
        if nodata is not None:
            raw[raw == nodata] = np.nan
        raw[raw == 0.0] = np.nan          # some tiles have 0 as fill

        valid_pct = np.sum(~np.isnan(raw)) / raw.size * 100
        records.append({
            "path":      p,
            "year":      year,
            "month":     month,
            "label":     f"{year}\n{MONTH_NAMES[month-1]}",
            "arr":       raw,
            "valid_pct": valid_pct,
        })

    records.sort(key=lambda x: (x["year"], x["month"]))
    return records


def _global_clim(records, lo=2, hi=98):
    """Robust global colour limits across all tiles."""
    all_vals = np.concatenate([r["arr"][~np.isnan(r["arr"])].ravel()
                               for r in records if r["valid_pct"] > 0])
    if all_vals.size == 0:
        return 0, 40
    return float(np.percentile(all_vals, lo)), float(np.percentile(all_vals, hi))


def _mean_lst(r):
    vals = r["arr"][~np.isnan(r["arr"])]
    return float(vals.mean()) if vals.size else np.nan


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Thumbnail grid
# ─────────────────────────────────────────────────────────────────────────────

def plot_overview(records: list[dict], out: Path):
    vmin, vmax = _global_clim(records)
    n = len(records)
    ncols = 12
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 2.1, nrows * 1.9 + 1.4),
                             gridspec_kw={"hspace": 0.55, "wspace": 0.08})
    fig.patch.set_facecolor(BG)
    fig.suptitle("Chengdu Monthly LST — Full Archive Preview",
                 fontsize=17, fontweight="bold", color=FG, y=0.99)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, rec in zip(axes_flat, records):
        ax.set_facecolor(BG)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID_CLR)

        arr = rec["arr"]
        if rec["valid_pct"] < 0.5:          # mostly-blank tile
            ax.text(0.5, 0.5, "NO\nDATA", ha="center", va="center",
                    fontsize=6, color="#555566",
                    transform=ax.transAxes)
        else:
            im = ax.imshow(arr, cmap=PALETTE, vmin=vmin, vmax=vmax,
                           interpolation="nearest", origin="upper")

        # border colour by coverage
        colour = ("#4fc97a" if rec["valid_pct"] > 50 else
                  "#f0c040" if rec["valid_pct"] > 5 else "#e05555")
        for sp in ax.spines.values():
            sp.set_edgecolor(colour); sp.set_linewidth(1.3)
        ax.set_title(rec["label"], fontsize=6.5, color=FG, pad=2)

    # Hide unused axes
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    # Shared colourbar
    sm = plt.cm.ScalarMappable(cmap=PALETTE,
                                norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.15, 0.01, 0.70, 0.012])
    cb = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cb.set_label("LST (°C)", color=FG, fontsize=10)
    cb.ax.xaxis.set_tick_params(color=FG)
    plt.setp(cb.ax.xaxis.get_ticklabels(), color=FG, fontsize=8)

    # Legend for border colour
    legend_elements = [
        Patch(facecolor=BG, edgecolor="#4fc97a", linewidth=1.5, label="> 50 % valid"),
        Patch(facecolor=BG, edgecolor="#f0c040", linewidth=1.5, label="5–50 % valid"),
        Patch(facecolor=BG, edgecolor="#e05555", linewidth=1.5, label="< 5 % valid"),
    ]
    fig.legend(handles=legend_elements, loc="lower right", ncol=3,
               fontsize=7.5, framealpha=0.0,
               bbox_to_anchor=(0.92, 0.025))

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ✓  Overview grid  →  {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — Time-series of mean LST
# ─────────────────────────────────────────────────────────────────────────────

def plot_timeseries(records: list[dict], out: Path):
    xs     = np.arange(len(records))
    labels = [f"{r['year']}\n{MONTH_NAMES[r['month']-1]}" for r in records]
    means  = np.array([_mean_lst(r) for r in records])
    valids = np.array([r["valid_pct"] for r in records])

    # Color points by valid coverage
    colours = np.where(valids > 50, "#4fc97a",
              np.where(valids > 5,  "#f0c040", "#e05555"))

    fig, ax = plt.subplots(figsize=(22, 5))
    fig.patch.set_facecolor(BG)

    # Line (skip NaN gaps)
    valid_mask = ~np.isnan(means)
    ax.plot(xs[valid_mask], means[valid_mask], color=ACCENT,
            linewidth=1.4, alpha=0.6, zorder=1)

    # Scatter with coverage colour
    sc = ax.scatter(xs, means, c=colours, s=55, zorder=3,
                    edgecolors="white", linewidths=0.4)

    # Thin vertical lines at year boundaries
    years = [r["year"] for r in records]
    for i, (a, b) in enumerate(zip(years[:-1], years[1:])):
        if a != b:
            ax.axvline(i + 0.5, color=GRID_CLR, linewidth=0.8, linestyle="--")

    # Seasonal shading (summer = Jun–Aug)
    for i, r in enumerate(records):
        if r["month"] in (6, 7, 8):
            ax.axvspan(i - 0.4, i + 0.4, alpha=0.06, color="#ff7755", lw=0)

    ax.set_title("Monthly Mean LST Over Time", fontsize=14, fontweight="bold",
                 color=FG)
    ax.set_ylabel("Mean LST (°C)", fontsize=11)
    ax.set_xlabel("Month", fontsize=10)
    ax.set_xticks(xs[::3])
    ax.set_xticklabels(labels[::3], fontsize=6.5)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f °C"))
    ax.grid(True, axis="y", alpha=0.35)

    # Legend
    leg = [
        Patch(color="#4fc97a", label="> 50 % valid pixels"),
        Patch(color="#f0c040", label="5–50 % valid"),
        Patch(color="#e05555", label="< 5 % valid"),
    ]
    ax.legend(handles=leg, fontsize=8.5, framealpha=0.15, loc="upper left")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ✓  Time-series     →  {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — Seasonal box-plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_seasonal(records: list[dict], out: Path):
    from collections import defaultdict
    by_month = defaultdict(list)
    for r in records:
        m = _mean_lst(r)
        if not np.isnan(m) and r["valid_pct"] > 5:
            by_month[r["month"]].append(m)

    data   = [by_month.get(m, []) for m in range(1, 13)]
    medians = [np.median(d) if d else np.nan for d in data]

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor(BG)

    # Box colours: gradient from cold (blue) to hot (red)
    cmap   = plt.get_cmap("RdYlBu_r")
    norm   = mcolors.Normalize(vmin=min(m for m in medians if not np.isnan(m)),
                                vmax=max(m for m in medians if not np.isnan(m)))

    bplot = ax.boxplot(data, patch_artist=True, notch=False,
                       medianprops=dict(color="white", linewidth=2),
                       whiskerprops=dict(color=FG, linewidth=1),
                       capprops=dict(color=FG, linewidth=1),
                       flierprops=dict(marker="o", markersize=3,
                                       markerfacecolor=ACCENT, alpha=0.6))
    for patch, med in zip(bplot["boxes"], medians):
        if not np.isnan(med):
            patch.set_facecolor(cmap(norm(med)))
            patch.set_alpha(0.85)
            patch.set_edgecolor(FG)
        else:
            patch.set_facecolor(GRID_CLR)

    ax.set_title("Seasonal Distribution of Mean LST (valid tiles only)",
                 fontsize=14, fontweight="bold", color=FG)
    ax.set_ylabel("Mean LST (°C)", fontsize=11)
    ax.set_xlabel("Month", fontsize=10)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_NAMES)
    ax.grid(True, axis="y", alpha=0.3)

    # Count annotation
    for i, d in enumerate(data):
        ax.text(i + 1, ax.get_ylim()[0], f"n={len(d)}",
                ha="center", va="bottom", fontsize=7.5, color=FG, alpha=0.7)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ✓  Seasonal box    →  {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4 — Coverage heatmap (year × month)
# ─────────────────────────────────────────────────────────────────────────────

def plot_coverage(records: list[dict], out: Path):
    years  = sorted(set(r["year"]  for r in records))
    months = list(range(1, 13))

    # Matrix: valid_pct; -1 = tile missing
    grid = np.full((len(years), 12), -1.0)
    for r in records:
        yi = years.index(r["year"])
        mi = r["month"] - 1
        grid[yi, mi] = r["valid_pct"]

    fig, ax = plt.subplots(figsize=(13, len(years) * 0.62 + 1.8))
    fig.patch.set_facecolor(BG)

    # Custom colourmap: grey for missing, green gradient for coverage
    cmap_base = plt.get_cmap("YlGn")
    cmap_list = [(0.12, 0.12, 0.18, 1.0)]   # dark grey = missing (-1)
    cmap_list += [cmap_base(v) for v in np.linspace(0, 1, 256)]
    cmap_cov  = mcolors.LinearSegmentedColormap.from_list(
        "cov", cmap_list, N=257)

    im = ax.imshow(grid, cmap=cmap_cov, vmin=-1, vmax=100,
                   aspect="auto", interpolation="nearest")

    # Annotations
    for yi in range(len(years)):
        for mi in range(12):
            v = grid[yi, mi]
            if v < 0:
                txt, col = "—", "#444455"
            elif v < 0.5:
                txt, col = "0%", "#cc4444"
            else:
                txt, col = f"{v:.0f}%", "white" if v < 60 else "#0a1a0a"
            ax.text(mi, yi, txt, ha="center", va="center",
                    fontsize=7, color=col, fontweight="bold")

    ax.set_xticks(range(12))
    ax.set_xticklabels(MONTH_NAMES, fontsize=9)
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels(years, fontsize=9)
    ax.set_title("Valid Pixel Coverage per Tile  (% of pixels not NaN)",
                 fontsize=13, fontweight="bold", color=FG, pad=12)
    ax.set_xlabel("Month", fontsize=10)
    ax.set_ylabel("Year",  fontsize=10)

    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02,
                      ticks=[0, 25, 50, 75, 100])
    cb.set_label("Valid pixels (%)", color=FG, fontsize=9)
    cb.ax.yaxis.set_tick_params(color=FG)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=FG, fontsize=8)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ✓  Coverage heatmap →  {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Extra — Individual hi-res tile maps
# ─────────────────────────────────────────────────────────────────────────────

def plot_single(rec: dict, out: Path, vmin: float, vmax: float):
    arr = rec["arr"]
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor("#050810")

    im = ax.imshow(arr, cmap=PALETTE, vmin=vmin, vmax=vmax,
                   interpolation="bilinear", origin="upper")
    ax.set_title(
        f"Chengdu LST — {rec['year']} {MONTH_NAMES[rec['month']-1]}"
        f"   (valid: {rec['valid_pct']:.1f} %)",
        fontsize=13, fontweight="bold", color=FG, pad=10)
    ax.axis("off")

    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03,
                      orientation="vertical")
    cb.set_label("LST (°C)", color=FG, fontsize=10)
    cb.ax.yaxis.set_tick_params(color=FG)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=FG)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Summary stats printout
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(records: list[dict]):
    print("\n" + "=" * 58)
    print(f"  TIF Summary   ({len(records)} tiles found)")
    print("=" * 58)
    means   = [_mean_lst(r)   for r in records]
    valids  = [r["valid_pct"] for r in records]

    valid_means = [m for m in means if not np.isnan(m)]
    if valid_means:
        print(f"  Mean LST range   : {min(valid_means):.2f} – "
              f"{max(valid_means):.2f} °C")
        print(f"  Grand mean LST   : {np.mean(valid_means):.2f} °C")

    all_valid_pct = [v for v in valids]
    good  = sum(v > 50 for v in all_valid_pct)
    ok    = sum(5 < v <= 50  for v in all_valid_pct)
    bad   = sum(0 < v <= 5   for v in all_valid_pct)
    empty = sum(v == 0        for v in all_valid_pct)
    miss  = len([r for r in records if r["valid_pct"] == 0])

    print(f"\n  Coverage quality :")
    print(f"    > 50 % valid    : {good:>3} tiles  ✓ good")
    print(f"    5–50 % valid    : {ok:>3} tiles  ⚠ partial")
    print(f"    < 5 % valid     : {bad:>3} tiles  ✗ poor")
    print(f"    All-NaN / empty : {empty:>3} tiles  ✗ blank")

    years = sorted(set(r["year"] for r in records))
    print(f"\n  Year span        : {years[0]} – {years[-1]}")
    print(f"  Months present   : {len(records)} / {len(years)*12} possible")
    print("=" * 58 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Preview Chengdu LST GeoTIFF archive")
    p.add_argument("--single", action="store_true",
                   help="Also save individual hi-res map per tile")
    p.add_argument("--tiles", nargs="+", metavar="YYYY_MM",
                   help="Only process these tiles (e.g. --tiles 2019_07 2022_08)")
    return p.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    records = load_tifs(TIF_DIR)

    if args.tiles:
        wanted = {t.strip() for t in args.tiles}
        records = [r for r in records
                   if f"{r['year']}_{r['month']:02d}" in wanted]
        if not records:
            print(f"No tiles matched: {wanted}")
            return

    print_summary(records)

    print("Generating plots …")
    plot_overview(  records, OUT_DIR / "tif_overview.png")
    plot_timeseries(records, OUT_DIR / "tif_timeseries.png")
    plot_seasonal(  records, OUT_DIR / "tif_seasonal.png")
    plot_coverage(  records, OUT_DIR / "tif_coverage.png")

    if args.single:
        vmin, vmax = _global_clim(records)
        print(f"\nSaving {len(records)} individual tile maps …")
        SINGLE_DIR.mkdir(parents=True, exist_ok=True)
        for r in tqdm(records, unit="tile", ncols=70):
            fname = f"LST_{r['year']}_{r['month']:02d}.png"
            plot_single(r, SINGLE_DIR / fname, vmin, vmax)
        print(f"  ✓  Single maps  →  {SINGLE_DIR}/")

    print(f"\n✅  All outputs saved to  {OUT_DIR}/")
    print("   Open outputs/tif_preview/ to inspect the plots.")


if __name__ == "__main__":
    main()
