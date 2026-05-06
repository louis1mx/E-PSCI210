# Chengdu AlphaEarth

This directory is a single-city branch within `Version2` dedicated to AlphaEarth analysis for Chengdu. It does not extend to multiple cities.

It implements the main pipeline described in `outputs/E_PSCI_210_Draft.pdf`:

1. Use `2021` as the pre-policy baseline.
2. Use `2024` (or the latest available post-policy year) for AlphaEarth embedding comparison.
3. Compute embedding change.
4. Align embedding change with the existing `ConvLSTM` cooling residual.
5. Identify "high change + high cooling" hotspots in Chengdu.

## Scripts

- `01_export_alphaearth_chengdu.py`
  Queues exports of annual AlphaEarth GeoTIFFs for Chengdu via Google Earth Engine.
  Output grid is fixed at `64x64` to match the existing ConvLSTM input grid.
  Also supports exporting annual `NDVI` rasters on the same grid.

- `02_analyze_chengdu_alphaearth.py`
  Local analysis script. Reads annual AlphaEarth embeddings, `chengdu_lst_tiles`, and the saved `convlstm_model.keras`, then outputs change maps, cooling residual maps, quadrant maps, and hotspot tables.

## Input Dependencies

This pipeline reuses existing assets from `Version2`:

- `data/raw/gee/chengdu_lst_tiles/LST_*.tif`
- `outputs/models/convlstm_model.keras`

You also need to place the GEE-exported AlphaEarth annual GeoTIFFs at:

- `AlphaEarth/data/raw/alphaearth_chengdu_2021.tif`
- `AlphaEarth/data/raw/alphaearth_chengdu_2025.tif`

Files are identified by year in the filename; the naming convention above is recommended.

To include `ΔNDVI` in outputs, also place:

- `AlphaEarth/data/raw/ndvi_annual_chengdu_2021.tif`
- `AlphaEarth/data/raw/ndvi_annual_chengdu_2025.tif`

## Run Order

First, export AlphaEarth embeddings:

```bash
cd /Users/louis/Desktop/E-PSCI210/final_project/Version2
venv311/bin/python AlphaEarth/01_export_alphaearth_chengdu.py
```

To export only annual NDVI without re-queuing AlphaEarth embeddings:

```bash
cd /Users/louis/Desktop/E-PSCI210/final_project/Version2
venv311/bin/python AlphaEarth/01_export_alphaearth_chengdu.py --ndvi-only
```

Then run the GEE tasks in Earth Engine and download the results to `AlphaEarth/data/raw/`.

Next, run the Chengdu analysis:

```bash
cd /Users/louis/Desktop/E-PSCI210/final_project/Version2
venv311/bin/python AlphaEarth/02_analyze_chengdu_alphaearth.py \
  --baseline-year 2021 \
  --compare-year latest
```

## Outputs

The script produces:

- `AlphaEarth/outputs/figures/chengdu_alphaearth_alignment_2021_2025.png`
- `AlphaEarth/outputs/tables/chengdu_alphaearth_summary_2021_2025.csv`
- `AlphaEarth/outputs/tables/chengdu_alphaearth_hotspots_2021_2025.csv`
- `AlphaEarth/outputs/arrays/*.npy`

If `ndvi_annual_chengdu_*.tif` files are present, figures and tables will additionally include:

- `ΔNDVI`
- `corr(change, ΔNDVI)`
- `delta_ndvi` per hotspot

## Current Scope

This version is a Chengdu pilot. It does not loop over multiple cities, does not perform land-cover classification, and does not attempt to interpret AlphaEarth embeddings at the level of "new park / new building."

It strictly implements:

- Embedding distance change detection
- ConvLSTM cooling alignment
- Hotspot identification for Chengdu

To extend to multiple cities, the three main changes needed are:

- Parameterize the Chengdu center point and buffer radius
- Batch-export annual AlphaEarth rasters for multiple cities
- Aggregate per-city summary and hotspot tables
