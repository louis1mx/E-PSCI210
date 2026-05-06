---
editor_options: 
  markdown: 
    wrap: 72
---

# Chengdu Urban Cooling Project

## AI for Earth & Planetary Sciences — E-PSCI 210

### Research question

Do Chengdu's park-city and greening policies produce detectable and
spatially localised cooling signals in Landsat-observed land surface
temperature? Where these signals exist, do they co-locate with
measurable surface-form change detected by satellite foundation model
embeddings?

------------------------------------------------------------------------

## Overview

This project studies Chengdu, China, which was designated China's first
National Park-City Demonstration Zone in January 2022. The analysis
combines three complementary methods to evaluate whether and where the
policy produced a cooling effect:

| Layer | Method | What it answers |
|---|---|---|
| Statistical baseline | ITS regression | Did city-average LST shift after January 2022, controlling for weather and seasonality? |
| Spatiotemporal AI | ConvLSTM | Where did post-policy LST deviate from a pre-policy learned counterfactual? |
| Surface change detection | AlphaEarth embeddings | Did locations with cooling also experience measurable land-surface transformation? |

The ITS model does not find a statistically significant city-wide
cooling break. The ConvLSTM residual maps suggest localized cooling in
western and southwestern Chengdu. AlphaEarth embedding change is weakly
but non-trivially correlated with cooling residuals (*r* = 0.113) and
ΔNDVI (*r* = 0.112), and negatively correlated with ΔLST (*r* = −0.124).
The overlap between high-change and high-cooling cells covers 14.95% of
valid grid cells (Jaccard = 0.255), indicating that surface
transformation and thermal anomalies are spatially related but not
co-extensive.

------------------------------------------------------------------------

## Project structure

```
chengdu_project/
├── scripts/
│   ├── 01_download_gee.py       # Landsat LST + NDVI (GEE Python API)
│   ├── 02_download_era5.py      # ERA5 weather controls (CDS API)
│   ├── 03_build_panel.py        # Merge → monthly panel CSV
│   ├── 04_descriptive.py        # Trend figures (Fig 1–5)
│   ├── 05_causal_impact.py      # ITS regression (Fig 6–7)
│   └── 06_convlstm.py           # ConvLSTM model (Fig 8–9)
├── AlphaEarth/
│   ├── 01_export_alphaearth_chengdu.py   # Export annual AlphaEarth embeddings via GEE
│   └── 02_analyze_chengdu_alphaearth.py  # Embedding change detection and hotspot identification
├── data/
│   ├── raw/gee/                 # Landsat + NDVI GeoTIFF tiles
│   ├── raw/era5/                # ERA5 NetCDF + CSV
│   └── panel/                   # chengdu_monthly_panel.csv  ← main panel file
├── outputs/
│   ├── figures/                 # All publication figures
│   ├── tables/                  # Regression results, metrics
│   └── models/                  # Saved ConvLSTM weights
└── requirements.txt
```

------------------------------------------------------------------------

## Run order

### 0. Install dependencies

```bash
source venv/bin/activate
pip install -r requirements.txt
```

### 1. Download Landsat LST and NDVI (GEE)

Edit `PROJECT_ID` in `scripts/01_download_gee.py`, then:

```bash
earthengine authenticate        # once only
python scripts/01_download_gee.py
```

Downloads monthly Landsat LST composites and NDVI CSVs, and submits
GeoTIFF export tasks to Google Drive (~120 tasks). After the script
finishes, go to [Earth Engine Tasks](https://code.earthengine.google.com/)
and run all pending exports. Download the resulting tiles into
`data/raw/gee/chengdu_lst_tiles/`.

### 2. Download ERA5 weather controls

```bash
python scripts/02_download_era5.py
```

### 3. Build monthly panel

```bash
python scripts/03_build_panel.py
```

Output: `data/panel/chengdu_monthly_panel.csv` (120 rows × 18 cols).

### 4. Descriptive figures

```bash
python scripts/04_descriptive.py
```

Produces Figures 1–5 in `outputs/figures/`.

### 5. ITS regression

```bash
python scripts/05_causal_impact.py
```

Produces Figures 6–7 and regression results tables.

### 6. ConvLSTM model

```bash
python scripts/06_convlstm.py
```

Produces Figures 8–9, saved model weights, and metrics CSV.

### 7. AlphaEarth embedding analysis

See `AlphaEarth/README.md` for the full run order. In brief:

```bash
python AlphaEarth/01_export_alphaearth_chengdu.py   # queue GEE exports
python AlphaEarth/02_analyze_chengdu_alphaearth.py  # change detection and alignment
```

------------------------------------------------------------------------

## Key outputs

| File | Paper section |
|---|---|
| `outputs/figures/fig1_lst_trend.png` | Results: descriptive |
| `outputs/figures/fig2_uhi_trend.png` | Results: descriptive |
| `outputs/figures/fig6_its_regression.png` | Results: ITS baseline |
| `outputs/figures/fig7_counterfactual.png` | Results: ITS counterfactual |
| `outputs/figures/fig8_convlstm_residuals.png` | Results: spatial cooling map |
| `outputs/figures/fig9_convlstm_timeseries.png` | Results: temporal model fit |
| `outputs/tables/its_results.txt` | Supplementary regression table |
| `outputs/tables/convlstm_metrics.csv` | Model evaluation |
| `AlphaEarth/outputs/figures/chengdu_alphaearth_alignment_2021_2025.png` | Results: embedding alignment |
| `AlphaEarth/outputs/tables/chengdu_alphaearth_summary_2021_2025.csv` | Results: AlphaEarth metrics |
