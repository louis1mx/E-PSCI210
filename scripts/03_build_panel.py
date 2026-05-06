"""
Chengdu Urban Cooling Project — Script 03
Purpose : Merge all data sources into a single monthly panel CSV.
          Also derives all variables needed for CausalImpact and ConvLSTM.

Inputs  :
  data/raw/gee/landsat_lst_urban_monthly.csv
  data/raw/gee/landsat_lst_rural_monthly.csv
  data/raw/gee/ndvi_urban_monthly.csv
  data/raw/gee/ndvi_rural_monthly.csv
  data/raw/era5/chengdu_ERA5_monthly.csv

Outputs :
  data/panel/chengdu_monthly_panel.csv   ← main analysis file
  data/panel/variable_codebook.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

GEE_DIR   = Path('data/raw/gee')
ERA5_FILE = Path('data/raw/era5/chengdu_ERA5_monthly.csv')
OUT_DIR   = Path('data/panel')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_gee(path: Path, rename: dict) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns
                           if c in ('.geo', 'system:index', 'zone')],
                 errors='ignore')
    df = df.rename(columns=rename)
    if 'ym' not in df.columns:
        df['ym'] = df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2)
    targets = ['ym', 'year', 'month'] + list(rename.values())
    return df[[c for c in targets if c in df.columns]]


def build_panel() -> pd.DataFrame:
    print('Loading files …')

    lst_u = load_gee(GEE_DIR / 'landsat_lst_urban_monthly.csv',
                     {'LST_urban_C': 'LST_urban_C'})
    lst_r = load_gee(GEE_DIR / 'landsat_lst_rural_monthly.csv',
                     {'LST_rural_C': 'LST_rural_C'})
    vi_u  = load_gee(GEE_DIR / 'ndvi_urban_monthly.csv',
                     {'NDVI_urban': 'NDVI_urban', 'EVI_urban': 'EVI_urban'})
    vi_r  = load_gee(GEE_DIR / 'ndvi_rural_monthly.csv',
                     {'NDVI_rural': 'NDVI_rural', 'EVI_rural': 'EVI_rural'})
    era5  = pd.read_csv(ERA5_FILE)

    # ── Merge ────────────────────────────────────────────────
    KEY = ['ym', 'year', 'month']
    panel = (lst_u
             .merge(lst_r, on=KEY, how='outer')
             .merge(vi_u,  on=KEY, how='outer')
             .merge(vi_r,  on=KEY, how='outer')
             .merge(era5,  on=KEY, how='outer')
             .sort_values(['year', 'month'])
             .reset_index(drop=True))

    # ── Derived variables ────────────────────────────────────

    # Urban Heat Island intensity
    panel['UHI_intensity'] = panel['LST_urban_C'] - panel['LST_rural_C']

    # Policy dummy: Chengdu park-city demonstration window → 2022 onward
    panel['post_policy']   = (panel['year'] >= 2022).astype(int)

    # Linear time trend (1 = Jan 2016 … 120 = Dec 2025)
    panel['time_trend']    = (panel['year'] - 2016) * 12 + panel['month']

    # Post-policy slope-change term for ITS (centred at Jan 2022 = t=73)
    panel['trend_post']    = panel['post_policy'] * (panel['time_trend'] - 72)

    # Season (for fixed-effects and plotting)
    panel['season'] = pd.cut(panel['month'],
                             bins=[0,3,6,9,12],
                             labels=['Winter','Spring','Summer','Autumn'])

    # NDVI change from 2016-2021 baseline (greening intensity proxy)
    baseline_ndvi = (panel[panel['year'] <= 2021]
                     .groupby('month')['NDVI_urban'].mean()
                     .rename('NDVI_baseline'))
    panel = panel.merge(baseline_ndvi, on='month', how='left')
    panel['delta_NDVI'] = panel['NDVI_urban'] - panel['NDVI_baseline']

    # ── Diagnostics ──────────────────────────────────────────
    print('\nMissing values:')
    miss = panel.isnull().sum()
    miss = miss[miss > 0]
    print('  None' if miss.empty else miss.to_string())

    key_vars = ['LST_urban_C','LST_rural_C','UHI_intensity',
                'NDVI_urban','temp_2m_C','precip_mm']
    print('\nSummary statistics:')
    print(panel[key_vars].describe().round(3).to_string())

    # ── Save ─────────────────────────────────────────────────
    out = OUT_DIR / 'chengdu_monthly_panel.csv'
    panel.to_csv(out, index=False)
    print(f'\n✅  Panel saved: {out}')
    print(f'   {len(panel)} rows × {len(panel.columns)} columns')

    # ── Codebook ─────────────────────────────────────────────
    codebook = pd.DataFrame([
        ('ym',             'Year-month string (YYYY-MM)',              '—',        'Time index'),
        ('year / month',   'Calendar year and month',                  '—',        'Time'),
        ('LST_urban_C',    'Urban monthly mean daytime LST (Landsat)', '°C',       'Primary outcome'),
        ('LST_rural_C',    'Rural monthly mean daytime LST (Landsat)', '°C',       'Reference / UHI'),
        ('UHI_intensity',  'UHI = urban LST − rural LST',             '°C',       'Secondary outcome'),
        ('NDVI_urban',     'Urban monthly mean NDVI (MODIS)',          '[0, 1]',   'Greening proxy'),
        ('EVI_urban',      'Urban monthly mean EVI (MODIS)',           '[0, 1]',   'Greening proxy (enhanced)'),
        ('NDVI_rural',     'Rural monthly mean NDVI (MODIS)',          '[0, 1]',   'Reference'),
        ('delta_NDVI',     'NDVI change vs 2016-2021 monthly mean',   '[0, 1]',   'Greening intensity for spatial regression'),
        ('temp_2m_C',      '2 m air temperature (ERA5)',               '°C',       'Weather control'),
        ('precip_mm',      'Monthly total precipitation (ERA5)',       'mm',       'Weather control'),
        ('solar_rad_Wm2',  'Surface solar radiation (ERA5)',           'W m⁻²',    'Weather control'),
        ('post_policy',    'Post-policy dummy (1 = 2022 onward)',      '0 / 1',    'Treatment indicator (CausalImpact)'),
        ('time_trend',     'Month index 1–120',                        'int',      'ITS linear trend'),
        ('trend_post',     'Post-policy slope-change term',            'int',      'ITS slope shift'),
        ('season',         'Meteorological season',                    'category', 'Plotting / fixed effects'),
    ], columns=['Variable', 'Description', 'Unit', 'Role'])
    codebook.to_csv(OUT_DIR / 'variable_codebook.csv', index=False)
    print(f'✅  Codebook saved: {OUT_DIR / "variable_codebook.csv"}')

    return panel


if __name__ == '__main__':
    panel = build_panel()
    print('\n✅  Script 03 complete. Next: python scripts/04_descriptive.py')
