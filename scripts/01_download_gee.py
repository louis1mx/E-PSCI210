"""
Chengdu Urban Cooling Project — Script 01
Purpose : Download satellite data via Google Earth Engine Python API

  Dataset A — Landsat Collection 2 Surface Temperature (30 m)
               Landsat 8 (2016-2022) + Landsat 9 (2022-2025)
               → replaces MODIS LST; higher spatial resolution for
                 neighbourhood-level cooling detection (per TA feedback)

  Dataset B — MOD13Q1 v6.1 NDVI / EVI (250 m)
               Greening intensity proxy

Outputs (saved to  data/raw/gee/):
  landsat_lst_urban_monthly.csv   — urban core  (0–25 km) monthly mean LST
  landsat_lst_rural_monthly.csv   — rural buffer (25–50 km) monthly mean LST
  landsat_lst_images/             — GeoTIFF tiles for ConvLSTM (one per month)
  ndvi_urban_monthly.csv
  ndvi_rural_monthly.csv

Prerequisites:
  pip install earthengine-api pandas
  earthengine authenticate
  → edit PROJECT_ID below with your GEE Cloud Project ID
"""

import ee
import pandas as pd
from pathlib import Path
import time

# ── CONFIG — edit this line ──────────────────────────────────
PROJECT_ID = 'e-psci210-project-louis'   # e.g. 'ee-yourname'

ee.Initialize(project=PROJECT_ID)

OUT_DIR = Path('data/raw/gee')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Study area ───────────────────────────────────────────────
# Chengdu city centre (Tianfu Square)
CENTER   = ee.Geometry.Point([104.0665, 30.5728])
URBAN    = CENTER.buffer(25_000)                     # 0–25 km
RURAL    = CENTER.buffer(50_000).difference(URBAN)   # 25–50 km ring

START, END = '2016-01-01', '2025-12-31'


# ════════════════════════════════════════════════════════════
# PART A  —  Landsat Surface Temperature
# ════════════════════════════════════════════════════════════

def mask_landsat(image):
    """
    Apply Landsat Collection 2 QA_PIXEL cloud/shadow mask and
    convert ST_B10 to Celsius.
    Works for both Landsat 8 and Landsat 9 (same band names).
    """
    qa = image.select('QA_PIXEL')
    # Bit 3 = cloud, Bit 4 = cloud shadow
    cloud_mask = qa.bitwiseAnd(1 << 3).eq(0).And(
                 qa.bitwiseAnd(1 << 4).eq(0))

    # ST_B10: scale 0.00341802, offset 149.0 → Kelvin
    lst_k = (image.select('ST_B10')
                  .multiply(0.00341802)
                  .add(149.0))
    lst_c = lst_k.subtract(273.15).rename('LST_C')
    return lst_c.updateMask(cloud_mask).copyProperties(
        image, ['system:time_start'])


def get_landsat_collection():
    """Merge Landsat 8 (2016-2021) and Landsat 9 (2022-2025)."""
    l8 = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
            .filterDate(START, '2022-01-01')
            .filterBounds(URBAN)
            .filter(ee.Filter.lt('CLOUD_COVER', 70))
            .map(mask_landsat))

    l9 = (ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
            .filterDate('2022-01-01', END)
            .filterBounds(URBAN)
            .filter(ee.Filter.lt('CLOUD_COVER', 70))
            .map(mask_landsat))

    return l8.merge(l9)


def extract_monthly_lst(collection, geometry, label):
    """Extract monthly mean LST over a geometry. Returns list of dicts."""
    results = []
    for year in range(2016, 2026):
        for month in range(1, 13):
            t0 = f'{year}-{month:02d}-01'
            ny = year + (1 if month == 12 else 0)
            nm = month % 12 + 1
            t1 = f'{ny}-{nm:02d}-01'

            monthly = collection.filterDate(t0, t1)
            row = {'year': year, 'month': month,
                   'ym': f'{year}-{month:02d}'}

            n = monthly.size().getInfo()
            if n == 0:
                row['LST_C'] = None
                print(f'  [{label}] {year}-{month:02d}  0 scenes — NaN')
                results.append(row)
                continue

            stats = (monthly.mean()
                            .reduceRegion(reducer=ee.Reducer.mean(),
                                          geometry=geometry,
                                          scale=30,
                                          maxPixels=1e10)
                            .getInfo())
            row['LST_C'] = stats.get('LST_C')
            v = f"{row['LST_C']:.3f}" if row['LST_C'] else 'NaN'
            print(f'  [{label}] {year}-{month:02d}  {n} scenes  LST={v} °C')
            results.append(row)
            time.sleep(0.4)

    return results


def download_landsat():
    print('\n' + '='*55)
    print('PART A  Landsat Surface Temperature (30 m)')
    print('='*55)
    col = get_landsat_collection()
    print(f'Total Landsat scenes available: {col.size().getInfo()}')

    print('\n  Urban core (0–25 km)')
    df_u = pd.DataFrame(extract_monthly_lst(col, URBAN,  'urban'))
    df_u = df_u.rename(columns={'LST_C': 'LST_urban_C'})

    print('\n  Rural buffer (25–50 km)')
    df_r = pd.DataFrame(extract_monthly_lst(col, RURAL, 'rural'))
    df_r = df_r.rename(columns={'LST_C': 'LST_rural_C'})

    df_u.to_csv(OUT_DIR / 'landsat_lst_urban_monthly.csv', index=False)
    df_r.to_csv(OUT_DIR / 'landsat_lst_rural_monthly.csv', index=False)
    print(f'\n  Saved to {OUT_DIR}')


# ════════════════════════════════════════════════════════════
# PART B  —  Export monthly Landsat GeoTIFFs for ConvLSTM
#            (clipped to urban core, 30 m, EPSG:4326)
# ════════════════════════════════════════════════════════════

def export_monthly_geotiffs():
    """
    Export one GeoTIFF per month to Google Drive.
    Each image = monthly median LST over the urban core.
    These are the input frames for the ConvLSTM model.

    After running this function, open the GEE Tasks panel at
    https://code.earthengine.google.com/ and click Run on each task.
    Files land in Google Drive → chengdu_lst_tiles/
    """
    print('\n' + '='*55)
    print('PART B  Exporting monthly GeoTIFFs for ConvLSTM')
    print('='*55)

    col = get_landsat_collection()
    count = 0

    for year in range(2016, 2026):
        for month in range(1, 13):
            t0 = f'{year}-{month:02d}-01'
            ny = year + (1 if month == 12 else 0)
            nm = month % 12 + 1
            t1 = f'{ny}-{nm:02d}-01'

            monthly = col.filterDate(t0, t1)
            n = monthly.size().getInfo()
            if n == 0:
                print(f'  {year}-{month:02d}  skipped (no scenes)')
                continue

            img = monthly.median().clip(URBAN)
            desc = f'LST_{year}_{month:02d}'

            task = ee.batch.Export.image.toDrive(
                image       = img,
                description = desc,
                folder      = 'chengdu_lst_tiles',
                fileNamePrefix = desc,
                region      = URBAN,
                scale       = 30,
                crs         = 'EPSG:4326',
                maxPixels   = 1e10,
            )
            task.start()
            count += 1
            print(f'  {year}-{month:02d}  export task submitted (n={n} scenes)')
            time.sleep(0.2)

    print(f'\n  {count} export tasks submitted.')
    print('  Open https://code.earthengine.google.com/ → Tasks → Run all')
    print('  Files will appear in Google Drive → chengdu_lst_tiles/')


# ════════════════════════════════════════════════════════════
# PART C  —  MODIS NDVI / EVI
# ════════════════════════════════════════════════════════════

def process_vi(image):
    qa   = image.select('SummaryQA').lte(1)
    ndvi = image.select('NDVI').multiply(0.0001).rename('NDVI').updateMask(qa)
    evi  = image.select('EVI' ).multiply(0.0001).rename('EVI' ).updateMask(qa)
    return ndvi.addBands(evi).copyProperties(image, ['system:time_start'])


def extract_monthly_vi(collection, geometry, label):
    results = []
    for year in range(2016, 2026):
        for month in range(1, 13):
            t0 = f'{year}-{month:02d}-01'
            ny = year + (1 if month == 12 else 0)
            nm = month % 12 + 1
            t1 = f'{ny}-{nm:02d}-01'

            monthly = collection.filterDate(t0, t1)
            row = {'year': year, 'month': month,
                   'ym': f'{year}-{month:02d}'}

            if monthly.size().getInfo() == 0:
                row.update({'NDVI': None, 'EVI': None})
                results.append(row)
                continue

            stats = (monthly.mean()
                            .reduceRegion(reducer=ee.Reducer.mean(),
                                          geometry=geometry,
                                          scale=250,
                                          maxPixels=1e9)
                            .getInfo())
            row['NDVI'] = stats.get('NDVI')
            row['EVI']  = stats.get('EVI')
            nd = f"{row['NDVI']:.4f}" if row['NDVI'] else 'NaN'
            print(f'  [{label}] {year}-{month:02d}  NDVI={nd}')
            results.append(row)
            time.sleep(0.3)

    return results


def download_ndvi():
    print('\n' + '='*55)
    print('PART C  MODIS NDVI / EVI (250 m)')
    print('='*55)
    col = (ee.ImageCollection('MODIS/061/MOD13Q1')
             .filterDate(START, END)
             .map(process_vi))

    print('\n  Urban core (0–25 km)')
    df_u = (pd.DataFrame(extract_monthly_vi(col, URBAN, 'urban'))
              .rename(columns={'NDVI': 'NDVI_urban', 'EVI': 'EVI_urban'}))

    print('\n  Rural buffer (25–50 km)')
    df_r = (pd.DataFrame(extract_monthly_vi(col, RURAL, 'rural'))
              .rename(columns={'NDVI': 'NDVI_rural', 'EVI': 'EVI_rural'}))

    df_u.to_csv(OUT_DIR / 'ndvi_urban_monthly.csv', index=False)
    df_r.to_csv(OUT_DIR / 'ndvi_rural_monthly.csv', index=False)
    print(f'\n  Saved to {OUT_DIR}')


# ── Main ────────────────────────────────────────────────────
if __name__ == '__main__':
    download_landsat()       # Part A: monthly CSV
    export_monthly_geotiffs() # Part B: GeoTIFFs for ConvLSTM
    download_ndvi()           # Part C: NDVI/EVI CSV

    print('\n✅  Script 01 complete.')
    print('   Check GEE Tasks panel for GeoTIFF export progress.')
    print('   Next: python scripts/02_download_era5.py')
