"""
Chengdu Urban Cooling Project — Script 02
Purpose : Download ERA5-Land monthly averaged reanalysis (weather controls)
Variables: 2 m air temperature, total precipitation,
           surface solar radiation downwards
Output  : data/raw/era5/chengdu_ERA5_monthly.csv

Prerequisites:
  pip install cdsapi xarray netCDF4 pandas numpy
  Create ~/.cdsapirc:
      url: https://cds.climate.copernicus.eu/api/v2
      key: <YOUR-UID>:<YOUR-API-KEY>
  Accept dataset terms at:
      https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land-monthly-means
"""

import cdsapi
import xarray as xr
import pandas as pd
from pathlib import Path

OUT_DIR  = Path('data/raw/era5')
OUT_DIR.mkdir(parents=True, exist_ok=True)
NC_FILE  = OUT_DIR / 'chengdu_ERA5_raw.nc'
CSV_FILE = OUT_DIR / 'chengdu_ERA5_monthly.csv'

BBOX = [31.1, 103.4, 30.0, 104.7]   # [N, W, S, E]  50 km envelope


def download_era5():
    if NC_FILE.exists():
        print(f'Found {NC_FILE} — skipping download.')
        return
    print('Connecting to CDS … (10–30 min depending on queue)')
    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-land-monthly-means',
        {
            'product_type': 'monthly_averaged_reanalysis',
            'variable': [
                '2m_temperature',
                'total_precipitation',
                'surface_solar_radiation_downwards',
            ],
            'year' : [str(y) for y in range(2016, 2026)],
            'month': [f'{m:02d}' for m in range(1, 13)],
            'time' : '00:00',
            'area' : BBOX,
            'format': 'netcdf',
        },
        str(NC_FILE)
    )
    print(f'✅  Saved to {NC_FILE}')


def process_era5():
    print('Processing NetCDF …')
    ds  = xr.open_dataset(NC_FILE)
    df  = ds.mean(dim=['latitude', 'longitude']).to_dataframe().reset_index()

    # Rename (variable names differ slightly across ERA5 versions)
    rn = {}
    for c in df.columns:
        if   c in ('t2m',  '2m_temperature'):                     rn[c] = 't2m_K'
        elif c in ('tp',   'total_precipitation'):                 rn[c] = 'tp_m'
        elif c in ('ssrd', 'surface_solar_radiation_downwards'):   rn[c] = 'ssrd_Jm2'
    df = df.rename(columns=rn)

    df['temp_2m_C']     = df['t2m_K']    - 273.15
    df['precip_mm']     = df['tp_m']     * 1000
    df['solar_rad_Wm2'] = df['ssrd_Jm2'] / (30.44 * 24 * 3600)

    df['date']  = pd.to_datetime(df['time'])
    df['year']  = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['ym']    = df['date'].dt.strftime('%Y-%m')

    df = (df[['ym','year','month','temp_2m_C','precip_mm','solar_rad_Wm2']]
            .query('year >= 2016 and year <= 2025')
            .sort_values(['year','month'])
            .reset_index(drop=True))

    assert len(df) == 120, f'Expected 120 rows, got {len(df)}'
    df.to_csv(CSV_FILE, index=False)

    print(f'✅  Saved to {CSV_FILE}  ({len(df)} rows)')
    print(df[['temp_2m_C','precip_mm','solar_rad_Wm2']].describe().round(2))
    return df


if __name__ == '__main__':
    download_era5()
    process_era5()
    print('\n✅  Script 02 complete. Next: python scripts/03_build_panel.py')
