"""
Chengdu AlphaEarth — Script 03
Purpose : Queue a Google Earth Engine export of ESA WorldCover 2021 for
          Chengdu on the same 64x64 grid used by AlphaEarth and ConvLSTM.

Dataset : ESA/WorldCover/v200  (2021 annual map, 10 m, global)
          Classes:
            10  Tree cover
            20  Shrubland
            30  Grassland
            40  Cropland
            50  Built-up
            60  Bare / sparse vegetation
            80  Permanent water bodies
            90  Herbaceous wetland

Output  : Google Drive folder `chengdu_alphaearth_64grid/`
            worldcover_chengdu_2021.tif   (int8, single band, 64x64)

After download, place the GeoTIFF in:
  AlphaEarth/data/raw/worldcover_chengdu_2021.tif

Then run:
  python AlphaEarth/04_worldcover_classifier.py
"""

from __future__ import annotations

import ee
from pathlib import Path

PROJECT_ID   = "e-psci210-project-louis"
DRIVE_FOLDER = "chengdu_alphaearth_64grid"
GRID_DIMS    = "64x64"
CRS          = "EPSG:4326"

RAW_DIR = Path(__file__).resolve().parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def study_area():
    center = ee.Geometry.Point([104.0665, 30.5728])
    urban  = center.buffer(25_000)
    return urban, urban.bounds()


def main() -> None:
    ee.Initialize(project=PROJECT_ID)
    urban, export_region = study_area()

    wc = (
        ee.ImageCollection("ESA/WorldCover/v200")
        .filterDate("2021-01-01", "2022-01-01")
        .first()
        .select("Map")
        .clip(urban)
        .toByte()
    )

    task = ee.batch.Export.image.toDrive(
        image=wc,
        description="worldcover_chengdu_2021",
        folder=DRIVE_FOLDER,
        fileNamePrefix="worldcover_chengdu_2021",
        region=export_region,
        dimensions=GRID_DIMS,
        crs=CRS,
        maxPixels=1e10,
        fileFormat="GeoTIFF",
    )
    task.start()

    print("=" * 60)
    print("ESA WorldCover 2021 export queued.")
    print("=" * 60)
    print(f"  Task ID      : {task.id}")
    print(f"  Drive folder : {DRIVE_FOLDER}/")
    print(f"  File         : worldcover_chengdu_2021.tif")
    print(f"  Grid         : {GRID_DIMS} over Chengdu 25 km urban buffer")
    print()
    print("Next steps:")
    print("  1. Open https://code.earthengine.google.com/ → Tasks → Run")
    print(f"  2. Download to: {RAW_DIR}/worldcover_chengdu_2021.tif")
    print("  3. Run: python AlphaEarth/04_worldcover_classifier.py")


if __name__ == "__main__":
    main()
