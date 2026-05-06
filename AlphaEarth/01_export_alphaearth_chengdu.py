"""
Chengdu AlphaEarth — Script 01
Purpose : Queue Google Earth Engine exports for annual AlphaEarth embeddings
          over Chengdu on the same 64x64 grid used by the ConvLSTM analysis.

Why this script exists:
  - The draft paper shifts the Chengdu project toward AlphaEarth change
    detection.
  - The existing project already has a 64x64 ConvLSTM cooling residual map.
  - Exporting AlphaEarth on that same grid makes the alignment analysis
    straightforward and keeps file sizes tractable.

Dataset:
  GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL
  MODIS/061/MOD13Q1  (optional annual NDVI export)

Outputs:
  Google Drive folder `chengdu_alphaearth_64grid/`
    alphaearth_chengdu_2017.tif
    alphaearth_chengdu_2018.tif
    ...
  Optional:
    alphaearth_chengdu_mask.tif
    ndvi_annual_chengdu_2017.tif
    ndvi_annual_chengdu_2018.tif
    ...

Usage:
  /Users/louis/Desktop/E-PSCI210/final_project/Version2/venv311/bin/python \
      AlphaEarth/01_export_alphaearth_chengdu.py

Notes:
  - This script only queues Earth Engine tasks. You still need to open the
    Earth Engine Tasks panel and run them.
  - After download, place the GeoTIFFs in AlphaEarth/data/raw/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import ee


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

PROJECT_ID = "e-psci210-project-louis"
DATASET_ID = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
NDVI_DATASET_ID = "MODIS/061/MOD13Q1"
DRIVE_FOLDER = "chengdu_alphaearth_64grid"
GRID_DIMS = "64x64"
CRS = "EPSG:4326"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Queue Chengdu AlphaEarth annual embedding exports."
    )
    parser.add_argument("--start-year", type=int, default=2017)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument(
        "--drive-folder",
        default=DRIVE_FOLDER,
        help="Google Drive folder for the export tasks.",
    )
    parser.add_argument(
        "--skip-mask",
        action="store_true",
        help="Do not queue the one-time Chengdu urban mask export.",
    )
    parser.add_argument(
        "--with-ndvi",
        action="store_true",
        help="Also queue annual MODIS NDVI exports on the same 64x64 grid.",
    )
    parser.add_argument(
        "--ndvi-only",
        action="store_true",
        help="Queue only annual MODIS NDVI exports and skip AlphaEarth embedding exports.",
    )
    return parser.parse_args()


def init_ee() -> None:
    ee.Initialize(project=PROJECT_ID)


def study_area() -> tuple[ee.geometry.Geometry, ee.geometry.Geometry]:
    center = ee.Geometry.Point([104.0665, 30.5728])
    urban = center.buffer(25_000)
    export_region = urban.bounds()
    return urban, export_region


def collection_for_year(
    year: int,
    urban: ee.geometry.Geometry,
) -> ee.imagecollection.ImageCollection:
    start = f"{year}-01-01"
    end = f"{year + 1}-01-01"
    return (
        ee.ImageCollection(DATASET_ID)
        .filterDate(start, end)
        .filterBounds(urban)
        .sort("system:time_start")
    )


def queue_mask_export(
    folder: str,
    urban: ee.geometry.Geometry,
    export_region: ee.geometry.Geometry,
) -> None:
    mask_img = ee.Image.constant(1).clip(urban).rename("urban_mask").toByte()
    task = ee.batch.Export.image.toDrive(
        image=mask_img,
        description="alphaearth_chengdu_mask",
        folder=folder,
        fileNamePrefix="alphaearth_chengdu_mask",
        region=export_region,
        dimensions=GRID_DIMS,
        crs=CRS,
        maxPixels=1e10,
        fileFormat="GeoTIFF",
    )
    task.start()
    print("Queued: alphaearth_chengdu_mask")


def queue_year_export(
    year: int,
    folder: str,
    urban: ee.geometry.Geometry,
    export_region: ee.geometry.Geometry,
) -> bool:
    col = collection_for_year(year, urban)
    size = col.size().getInfo()
    if size == 0:
        print(f"Skipped {year}: no AlphaEarth image found.")
        return False

    image = ee.Image(col.first()).clip(urban).toFloat()
    description = f"alphaearth_chengdu_{year}"
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        fileNamePrefix=description,
        region=export_region,
        dimensions=GRID_DIMS,
        crs=CRS,
        maxPixels=1e10,
        fileFormat="GeoTIFF",
    )
    task.start()
    band_names = image.bandNames().getInfo()
    print(
        f"Queued {year}: {description} "
        f"({len(band_names)} bands, {size} image(s) in collection)"
    )
    return True


def queue_ndvi_export(
    year: int,
    folder: str,
    urban: ee.geometry.Geometry,
    export_region: ee.geometry.Geometry,
) -> bool:
    start = f"{year}-01-01"
    end = f"{year + 1}-01-01"
    col = (
        ee.ImageCollection(NDVI_DATASET_ID)
        .filterDate(start, end)
        .filterBounds(urban)
        .select("NDVI")
    )
    size = col.size().getInfo()
    if size == 0:
        print(f"Skipped {year}: no NDVI image found.")
        return False

    image = col.mean().multiply(0.0001).rename("NDVI").clip(urban).toFloat()
    description = f"ndvi_annual_chengdu_{year}"
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        fileNamePrefix=description,
        region=export_region,
        dimensions=GRID_DIMS,
        crs=CRS,
        maxPixels=1e10,
        fileFormat="GeoTIFF",
    )
    task.start()
    print(f"Queued {year}: {description} ({size} image(s) in collection)")
    return True


def main() -> None:
    args = parse_args()
    init_ee()
    urban, export_region = study_area()

    print("=" * 60)
    print("Chengdu AlphaEarth export tasks")
    print("=" * 60)
    print(f"Dataset      : {DATASET_ID}")
    print(f"Grid         : {GRID_DIMS} over Chengdu 25 km urban buffer")
    print(f"Drive folder : {args.drive_folder}")
    print(f"Years        : {args.start_year}–{args.end_year}")
    print(f"With NDVI    : {args.with_ndvi}")
    print(f"NDVI only    : {args.ndvi_only}")

    if not args.skip_mask:
        queue_mask_export(args.drive_folder, urban, export_region)

    queued = 0
    for year in range(args.start_year, args.end_year + 1):
        if not args.ndvi_only:
            queued += int(
                queue_year_export(year, args.drive_folder, urban, export_region)
            )
        if args.with_ndvi or args.ndvi_only:
            queued += int(
                queue_ndvi_export(year, args.drive_folder, urban, export_region)
            )

    print("\nNext steps:")
    print("1. Open https://code.earthengine.google.com/ and run the queued tasks.")
    print(
        "2. Download the GeoTIFFs into "
        f"{RAW_DIR}"
    )
    print(
        "3. Run AlphaEarth/02_analyze_chengdu_alphaearth.py to compare "
        "embedding change with ConvLSTM cooling residuals, delta LST, and delta NDVI."
    )
    print(f"\nQueued {queued} export task(s).")


if __name__ == "__main__":
    main()
