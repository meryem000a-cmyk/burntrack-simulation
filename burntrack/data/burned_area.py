"""
burned_area.py — Build a 30 m fused burned-area product from Landsat 8/9
======================================================================

Pipeline:
    1. Discover pre-fire and post-fire Landsat 8/9 scenes in a directory.
    2. For each scene, read SWIR22, NIR08, and QA_PIXEL bands.
    3. Convert DN to surface reflectance (Collection 2 scaling: 0.0000275 * DN - 0.2).
    4. Apply the cloud / snow / water QA mask.
    5. Compute NBR = (NIR - SWIR) / (NIR + SWIR) per scene.
    6. Median-composite the pre-fire NBR and the post-fire NBR.
    7. dNBR = pre_NBR - post_NBR.
    8. Threshold dNBR at the supplied cutoff (default 0.10, sens range 0.10-0.27).
    9. Reproject + resample the binary raster onto the simulation grid
       (default: 40x40 cells of 100 m centred on a bbox).
   10. Save the 30 m product and the 100 m sim-grid product as GeoTIFFs.

CLI:
    python -m burntrack.data.burned_area \
        --landsat-dir south africa data/table_mountain_2021/landsat \
        --bbox 18.418 -33.962 18.470 -33.933 \
        --grid-rows 40 --grid-cols 40 --cell-size 100 \
        --threshold 0.10 \
        --out south africa data/table_mountain_2021/burned_area_30m.tif \
        --out-grid south africa data/table_mountain_2021/burned_area_40x40.tif
"""

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject

# Landsat Collection 2 Level-2 surface reflectance scaling
LANDSAT_SCALE = 0.0000275
LANDSAT_OFFSET = -0.2


@dataclass
class LandsatScene:
    path: Path
    item_id: str
    date: str
    cloud_cover: float
    swir22: np.ndarray          # (H, W) reflectance
    nir08: np.ndarray           # (H, W) reflectance
    qa_pixel: np.ndarray        # (H, W) uint16
    transform: object
    crs: object
    nodata: float

    def nbr(self) -> np.ndarray:
        """NBR = (NIR - SWIR) / (NIR + SWIR). NaN where denominator is 0 or pixel is masked."""
        good = self._good_mask()
        nir = self.nir08
        swir = self.swir22
        denom = nir + swir
        out = np.where(denom > 0, (nir - swir) / denom, np.nan).astype(np.float32)
        out[~good] = np.nan
        return out

    def _good_mask(self) -> np.ndarray:
        """QA_PIXEL bit 0 = fill, bit 1 = dilated cloud, bit 2 = cirrus,
        bit 3 = cloud, bit 4 = cloud shadow, bit 5 = snow, bit 6 = clear.
        We want clear (bit 6) AND not water (bit 7).
        Conservative: reject fill/cloud/shadow/snow, allow water/clear.
        """
        qa = self.qa_pixel
        # bit 0 (fill), bit 3 (cloud), bit 4 (cloud shadow), bit 5 (snow)
        bad = (qa & (1 << 0)) | (qa & (1 << 3)) | (qa & (1 << 4)) | (qa & (1 << 5))
        return bad == 0


def _scene_id_and_band(path: Path) -> Optional[Tuple[str, str]]:
    """Return (scene_id, band) parsed from the filename, e.g.
    'LC08_L2SP_175083_20210327_02_T1_qa_pixel.tif' -> ('LC08_L2SP_175083_20210327_02_T1', 'qa_pixel')
    """
    m = re.match(
        r"((?:LC08|LC09)_L2SP_\d{6}_\d{8}_02_T1)_(swir22|nir08|qa_pixel)\.tif$",
        path.name,
    )
    if not m:
        return None
    return m.group(1), m.group(2)


def _scene_id(path: Path) -> Optional[str]:
    parsed = _scene_id_and_band(path)
    return parsed[0] if parsed else None


def _read_band(path: Path) -> Tuple[np.ndarray, object, object, float]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        # Convert to reflectance
        arr = arr * LANDSAT_SCALE + LANDSAT_OFFSET
        arr = np.where(arr < 0, 0, arr)
        return arr, src.transform, src.crs, src.nodata if src.nodata is not None else -9999.0


def discover_scenes(
    landsat_dir: Path,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> Tuple[List[LandsatScene], List[LandsatScene]]:
    """Return (pre_scenes, post_scenes) from the standard layout.
    If bbox is given, each scene is reprojected onto a common 30 m grid covering
    bbox (with a small buffer so adjacent scenes overlap).
    """
    pre_dir = landsat_dir / "pre"
    post_dir = landsat_dir / "post"
    return _load_dir(pre_dir, bbox), _load_dir(post_dir, bbox)


def _read_band_to_grid(path: Path, dst_transform, dst_crs, dst_shape, resampling=Resampling.bilinear):
    """Read a band and reproject to the (dst_shape, dst_transform, dst_crs) grid."""
    with rasterio.open(path) as src:
        if src.count == 0:
            return None
        src_arr = src.read(1).astype(np.float32)
        # Convert DN to reflectance before reprojection
        src_arr = src_arr * LANDSAT_SCALE + LANDSAT_OFFSET
        src_arr = np.where(src_arr < 0, 0, src_arr)
        dst = np.zeros(dst_shape, dtype=np.float32)
        reproject(
            source=src_arr,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=resampling,
        )
    return dst


def _read_qa_to_grid(path: Path, dst_transform, dst_crs, dst_shape):
    return _read_band_to_grid(path, dst_transform, dst_crs, dst_shape, resampling=Resampling.nearest)


def _common_grid(bbox: Tuple[float, float, float, float], buffer_m: float = 2000.0):
    """Build a 30 m grid covering bbox + buffer, in EPSG:4326 (geographic)."""
    min_lon, min_lat, max_lon, max_lat = bbox
    # Buffer: 2 km in degrees, scaled by latitude
    dlon = buffer_m / (111320.0 * np.cos(np.radians(0.5 * (min_lat + max_lat))))
    dlat = buffer_m / 111320.0
    bbox_b = (min_lon - dlon, min_lat - dlat, max_lon + dlon, max_lat + dlat)
    # Approx 30 m grid in geographic degrees
    res_lon = 30.0 / (111320.0 * np.cos(np.radians(0.5 * (bbox_b[1] + bbox_b[3]))))
    res_lat = 30.0 / 111320.0
    width = max(1, int(round((bbox_b[2] - bbox_b[0]) / res_lon)))
    height = max(1, int(round((bbox_b[3] - bbox_b[1]) / res_lat)))
    transform = from_bounds(*bbox_b, width, height)
    return transform, rasterio.crs.CRS.from_epsg(4326), (height, width)


def _load_dir(d: Path, bbox: Optional[Tuple[float, float, float, float]]) -> List[LandsatScene]:
    if bbox is None:
        # Read scenes as-is (no common grid). All scenes must have the same shape.
        return _load_dir_native(d)
    return _load_dir_on_common_grid(d, bbox)


def _load_dir_native(d: Path) -> List[LandsatScene]:
    scenes = {}
    for p in sorted(d.glob("LC0*_L2SP_*_02_T1_*.tif")):
        parsed = _scene_id_and_band(p)
        if not parsed:
            continue
        sid, band = parsed
        s = scenes.setdefault(sid, {"path": d, "item_id": sid, "parts": {}})
        s["parts"][band] = p

    out = []
    for sid, s in scenes.items():
        if not {"swir22", "nir08", "qa_pixel"}.issubset(s["parts"].keys()):
            print(f"  ! {sid}: missing bands, skipping")
            continue
        swir, t, c, nd = _read_band(s["parts"]["swir22"])
        nir, _, _, _ = _read_band(s["parts"]["nir08"])
        qa, _, _, _ = _read_band(s["parts"]["qa_pixel"])
        date_m = re.search(r"_(\d{8})_", sid)
        date = date_m.group(1) if date_m else "unknown"
        out.append(
            LandsatScene(
                path=s["path"],
                item_id=sid,
                date=date,
                cloud_cover=-1.0,
                swir22=swir,
                nir08=nir,
                qa_pixel=qa.astype(np.uint16),
                transform=t,
                crs=c,
                nodata=nd,
            )
        )
    return out


def _load_dir_on_common_grid(d: Path, bbox: Tuple[float, float, float, float]) -> List[LandsatScene]:
    """Reproject every band to a 30 m common grid covering bbox+buffer."""
    transform, crs, shape = _common_grid(bbox, buffer_m=2000.0)
    print(f"  Common grid: {shape[1]}x{shape[0]} @ ~30 m covering {bbox}")

    file_groups = {}
    for p in sorted(d.glob("LC0*_L2SP_*_02_T1_*.tif")):
        parsed = _scene_id_and_band(p)
        if not parsed:
            continue
        sid, band = parsed
        s = file_groups.setdefault(sid, {"path": d, "item_id": sid, "parts": {}})
        s["parts"][band] = p

    out = []
    for sid, s in file_groups.items():
        if not {"swir22", "nir08", "qa_pixel"}.issubset(s["parts"].keys()):
            print(f"  ! {sid}: missing bands, skipping")
            continue
        # Reproject all 3 bands to the common grid
        swir = _read_band_to_grid(s["parts"]["swir22"], transform, crs, shape)
        nir = _read_band_to_grid(s["parts"]["nir08"], transform, crs, shape)
        qa = _read_qa_to_grid(s["parts"]["qa_pixel"], transform, crs, shape)
        if swir is None or nir is None or qa is None:
            print(f"  ! {sid}: read failed, skipping")
            continue
        date_m = re.search(r"_(\d{8})_", sid)
        date = date_m.group(1) if date_m else "unknown"
        out.append(
            LandsatScene(
                path=s["path"],
                item_id=sid,
                date=date,
                cloud_cover=-1.0,
                swir22=swir,
                nir08=nir,
                qa_pixel=qa.astype(np.uint16),
                transform=transform,
                crs=crs,
                nodata=0.0,
            )
        )
        print(f"  ✓ {sid} date={date} shape={swir.shape}")
    return out
    out = []
    for sid, s in scenes.items():
        if not {"swir22", "nir08", "qa_pixel"}.issubset(s["parts"].keys()):
            print(f"  ! {sid}: missing bands, skipping")
            continue
        swir, t, c, nd = _read_band(s["parts"]["swir22"])
        nir, _, _, _ = _read_band(s["parts"]["nir08"])
        qa, _, _, _ = _read_band(s["parts"]["qa_pixel"])
        # Date from item id
        date_m = re.search(r"_(\d{8})_", sid)
        date = date_m.group(1) if date_m else "unknown"
        out.append(
            LandsatScene(
                path=s["path"],
                item_id=sid,
                date=date,
                cloud_cover=-1.0,  # not parsed here
                swir22=swir,
                nir08=nir,
                qa_pixel=qa.astype(np.uint16),
                transform=t,
                crs=c,
                nodata=nd,
            )
        )
    return out


def composite_nbr(scenes: List[LandsatScene]) -> Tuple[np.ndarray, object, object]:
    """Per-pixel median of NBR across all scenes. NaN where no scene contributed."""
    if not scenes:
        raise ValueError("no scenes to composite")
    nbrs = [s.nbr() for s in scenes]
    stack = np.stack(nbrs, axis=0)  # (N, H, W)
    with np.errstate(invalid="ignore"):
        comp = np.nanmedian(stack, axis=0)
    # Use first scene's transform/crs
    return comp, scenes[0].transform, scenes[0].crs


def make_dnbr(pre: np.ndarray, post: np.ndarray) -> np.ndarray:
    return (pre - post).astype(np.float32)


def threshold(dnbr: np.ndarray, cutoff: float) -> np.ndarray:
    """Return a uint8 binary mask (1 = burned, 0 = not)."""
    return (dnbr >= cutoff).astype(np.uint8)


def reproject_to_grid(
    src: np.ndarray,
    src_transform,
    src_crs,
    bbox: Tuple[float, float, float, float],  # min_lon, min_lat, max_lon, max_lat
    grid_rows: int,
    grid_cols: int,
    resampling: Resampling = Resampling.bilinear,
) -> Tuple[np.ndarray, object]:
    """Reproject/resample `src` to a (grid_rows, grid_cols) raster covering `bbox`.
    bbox is in lon/lat (EPSG:4326).
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    dst_transform = from_bounds(min_lon, min_lat, max_lon, max_lat, grid_cols, grid_rows)
    dst_crs = rasterio.crs.CRS.from_epsg(4326)
    dst = np.zeros((grid_rows, grid_cols), dtype=src.dtype)
    reproject(
        source=src,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=resampling,
    )
    return dst, dst_transform


def save_geotiff(path: Path, arr: np.ndarray, transform, crs, dtype=None, nodata=None):
    if dtype is None:
        dtype = arr.dtype
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff",
        height=arr.shape[0], width=arr.shape[1], count=1,
        dtype=dtype, crs=crs, transform=transform, nodata=nodata,
    ) as dst:
        dst.write(arr, 1)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--landsat-dir", required=True, type=Path)
    ap.add_argument("--bbox", nargs=4, type=float, required=True,
                    metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    ap.add_argument("--grid-rows", type=int, default=40)
    ap.add_argument("--grid-cols", type=int, default=40)
    ap.add_argument("--threshold", type=float, default=0.10)
    ap.add_argument("--out", type=Path, required=True,
                    help="Path to save the 30 m binary burned-area GeoTIFF")
    ap.add_argument("--out-grid", type=Path, default=None,
                    help="Optional: path to save the resampled sim-grid (40x40) GeoTIFF")
    ap.add_argument("--save-dnbr", type=Path, default=None,
                    help="Optional: path to save the dNBR GeoTIFF (30 m)")
    args = ap.parse_args()

    print(f"=== Building 30 m burned-area product for {args.landsat_dir} ===")
    bbox = tuple(args.bbox)

    pre, post = discover_scenes(args.landsat_dir, bbox=bbox)
    print(f"  Pre-fire scenes:  {len(pre)}")
    print(f"  Post-fire scenes: {len(post)}")
    if not pre or not post:
        raise SystemExit("Need at least one pre and one post scene")

    print("  Compositing pre-fire NBR (median)...")
    pre_nbr, t, c = composite_nbr(pre)
    print("  Compositing post-fire NBR (median)...")
    post_nbr, _, _ = composite_nbr(post)
    dnbr = make_dnbr(pre_nbr, post_nbr)
    print(f"  dNBR range: [{dnbr.min():.3f}, {dnbr.max():.3f}]")

    if args.save_dnbr:
        save_geotiff(args.save_dnbr, dnbr, t, c, dtype="float32", nodata=np.nan)
        print(f"  ✓ dNBR (30 m) -> {args.save_dnbr}")

    burned_30m = threshold(dnbr, args.threshold)
    print(f"  Burned pixels @ threshold={args.threshold}: {int(burned_30m.sum())} "
          f"({100 * burned_30m.mean():.1f} % of AOI)")
    save_geotiff(args.out, burned_30m, t, c, dtype="uint8", nodata=0)
    print(f"  ✓ 30 m burned-area -> {args.out}")

    if args.out_grid:
        # Resample the 30 m product onto the sim grid; for a binary mask, nearest is honest
        grid, grid_t = reproject_to_grid(
            burned_30m, t, c, bbox, args.grid_rows, args.grid_cols,
            resampling=Resampling.nearest,
        )
        # If a sim cell is partially burned, threshold at >= 10 % sub-pixel burned
        # We have to compute the fractional burn at sim-grid resolution
        grid_frac, _ = reproject_to_grid(
            burned_30m.astype(np.float32), t, c, bbox, args.grid_rows, args.grid_cols,
            resampling=Resampling.average,
        )
        binary_grid = (grid_frac >= 0.10).astype(np.uint8)
        save_geotiff(args.out_grid, binary_grid, grid_t,
                     rasterio.crs.CRS.from_epsg(4326), dtype="uint8", nodata=0)
        print(f"  ✓ Sim-grid {args.grid_rows}x{args.grid_cols} burned-area -> {args.out_grid}")
        print(f"    Burned sim cells: {int(binary_grid.sum())} / {binary_grid.size} "
              f"({100 * binary_grid.mean():.1f} %)")


if __name__ == "__main__":
    main()
