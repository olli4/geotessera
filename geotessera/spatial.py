"""Spatial utilities for working with tiles and geometries.

This module provides spatial helper functions that are commonly needed
when building classifiers with geotessera data. It focuses on practical
utilities while leveraging existing libraries like rasterio and geopandas.
"""

from pathlib import Path
from typing import Union, List, Tuple, Optional, Dict
import tempfile
import shutil
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
import concurrent.futures


def bbox_from_points(
    points: Union[List[Dict], pd.DataFrame], buffer_meters: float = 0
) -> Tuple[float, float, float, float]:
    """Calculate bounding box from points with optional buffer.

    Creates a bounding box that encompasses all points, with an optional
    buffer distance. The buffer is applied in a suitable projected coordinate
    system for accuracy.

    Args:
        points: List of dicts or DataFrame with 'lat'/'lon' columns
        buffer_meters: Buffer distance in meters to add around the bounding box

    Returns:
        Tuple of (west, south, east, north) in EPSG:4326

    Examples:
        >>> points = [{'lat': 52.2, 'lon': 0.1}, {'lat': 52.3, 'lon': 0.2}]
        >>> bbox = bbox_from_points(points, buffer_meters=1000)
        >>> print(bbox)
        (0.09, 52.19, 0.21, 52.31)  # Approximate values
    """
    # Convert to DataFrame if needed
    if isinstance(points, list):
        df = pd.DataFrame(points)
    else:
        df = points

    if "lat" not in df.columns or "lon" not in df.columns:
        raise ValueError("Points must have 'lat' and 'lon' columns")

    # Get initial bounds
    west = df["lon"].min()
    east = df["lon"].max()
    south = df["lat"].min()
    north = df["lat"].max()

    # If no buffer requested, return WGS84 bounds
    if buffer_meters == 0:
        return (west, south, east, north)

    # For buffering, we need to project to a metric CRS
    # Create a GeoDataFrame
    from shapely.geometry import box

    # Use the centroid to determine appropriate UTM zone
    center_lon = (west + east) / 2
    center_lat = (south + north) / 2
    utm_crs = _get_utm_crs(center_lon, center_lat)

    # Create bounding box geometry
    bbox_geom = box(west, south, east, north)
    gdf = gpd.GeoDataFrame([1], geometry=[bbox_geom], crs="EPSG:4326")

    # Project to UTM, buffer, and project back
    gdf_utm = gdf.to_crs(utm_crs)
    gdf_buffered = gdf_utm.buffer(buffer_meters)
    gdf_buffered_wgs = gdf_buffered.to_crs("EPSG:4326")

    # Extract new bounds
    bounds = gdf_buffered_wgs.total_bounds
    return tuple(bounds)


def create_grid(
    bounds: Tuple[float, float, float, float], spacing_degrees: float = 0.1
) -> List[Tuple[float, float]]:
    """Create a grid of tile coordinates within bounds.

    Generates tile coordinates (lower-left corners) for a regular grid
    that covers the specified bounding box.

    Args:
        bounds: (west, south, east, north) in EPSG:4326
        spacing_degrees: Grid spacing in degrees (default 0.1 for tessera tiles)

    Returns:
        List of (lat, lon) tuples for tile lower-left corners

    Examples:
        >>> bounds = (0.0, 52.0, 0.3, 52.2)
        >>> tiles = create_grid(bounds)
        >>> print(len(tiles))
        6  # 3x2 grid
    """
    west, south, east, north = bounds

    # Align to grid
    start_lon = np.floor(west / spacing_degrees) * spacing_degrees
    start_lat = np.floor(south / spacing_degrees) * spacing_degrees
    end_lon = np.ceil(east / spacing_degrees) * spacing_degrees
    end_lat = np.ceil(north / spacing_degrees) * spacing_degrees

    # Generate grid
    tiles = []
    current_lat = start_lat
    while current_lat < end_lat:
        current_lon = start_lon
        while current_lon < end_lon:
            tiles.append((current_lat, current_lon))
            current_lon += spacing_degrees
        current_lat += spacing_degrees

    return tiles


def stitch_rasters(
    raster_paths: List[Union[str, Path]],
    output_path: Union[str, Path],
    resampling: str = "nearest",
    max_workers: int = 20,
) -> str:
    """Merge multiple georeferenced rasters into one.

    Efficiently stitches multiple raster files, handling different CRS
    automatically. Uses parallel processing for reprojection when needed.

    Args:
        raster_paths: List of paths to raster files
        output_path: Path for output merged raster
        resampling: Resampling method ('nearest', 'bilinear', 'cubic')
        max_workers: Maximum workers for parallel reprojection

    Returns:
        Path to the created output file

    Examples:
        >>> tiles = ['tile1.tif', 'tile2.tif', 'tile3.tif']
        >>> output = stitch_rasters(tiles, 'merged.tif')
        >>> print(f"Created: {output}")
    """
    if not raster_paths:
        raise ValueError("No raster files provided to stitch")

    raster_paths = [Path(p) for p in raster_paths]
    output_path = Path(output_path)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if reprojection is needed
    with rasterio.open(raster_paths[0]) as first_src:
        target_crs = first_src.crs

    # Check if all files have the same CRS
    needs_reprojection = False
    for path in raster_paths[1:]:
        with rasterio.open(path) as src:
            if src.crs != target_crs:
                needs_reprojection = True
                break

    if needs_reprojection:
        # Reproject all files to common CRS
        temp_dir = tempfile.mkdtemp(prefix="geotessera_stitch_")
        try:
            reprojected_paths = _reproject_rasters_parallel(
                raster_paths, temp_dir, target_crs, resampling, max_workers
            )
            paths_to_merge = reprojected_paths
        except Exception as e:
            shutil.rmtree(temp_dir)
            raise e
    else:
        paths_to_merge = raster_paths
        temp_dir = None

    # Merge the rasters
    try:
        src_files = [rasterio.open(p) for p in paths_to_merge]
        mosaic, out_trans = merge(src_files)

        # Get metadata from first file
        out_meta = src_files[0].meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans,
                "crs": target_crs,
                "compress": "lzw",
            }
        )

        # Write merged result
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(mosaic)

        return str(output_path)

    finally:
        # Clean up
        for src in src_files:
            src.close()
        if temp_dir:
            shutil.rmtree(temp_dir)


def _get_utm_crs(lon: float, lat: float) -> str:
    """Calculate the appropriate UTM Zone CRS for a given lat/lon.

    Args:
        lon: Longitude in degrees
        lat: Latitude in degrees

    Returns:
        EPSG code for the UTM zone (e.g., 'EPSG:32630')
    """
    utm_band = str((int((lon + 180) / 6) % 60) + 1)
    if len(utm_band) == 1:
        utm_band = "0" + utm_band
    if lat >= 0:
        return "EPSG:326" + utm_band
    else:
        return "EPSG:327" + utm_band


def _reproject_rasters_parallel(
    raster_paths: List[Path],
    output_dir: Path,
    target_crs: str,
    resampling: str,
    max_workers: int,
) -> List[Path]:
    """Reproject multiple rasters in parallel.

    Internal function used by stitch_rasters.
    """
    # Map resampling string to rasterio enum
    resampling_map = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
    }
    resampling_enum = resampling_map.get(resampling, Resampling.nearest)

    # Prepare tasks
    tasks = [(path, output_dir, target_crs, resampling_enum) for path in raster_paths]

    # Process in parallel
    reprojected_paths = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_reproject_single_raster, *task) for task in tasks]

        for future in concurrent.futures.as_completed(futures):
            try:
                result_path = future.result()
                if result_path:
                    reprojected_paths.append(result_path)
            except Exception as e:
                print(f"Warning: Failed to reproject a raster: {e}")

    if not reprojected_paths:
        raise RuntimeError("All reprojection tasks failed")

    return reprojected_paths


def _reproject_single_raster(
    input_path: Path, output_dir: Path, target_crs: str, resampling: Resampling
) -> Optional[Path]:
    """Reproject a single raster file.

    Worker function for parallel reprojection.
    """
    try:
        with rasterio.open(input_path) as src:
            # If already in target CRS, just copy
            if src.crs == target_crs:
                output_path = output_dir / input_path.name
                shutil.copy(input_path, output_path)
                return output_path

            # Calculate transform for target CRS
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds
            )

            # Update metadata
            kwargs = src.meta.copy()
            kwargs.update(
                {
                    "crs": target_crs,
                    "transform": transform,
                    "width": width,
                    "height": height,
                }
            )

            # Reproject
            output_path = output_dir / input_path.name
            with rasterio.open(output_path, "w", **kwargs) as dst:
                for band_idx in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, band_idx),
                        destination=rasterio.band(dst, band_idx),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=target_crs,
                        resampling=resampling,
                    )

            return output_path

    except Exception as e:
        print(f"Failed to reproject {input_path.name}: {e}")
        return None


def roi_from_points(
    points: Union[List[Dict], pd.DataFrame], buffer_km: float = 1.0
) -> gpd.GeoDataFrame:
    """Create a buffered bounding box GeoDataFrame around a list of points.

    This is a convenience function that creates a region of interest (ROI)
    as a GeoDataFrame, useful for passing to find_tiles_for_geometry.

    Args:
        points: List of dicts or DataFrame with 'lat'/'lon' columns
        buffer_km: Buffer distance in kilometers around the bounding box

    Returns:
        GeoDataFrame containing a single polygon feature in EPSG:4326

    Examples:
        >>> points = [{'lat': 52.2, 'lon': 0.1, 'label': 'Urban'}]
        >>> roi = roi_from_points(points, buffer_km=5)
        >>> print(roi.geometry[0].bounds)
    """
    from shapely.geometry import box

    # Get buffered bounds
    bounds = bbox_from_points(points, buffer_meters=buffer_km * 1000)

    # Create polygon
    roi_polygon = box(*bounds)

    # Create GeoDataFrame
    roi_gdf = gpd.GeoDataFrame(
        [{"buffer_km": buffer_km}], geometry=[roi_polygon], crs="EPSG:4326"
    )

    return roi_gdf
