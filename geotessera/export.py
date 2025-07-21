"""Export utilities for saving embeddings and classification results.

This module provides functions to export processed data in various
georeferenced formats, with proper metadata and visualization support.
"""

from pathlib import Path
from typing import Union, Optional, List, Dict, Tuple
import numpy as np
import rasterio
from rasterio.enums import ColorInterp
import json
import pandas as pd


def embeddings_to_geotiff(
    embedding: np.ndarray,
    lat: float,
    lon: float,
    output_path: Union[str, Path],
    gt,
    bands: Optional[List[int]] = None,
    normalize: bool = True,
    descriptions: Optional[List[str]] = None,
) -> str:
    """Export embedding (or subset) as georeferenced GeoTIFF.

    Saves embeddings with proper georeferencing, optionally selecting
    specific bands and normalizing values for visualization.

    Args:
        embedding: Embedding array (H, W, C)
        lat: Tile latitude
        lon: Tile longitude
        output_path: Path for output GeoTIFF
        gt: GeoTessera instance for georeferencing info
        bands: List of band indices to export. If None, exports all 128 bands.
        normalize: If True, normalizes each band to 0-255 range
        descriptions: Optional band descriptions for metadata

    Returns:
        Path to created file

    Examples:
        >>> # Export full embedding
        >>> embeddings_to_geotiff(embedding, 52.2, 0.1, "full_embedding.tif", gt)

        >>> # Export RGB visualization
        >>> embeddings_to_geotiff(
        ...     embedding, 52.2, 0.1, "rgb.tif", gt,
        ...     bands=[10, 20, 30],
        ...     descriptions=["Red", "Green", "Blue"]
        ... )
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get georeferencing info
    crs = gt.get_tile_crs(lat, lon)
    transform = gt.get_tile_transform(lat, lon)

    # Select bands
    if bands is not None:
        data = embedding[:, :, bands].copy()
    else:
        data = embedding.copy()

    # Ensure correct shape (bands, height, width)
    if data.ndim == 2:
        data = data[np.newaxis, :, :]
    elif data.ndim == 3 and data.shape[2] < data.shape[0]:
        data = np.transpose(data, (2, 0, 1))

    n_bands = data.shape[0]

    # Normalize if requested
    if normalize:
        normalized_data = np.zeros_like(data, dtype=np.uint8)
        for i in range(n_bands):
            band = data[i]
            min_val, max_val = band.min(), band.max()
            if max_val > min_val:
                normalized = (band - min_val) / (max_val - min_val) * 255
                normalized_data[i] = normalized.astype(np.uint8)
            else:
                normalized_data[i] = 0
        data = normalized_data
        dtype = "uint8"
    else:
        dtype = data.dtype

    # Set color interpretation for RGB
    if n_bands == 3:
        colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue]
    elif n_bands == 4:
        colorinterp = [
            ColorInterp.red,
            ColorInterp.green,
            ColorInterp.blue,
            ColorInterp.alpha,
        ]
    else:
        colorinterp = [ColorInterp.gray] * n_bands

    # Write GeoTIFF
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=n_bands,
        dtype=dtype,
        crs=crs,
        transform=transform,
        compress="lzw",
    ) as dst:
        dst.write(data)

        # Set color interpretation
        dst.colorinterp = colorinterp

        # Add band descriptions if provided
        if descriptions:
            for i, desc in enumerate(descriptions[:n_bands]):
                dst.set_band_description(i + 1, desc)
        elif bands:
            # Add default descriptions for selected bands
            for i, band_idx in enumerate(bands[:n_bands]):
                dst.set_band_description(i + 1, f"Band {band_idx}")

    return str(output_path)


def classification_to_geotiff(
    classification: np.ndarray,
    lat: float,
    lon: float,
    output_path: Union[str, Path],
    gt,
    class_names: Optional[Dict[int, str]] = None,
    class_colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
    nodata_value: Optional[int] = None,
) -> str:
    """Export classification results as georeferenced GeoTIFF with metadata.

    Creates a single-band GeoTIFF with optional color table and class names
    stored in metadata. Compatible with QGIS and other GIS software.

    Args:
        classification: 2D array of class indices
        lat: Tile latitude
        lon: Tile longitude
        output_path: Path for output GeoTIFF
        gt: GeoTessera instance for georeferencing info
        class_names: Mapping from class index to name
        class_colors: Mapping from class index to RGB tuple (0-255)
        nodata_value: Value to use for no-data pixels

    Returns:
        Path to created file

    Examples:
        >>> # Basic classification export
        >>> classification_to_geotiff(
        ...     predictions, 52.2, 0.1, "classified.tif", gt
        ... )

        >>> # With class names and colors
        >>> class_names = {0: "Water", 1: "Forest", 2: "Urban"}
        >>> class_colors = {0: (0, 0, 255), 1: (0, 255, 0), 2: (255, 0, 0)}
        >>> classification_to_geotiff(
        ...     predictions, 52.2, 0.1, "classified.tif", gt,
        ...     class_names=class_names,
        ...     class_colors=class_colors
        ... )
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get georeferencing info
    crs = gt.get_tile_crs(lat, lon)
    transform = gt.get_tile_transform(lat, lon)

    # Ensure 2D array
    if classification.ndim != 2:
        raise ValueError("Classification must be a 2D array")

    # Determine dtype based on number of classes
    unique_classes = np.unique(classification[~np.isnan(classification)])
    max_class = int(unique_classes.max()) if len(unique_classes) > 0 else 0

    if max_class <= 255:
        dtype = "uint8"
        classification = classification.astype(np.uint8)
    else:
        dtype = "uint16"
        classification = classification.astype(np.uint16)

    # Prepare metadata
    metadata = {}
    if class_names:
        metadata["classes"] = json.dumps(class_names)

    # Write GeoTIFF
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=classification.shape[0],
        width=classification.shape[1],
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform,
        compress="lzw",
        nodata=nodata_value,
    ) as dst:
        dst.write(classification, 1)

        # Add color table if provided
        if class_colors:
            colormap = {i: (0, 0, 0, 0) for i in range(256)}  # Initialize transparent
            for class_idx, (r, g, b) in class_colors.items():
                if class_idx < 256:  # Color tables only support 256 colors
                    colormap[class_idx] = (r, g, b, 255)
            dst.write_colormap(1, colormap)

        # Add metadata
        dst.update_tags(**metadata)

        # Add band description
        dst.set_band_description(1, "Classification")

    return str(output_path)


def export_points_with_embeddings(
    points_df: pd.DataFrame,
    output_path: Union[str, Path],
    format: str = "auto",
    include_embeddings: bool = True,
    embedding_columns: Optional[List[str]] = None,
) -> str:
    """Export points with their extracted embeddings to various formats.

    Exports the results of extract_points() to geospatial formats,
    optionally including or excluding the embedding columns.

    Args:
        points_df: DataFrame from extract_points() with embeddings and metadata
        output_path: Path for output file
        format: Output format ('auto', 'geojson', 'gpkg', 'shp')
        include_embeddings: Whether to include embedding columns
        embedding_columns: Specific embedding columns to include (default: all)

    Returns:
        Path to created file

    Examples:
        >>> # Export points with full embeddings
        >>> export_points_with_embeddings(extracted_df, "points.geojson")

        >>> # Export only metadata (no embeddings)
        >>> export_points_with_embeddings(
        ...     extracted_df, "points_meta.shp",
        ...     include_embeddings=False
        ... )

        >>> # Export with selected embedding columns
        >>> export_points_with_embeddings(
        ...     extracted_df, "points_subset.gpkg",
        ...     embedding_columns=['emb_10', 'emb_20', 'emb_30']
        ... )
    """
    import geopandas as gpd
    from shapely.geometry import Point

    output_path = Path(output_path)

    # Auto-detect format
    if format == "auto":
        ext = output_path.suffix.lower()
        format_map = {
            ".geojson": "geojson",
            ".json": "geojson",
            ".gpkg": "gpkg",
            ".shp": "shp",
        }
        format = format_map.get(ext, "geojson")

    # Prepare data
    export_df = points_df.copy()

    # Handle embedding columns
    all_emb_cols = [col for col in export_df.columns if col.startswith("emb_")]

    if not include_embeddings:
        # Remove all embedding columns
        export_df = export_df.drop(columns=all_emb_cols)
    elif embedding_columns:
        # Keep only specified embedding columns
        cols_to_drop = [col for col in all_emb_cols if col not in embedding_columns]
        export_df = export_df.drop(columns=cols_to_drop)

    # Create geometry if not present
    if "lat" in export_df.columns and "lon" in export_df.columns:
        geometry = [Point(row.lon, row.lat) for _, row in export_df.iterrows()]
        gdf = gpd.GeoDataFrame(export_df, geometry=geometry, crs="EPSG:4326")
    else:
        raise ValueError("DataFrame must contain 'lat' and 'lon' columns")

    # Handle format-specific limitations
    if format == "shp":
        # Shapefile has column name length limit (10 chars)
        # Rename embedding columns if needed
        rename_map = {}
        for col in gdf.columns:
            if col.startswith("emb_") and len(col) > 10:
                # Create short name like 'e0', 'e1', etc.
                emb_idx = col.split("_")[1]
                short_name = f"e{emb_idx}"
                rename_map[col] = short_name

        if rename_map:
            gdf = gdf.rename(columns=rename_map)
            print(
                f"Note: Renamed {len(rename_map)} columns for shapefile compatibility"
            )

    # Export
    gdf.to_file(output_path, driver=format.upper())

    return str(output_path)


def create_vrt(
    raster_paths: List[Union[str, Path]],
    output_path: Union[str, Path],
    relative: bool = True,
) -> str:
    """Create a GDAL Virtual Raster (VRT) from multiple rasters.

    VRT files are lightweight XML files that reference other rasters,
    useful for viewing multiple tiles as a single dataset without
    actually merging them.

    Args:
        raster_paths: List of raster file paths
        output_path: Path for output VRT file
        relative: Use relative paths in VRT (more portable)

    Returns:
        Path to created VRT file

    Examples:
        >>> # Create VRT from classified tiles
        >>> tiles = ["tile1.tif", "tile2.tif", "tile3.tif"]
        >>> create_vrt(tiles, "classification.vrt")

        >>> # Open in QGIS or with rasterio
        >>> with rasterio.open("classification.vrt") as src:
        ...     print(src.bounds)
    """
    from osgeo import gdal

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert paths
    raster_paths = [str(Path(p).absolute()) for p in raster_paths]

    # Create VRT
    vrt_options = gdal.BuildVRTOptions(relative=relative)
    gdal.BuildVRT(str(output_path), raster_paths, options=vrt_options)

    return str(output_path)
