"""I/O utilities for working with various geospatial data formats.

This module provides flexible format support for loading training data,
saving embeddings, and working with regions of interest. It supports
common formats used in geospatial ML workflows.
"""

from pathlib import Path
from typing import Union, List, Dict, Optional
import json
import pandas as pd
import geopandas as gpd


def load_points(file_path: Union[str, Path], format: str = "auto") -> List[Dict]:
    """Load labeled points from various formats.

    Supports multiple formats for maximum flexibility:
    - JSON: Tessera-classification format with [[lat, lon], label] structure
    - GeoJSON: Point features with properties
    - CSV: Tabular data with lat, lon columns
    - Shapefile: Point geometries with attributes
    - GeoPackage: Modern geospatial format

    Args:
        file_path: Path to the input file
        format: File format. If 'auto', will infer from extension.
               Options: 'auto', 'json', 'geojson', 'csv', 'shp', 'gpkg'

    Returns:
        List of dictionaries with at least 'lat', 'lon' keys, plus any other attributes

    Examples:
        >>> # Load from tessera-classification JSON format
        >>> points = load_points("training_labels.json")
        >>> print(points[0])
        {'lat': 52.2053, 'lon': 0.1218, 'label': 'Urban'}

        >>> # Load from CSV with custom columns
        >>> points = load_points("survey_data.csv")
        >>> print(points[0])
        {'lat': 52.2053, 'lon': 0.1218, 'species': 'Oak', 'height': 15.2}
    """
    file_path = Path(file_path)

    # Auto-detect format from extension
    if format == "auto":
        ext = file_path.suffix.lower()
        format_map = {
            ".json": "json",
            ".geojson": "geojson",
            ".csv": "csv",
            ".shp": "shp",
            ".gpkg": "gpkg",
        }
        format = format_map.get(ext)
        if not format:
            raise ValueError(f"Cannot auto-detect format for extension: {ext}")

    # Load based on format
    if format == "json":
        # Tessera-classification format
        with open(file_path, "r") as f:
            data = json.load(f)

        # Handle both old and new tessera-classification formats
        if "training_points" in data:
            # New format: [[lat, lon], label]
            points = []
            for item in data["training_points"]:
                coord, label = item[0], item[1]
                points.append({"lat": coord[0], "lon": coord[1], "label": label})
            return points
        else:
            # Assume it's a simple list of point dictionaries
            return data

    elif format == "geojson":
        gdf = gpd.read_file(file_path)
        points = []
        for idx, row in gdf.iterrows():
            point_dict = {"lat": row.geometry.y, "lon": row.geometry.x}
            # Add all properties
            for col in gdf.columns:
                if col != "geometry":
                    point_dict[col] = row[col]
            points.append(point_dict)
        return points

    elif format == "csv":
        df = pd.read_csv(file_path)

        # Try to find latitude/longitude columns (case-insensitive)
        lat_cols = [
            col for col in df.columns if col.lower() in ["lat", "latitude", "y"]
        ]
        lon_cols = [
            col for col in df.columns if col.lower() in ["lon", "lng", "longitude", "x"]
        ]

        if not lat_cols or not lon_cols:
            raise ValueError("CSV must contain latitude and longitude columns")

        # Rename to standard names
        df = df.rename(columns={lat_cols[0]: "lat", lon_cols[0]: "lon"})

        return df.to_dict("records")

    elif format in ["shp", "gpkg"]:
        gdf = gpd.read_file(file_path)

        # Ensure we have point geometries
        if not all(geom.geom_type == "Point" for geom in gdf.geometry):
            raise ValueError(
                f"{format.upper()} file must contain only Point geometries"
            )

        points = []
        for idx, row in gdf.iterrows():
            point_dict = {"lat": row.geometry.y, "lon": row.geometry.x}
            # Add all attributes
            for col in gdf.columns:
                if col != "geometry":
                    point_dict[col] = row[col]
            points.append(point_dict)
        return points

    else:
        raise ValueError(f"Unsupported format: {format}")


def save_embeddings(
    embeddings: pd.DataFrame,
    output_path: Union[str, Path],
    format: str = "auto",
    compression: Optional[str] = None,
):
    """Save extracted embeddings to various formats.

    Supports efficient storage formats for ML workflows:
    - Parquet: Recommended for large datasets (columnar, compressed)
    - CSV: Universal compatibility
    - HDF5: Hierarchical data format
    - Pickle: Python native format

    Args:
        embeddings: DataFrame with embeddings and metadata
        output_path: Path for output file
        format: Output format. If 'auto', will infer from extension.
               Options: 'auto', 'parquet', 'csv', 'hdf5', 'h5', 'pickle', 'pkl'
        compression: Compression to use. For parquet: 'snappy', 'gzip', 'brotli'.
                    For CSV: 'gzip', 'bz2', 'xz'. None for no compression.

    Examples:
        >>> # Save to parquet (recommended for large datasets)
        >>> save_embeddings(df, "embeddings.parquet")

        >>> # Save to compressed CSV
        >>> save_embeddings(df, "embeddings.csv.gz", compression='gzip')
    """
    output_path = Path(output_path)

    # Auto-detect format
    if format == "auto":
        # Handle compressed extensions
        if output_path.suffix == ".gz":
            base_ext = output_path.with_suffix("").suffix.lower()
            compression = "gzip"
        else:
            base_ext = output_path.suffix.lower()

        format_map = {
            ".parquet": "parquet",
            ".csv": "csv",
            ".h5": "hdf5",
            ".hdf5": "hdf5",
            ".pkl": "pickle",
            ".pickle": "pickle",
        }
        format = format_map.get(base_ext)
        if not format:
            raise ValueError(f"Cannot auto-detect format for extension: {base_ext}")

    # Save based on format
    if format == "parquet":
        compression = compression or "snappy"  # Default compression for parquet
        embeddings.to_parquet(output_path, compression=compression, index=False)

    elif format == "csv":
        embeddings.to_csv(output_path, index=False, compression=compression)

    elif format in ["hdf5", "h5"]:
        # Use fixed format for better compatibility
        embeddings.to_hdf(output_path, key="embeddings", mode="w", format="fixed")

    elif format in ["pickle", "pkl"]:
        embeddings.to_pickle(output_path, compression=compression)

    else:
        raise ValueError(f"Unsupported format: {format}")


def load_roi(file_path: Union[str, Path]) -> gpd.GeoDataFrame:
    """Load region of interest from any spatial format supported by GeoPandas.

    Supports all vector formats that GeoPandas/Fiona can read:
    - Shapefile (.shp)
    - GeoJSON (.geojson, .json)
    - GeoPackage (.gpkg)
    - KML (.kml)
    - And many more

    The returned GeoDataFrame will be in EPSG:4326 (WGS84) coordinate system.

    Args:
        file_path: Path to the spatial file

    Returns:
        GeoDataFrame containing the region of interest geometry in EPSG:4326

    Examples:
        >>> # Load a study area boundary
        >>> roi = load_roi("study_area.shp")
        >>> print(roi.crs)  # EPSG:4326
        >>> print(roi.total_bounds)  # [west, south, east, north]
    """
    # Load the file
    gdf = gpd.read_file(file_path)

    # Convert to WGS84 if needed
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    return gdf


def load_embeddings(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load previously saved embeddings from various formats.

    Automatically detects format based on file extension.

    Args:
        file_path: Path to embeddings file

    Returns:
        DataFrame with embeddings and metadata

    Examples:
        >>> # Load from parquet
        >>> df = load_embeddings("embeddings.parquet")
        >>> print(df.shape)
        (1000, 130)  # 1000 points, 128 embeddings + 2 metadata columns
    """
    file_path = Path(file_path)

    # Detect format from extension
    ext = file_path.suffix.lower()

    # Handle compressed files
    if ext == ".gz":
        base_ext = file_path.with_suffix("").suffix.lower()
        if base_ext == ".csv":
            return pd.read_csv(file_path, compression="gzip")

    # Load based on extension
    if ext == ".parquet":
        return pd.read_parquet(file_path)
    elif ext == ".csv":
        return pd.read_csv(file_path)
    elif ext in [".h5", ".hdf5"]:
        return pd.read_hdf(file_path, key="embeddings")
    elif ext in [".pkl", ".pickle"]:
        return pd.read_pickle(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
