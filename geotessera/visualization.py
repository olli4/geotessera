"""Visualization utilities for GeoTessera GeoTIFF files.

This module provides tools for visualizing and processing GeoTIFF files
created by GeoTessera, including bounding box calculations, mosaicking,
and web map generation for CLI display.
"""

from pathlib import Path
from typing import Union, List, Tuple, Optional, Dict, Callable
import tempfile
import shutil
import json

import numpy as np
import geopandas as gpd
import pandas as pd


def calculate_bbox_from_file(filepath: Union[str, Path]) -> Tuple[float, float, float, float]:
    """Calculate bounding box from a geometry file.
    
    Args:
        filepath: Path to GeoJSON, Shapefile, etc.
        
    Returns:
        Bounding box as (min_lon, min_lat, max_lon, max_lat)
    """
    gdf = gpd.read_file(filepath)
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    return tuple(bounds)


def calculate_bbox_from_points(
    points: Union[List[Dict], pd.DataFrame], 
    buffer_degrees: float = 0.1
) -> Tuple[float, float, float, float]:
    """Calculate bounding box from point data.
    
    Args:
        points: List of dicts with 'lat'/'lon' keys or DataFrame with lat/lon columns
        buffer_degrees: Buffer around points in degrees
        
    Returns:
        Bounding box as (min_lon, min_lat, max_lon, max_lat)
    """
    if isinstance(points, list):
        df = pd.DataFrame(points)
    else:
        df = points
        
    if 'lat' not in df.columns or 'lon' not in df.columns:
        raise ValueError("Points must have 'lat' and 'lon' columns")
        
    min_lon = df['lon'].min() - buffer_degrees
    max_lon = df['lon'].max() + buffer_degrees
    min_lat = df['lat'].min() - buffer_degrees
    max_lat = df['lat'].max() + buffer_degrees
    
    return (min_lon, min_lat, max_lon, max_lat)


def create_rgb_mosaic_from_geotiffs(
    geotiff_paths: List[str],
    output_path: str,
    bands: Tuple[int, int, int] = (0, 1, 2),
    normalize: bool = True,
    progress_callback: Optional[Callable] = None
) -> str:
    """Create an RGB visualization mosaic from multiple GeoTIFF files.
    
    Args:
        geotiff_paths: List of paths to GeoTIFF files
        output_path: Output path for RGB mosaic
        bands: Three band indices to map to RGB channels
        normalize: Whether to normalize each band to 0-255 range
        progress_callback: Optional callback function(current, total, status) for progress tracking
        
    Returns:
        Path to created RGB mosaic file
    """
    try:
        import rasterio
        from rasterio.merge import merge
        from rasterio.enums import ColorInterp
    except ImportError:
        raise ImportError("rasterio required: pip install rasterio")
        
    if not geotiff_paths:
        raise ValueError("No GeoTIFF files provided")
    
    if progress_callback:
        progress_callback(10, 100, f"Opening {len(geotiff_paths)} GeoTIFF files...")
        
    # Open all files
    src_files = [rasterio.open(path) for path in geotiff_paths]
    
    try:
        if progress_callback:
            progress_callback(20, 100, "Checking coordinate systems...")
            
        # Check if all files have the same CRS
        first_crs = src_files[0].crs
        different_crs = [src for src in src_files if src.crs != first_crs]
        
        if different_crs:
            if progress_callback:
                progress_callback(25, 100, f"Reprojecting {len(different_crs)} files to common CRS...")
            
            # Use rasterio's warp functionality to reproject to common CRS
            from rasterio.warp import reproject, calculate_default_transform, Resampling
            from rasterio.io import MemoryFile
            
            # Create reprojected datasets for files with different CRS
            reprojected_datasets = []
            
            for i, src in enumerate(src_files):
                if src.crs == first_crs:
                    # Same CRS, use as-is
                    reprojected_datasets.append(src)
                else:
                    # Different CRS, reproject to first_crs
                    if progress_callback:
                        progress_callback(25 + int((i / len(src_files)) * 10), 100, 
                                        f"Reprojecting file {i+1}/{len(src_files)}...")
                    
                    # Calculate target transform
                    dst_transform, dst_width, dst_height = calculate_default_transform(
                        src.crs, first_crs, src.width, src.height, *src.bounds
                    )
                    
                    # Create in-memory reprojected dataset
                    memfile = MemoryFile()
                    dst_dataset = memfile.open(
                        driver='GTiff',
                        height=dst_height, width=dst_width, count=src.count,
                        dtype=src.dtypes[0], crs=first_crs, transform=dst_transform
                    )
                    
                    # Reproject each band
                    for band_idx in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, band_idx),
                            destination=rasterio.band(dst_dataset, band_idx),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=dst_transform,
                            dst_crs=first_crs,
                            resampling=Resampling.bilinear
                        )
                    
                    reprojected_datasets.append(dst_dataset)
            
            # Now merge the reprojected datasets
            if progress_callback:
                progress_callback(35, 100, "Merging reprojected files...")
            
            merged_array, merged_transform = merge(reprojected_datasets, method='first')
            merged_crs = first_crs
            
            # Close any temporary datasets we created
            for i, dataset in enumerate(reprojected_datasets):
                if src_files[i].crs != first_crs:
                    dataset.close()
        else:
            if progress_callback:
                progress_callback(30, 100, "Merging GeoTIFF files...")
                
            # All files have same CRS, use rasterio merge
            merged_array, merged_transform = merge(src_files, method='first')
            merged_crs = first_crs
        
        if progress_callback:
            progress_callback(40, 100, f"Extracting RGB bands {bands}...")
        
        # Extract the three bands for RGB
        if merged_array.shape[0] < max(bands) + 1:
            raise ValueError(f"Not enough bands in source files. Requested bands {bands}, but only {merged_array.shape[0]} available")
            
        rgb_data = merged_array[list(bands)]  # Shape: (3, height, width)
        
        # Normalize if requested
        if normalize:
            if progress_callback:
                progress_callback(60, 100, "Normalizing RGB bands...")
                
            for i in range(3):
                band = rgb_data[i]
                band_min, band_max = np.nanmin(band), np.nanmax(band)
                if band_max > band_min:
                    rgb_data[i] = (band - band_min) / (band_max - band_min)
                else:
                    rgb_data[i] = 0
        else:
            if progress_callback:
                progress_callback(60, 100, "Processing RGB bands...")
                    
        if progress_callback:
            progress_callback(80, 100, "Converting to RGB format...")
            
        # Convert to uint8
        rgb_uint8 = (np.clip(rgb_data, 0, 1) * 255).astype(np.uint8)
        
        if progress_callback:
            progress_callback(90, 100, f"Writing RGB mosaic to {Path(output_path).name}...")
        
        # Write RGB GeoTIFF
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=rgb_uint8.shape[1],
            width=rgb_uint8.shape[2],
            count=3,
            dtype='uint8',
            crs=merged_crs,
            transform=merged_transform,
            compress='lzw',
            photometric='RGB'
        ) as dst:
            dst.write(rgb_uint8)
            dst.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue]
            dst.update_tags(
                TIFFTAG_ARTIST="GeoTessera",
                TIFFTAG_IMAGEDESCRIPTION=f"RGB visualization using bands {bands}"
            )
    
    finally:
        # Close all source files
        for src in src_files:
            src.close()
    
    if progress_callback:
        progress_callback(100, 100, f"Completed RGB mosaic: {Path(output_path).name}")
        
    return output_path


def geotiff_to_web_tiles(
    geotiff_path: str,
    output_dir: str,
    zoom_levels: Tuple[int, int] = (8, 15)
) -> str:
    """Convert GeoTIFF to web tiles for interactive display.
    
    Args:
        geotiff_path: Path to input GeoTIFF
        output_dir: Directory for web tiles output
        zoom_levels: Min and max zoom levels
        
    Returns:
        Path to tiles directory
    """
    try:
        import subprocess
    except ImportError:
        raise ImportError("gdal2tiles required")
        
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    min_zoom, max_zoom = zoom_levels
    
    # Run gdal2tiles
    cmd = [
        'gdal2tiles.py',
        '-z', f'{min_zoom}-{max_zoom}',
        '-w', 'leaflet',
        '-p', 'mercator',  # Explicitly use mercator projection
        '--resampling', 'bilinear',
        geotiff_path,
        str(output_dir)
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return str(output_dir)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"gdal2tiles failed: {e}")
    except FileNotFoundError:
        raise RuntimeError("gdal2tiles.py not found. Install GDAL tools.")


def create_simple_web_viewer(
    tiles_dir: str,
    output_html: str,
    center_lat: float = 0,
    center_lon: float = 0,
    zoom: int = 10,
    title: str = "GeoTessera Visualization"
) -> str:
    """Create a simple HTML viewer for web tiles.
    
    Args:
        tiles_dir: Directory containing web tiles
        output_html: Output path for HTML file
        center_lat: Initial map center latitude
        center_lon: Initial map center longitude
        zoom: Initial zoom level
        title: Page title
        
    Returns:
        Path to created HTML file
    """
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <style>
        html, body {{ height: 100%; margin: 0; padding: 0; }}
        #map {{ height: 100%; }}
        .opacity-control {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: white;
            border-radius: 5px;
            padding: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            z-index: 1000;
            font-family: Arial, sans-serif;
            font-size: 12px;
        }}
        .opacity-control label {{
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }}
        .opacity-control input[type="range"] {{
            width: 150px;
        }}
        .opacity-value {{
            font-size: 11px;
            color: #666;
            margin-top: 2px;
        }}
    </style>
</head>
<body>
    <div id="map"></div>
    
    <!-- Opacity Control -->
    <div class="opacity-control">
        <label for="opacity-slider">GeoTessera Opacity</label>
        <input type="range" id="opacity-slider" min="0" max="100" value="80" step="5">
        <div class="opacity-value" id="opacity-value">80%</div>
    </div>
    
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        var map = L.map('map').setView([{center_lat}, {center_lon}], {zoom});
        
        // Add OpenStreetMap base layer
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '© OpenStreetMap contributors'
        }}).addTo(map);
        
        // Add GeoTessera layer
        var tesseraLayer = L.tileLayer('./tiles/{{z}}/{{x}}/{{y}}.png', {{
            attribution: 'GeoTessera data',
            opacity: 0.8,
            tms: true
        }}).addTo(map);
        
        // Layer control
        var baseMaps = {{
            "OpenStreetMap": L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png')
        }};
        
        var overlayMaps = {{
            "Tessera Data": tesseraLayer
        }};
        
        L.control.layers(baseMaps, overlayMaps).addTo(map);
        
        // Opacity slider functionality
        var opacitySlider = document.getElementById('opacity-slider');
        var opacityValue = document.getElementById('opacity-value');
        
        opacitySlider.addEventListener('input', function() {{
            var opacity = this.value / 100;
            tesseraLayer.setOpacity(opacity);
            opacityValue.textContent = this.value + '%';
        }});
    </script>
</body>
</html>"""
    
    with open(output_html, 'w') as f:
        f.write(html_content)
        
    return output_html


def analyze_geotiff_coverage(geotiff_paths: List[str]) -> Dict:
    """Analyze coverage and metadata of GeoTIFF files.
    
    Args:
        geotiff_paths: List of GeoTIFF file paths
        
    Returns:
        Dictionary with coverage statistics and metadata
    """
    try:
        import rasterio
    except ImportError:
        raise ImportError("rasterio required: pip install rasterio")
        
    if not geotiff_paths:
        return {"error": "No files provided"}
        
    coverage_info = {
        "total_files": len(geotiff_paths),
        "tiles": [],
        "bounds": {"min_lon": float('inf'), "min_lat": float('inf'),
                  "max_lon": float('-inf'), "max_lat": float('-inf')},
        "band_counts": {},
        "years": set(),
        "crs": set()
    }
    
    for path in geotiff_paths:
        try:
            with rasterio.open(path) as src:
                bounds = src.bounds
                
                # Convert bounds to lat/lon if needed
                if src.crs and src.crs != 'EPSG:4326':
                    from rasterio.warp import transform_bounds
                    # Transform bounds to WGS84 (lat/lon)
                    lon_min, lat_min, lon_max, lat_max = transform_bounds(
                        src.crs, 'EPSG:4326', bounds.left, bounds.bottom, bounds.right, bounds.top
                    )
                else:
                    # Already in lat/lon
                    lon_min, lat_min, lon_max, lat_max = bounds.left, bounds.bottom, bounds.right, bounds.top
                
                # Update overall bounds
                coverage_info["bounds"]["min_lon"] = min(coverage_info["bounds"]["min_lon"], lon_min)
                coverage_info["bounds"]["min_lat"] = min(coverage_info["bounds"]["min_lat"], lat_min)
                coverage_info["bounds"]["max_lon"] = max(coverage_info["bounds"]["max_lon"], lon_max)
                coverage_info["bounds"]["max_lat"] = max(coverage_info["bounds"]["max_lat"], lat_max)
                
                # Track band counts
                band_count = src.count
                coverage_info["band_counts"][band_count] = coverage_info["band_counts"].get(band_count, 0) + 1
                
                # Extract metadata
                tags = src.tags()
                if "TESSERA_YEAR" in tags:
                    coverage_info["years"].add(tags["TESSERA_YEAR"])
                    
                coverage_info["crs"].add(str(src.crs))
                
                # Tile info (use lat/lon bounds)
                coverage_info["tiles"].append({
                    "path": path,
                    "bounds": [lon_min, lat_min, lon_max, lat_max],
                    "bands": band_count,
                    "year": tags.get("TESSERA_YEAR", "unknown"),
                    "tile_lat": tags.get("TESSERA_TILE_LAT", "unknown"),
                    "tile_lon": tags.get("TESSERA_TILE_LON", "unknown")
                })
                
        except Exception as e:
            print(f"Warning: Failed to read {path}: {e}")
            continue
    
    # Convert sets to lists for JSON serialization
    coverage_info["years"] = sorted(list(coverage_info["years"]))
    coverage_info["crs"] = list(coverage_info["crs"])
    
    return coverage_info


def create_coverage_summary_map(
    geotiff_paths: List[str],
    output_html: str,
    title: str = "GeoTessera Coverage Map"
) -> str:
    """Create an HTML map showing tile coverage.
    
    Args:
        geotiff_paths: List of GeoTIFF file paths
        output_html: Output HTML file path
        title: Map title
        
    Returns:
        Path to created HTML file
    """
    # Analyze coverage
    coverage = analyze_geotiff_coverage(geotiff_paths)
    
    if not coverage["tiles"]:
        raise ValueError("No valid GeoTIFF files found")
        
    # Calculate center
    bounds = coverage["bounds"]
    center_lat = (bounds["min_lat"] + bounds["max_lat"]) / 2
    center_lon = (bounds["min_lon"] + bounds["max_lon"]) / 2
    
    # Generate tile rectangles for map
    tile_geojson = {
        "type": "FeatureCollection",
        "features": []
    }
    
    for tile in coverage["tiles"]:
        min_lon, min_lat, max_lon, max_lat = tile["bounds"]
        
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [min_lon, min_lat],
                    [max_lon, min_lat], 
                    [max_lon, max_lat],
                    [min_lon, max_lat],
                    [min_lon, min_lat]
                ]]
            },
            "properties": {
                "year": tile["year"],
                "bands": tile["bands"],
                "lat": tile["tile_lat"],
                "lon": tile["tile_lon"],
                "path": Path(tile["path"]).name
            }
        }
        tile_geojson["features"].append(feature)
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <style>
        html, body {{ height: 100%; margin: 0; padding: 0; }}
        #map {{ height: 100%; }}
        .info {{ 
            padding: 10px; background: white; background: rgba(255,255,255,0.9);
            box-shadow: 0 0 15px rgba(0,0,0,0.2); border-radius: 5px; 
            font-family: Arial, sans-serif; font-size: 12px;
        }}
        .info h4 {{ margin: 0 0 5px; color: #777; }}
    </style>
</head>
<body>
    <div id="map"></div>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        var map = L.map('map').setView([{center_lat}, {center_lon}], 8);
        
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '© OpenStreetMap contributors'
        }}).addTo(map);
        
        var geojsonData = {json.dumps(tile_geojson)};
        
        function style(feature) {{
            return {{
                fillColor: '#3388ff',
                weight: 1,
                opacity: 1,
                color: 'white',
                fillOpacity: 0.3
            }};
        }}
        
        function onEachFeature(feature, layer) {{
            var props = feature.properties;
            var popupContent = 
                "<b>Tessera Tile</b><br>" +
                "Year: " + props.year + "<br>" +
                "Bands: " + props.bands + "<br>" +
                "Position: (" + props.lat + ", " + props.lon + ")<br>" +
                "File: " + props.path;
            layer.bindPopup(popupContent);
        }}
        
        L.geoJSON(geojsonData, {{
            style: style,
            onEachFeature: onEachFeature
        }}).addTo(map);
        
        // Add info control
        var info = L.control();
        info.onAdd = function (map) {{
            this._div = L.DomUtil.create('div', 'info');
            this.update();
            return this._div;
        }};
        info.update = function (props) {{
            this._div.innerHTML = '<h4>GeoTessera Coverage</h4>' +
                'Total tiles: {coverage["total_files"]}<br>' +
                'Years: {", ".join(coverage["years"])}<br>' +
                'Click on tiles for details';
        }};
        info.addTo(map);
    </script>
</body>
</html>"""
    
    with open(output_html, 'w') as f:
        f.write(html_content)
        
    return output_html


def visualize_global_coverage(
    tessera_client,
    output_path: str = "tessera_coverage.png",
    year: Optional[int] = None,
    figsize: Tuple[int, int] = (20, 10),
    dpi: int = 100,
    show_countries: bool = True,
    tile_color: str = "red",
    tile_alpha: float = 0.6,
    tile_size: float = 1.0,
    progress_callback: Optional[Callable] = None
) -> str:
    """Create a world map visualization showing Tessera embedding coverage.
    
    Generates a PNG map with available tiles overlaid on world countries to
    help users understand data availability for their regions of interest.
    
    Args:
        tessera_client: GeoTessera instance with loaded registries
        output_path: Output filename for the PNG map
        year: Specific year to show coverage for. If None, shows all years
        figsize: Figure size as (width, height) in inches
        dpi: Dots per inch for output resolution
        show_countries: Whether to show country boundaries
        tile_color: Color for tile rectangles
        tile_alpha: Transparency of tile rectangles (0=transparent, 1=opaque)
        tile_size: Size multiplier for tile rectangles (1.0 = actual size)
        progress_callback: Optional callback function(current, total, status) for progress tracking
        
    Returns:
        Path to the created PNG file
        
    Example:
        >>> from geotessera import GeoTessera
        >>> gt = GeoTessera()
        >>> from geotessera.visualization import visualize_global_coverage
        >>> visualize_global_coverage(gt, "coverage_2024.png", year=2024)
        >>> visualize_global_coverage(gt, "coverage_all.png")  # All years
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.collections import PatchCollection
        import geodatasets
    except ImportError:
        raise ImportError(
            "Please install required packages: pip install matplotlib geodatasets"
        )
    
    # Import get_tile_bounds from registry module
    from .registry import get_tile_bounds
    
    # Load world countries from geodatasets
    if not progress_callback:
        print("Loading world map data...")
    world = gpd.read_file(geodatasets.get_path("naturalearth.land"))
    
    # Ensure we have embeddings loaded
    tessera_client.registry.ensure_all_blocks_loaded(progress_callback=progress_callback)
    
    # Get available embeddings
    available_embeddings = tessera_client.registry.get_available_embeddings()
    
    # Filter embeddings by year if specified
    if year is not None:
        tiles = [(lat, lon) for y, lat, lon in available_embeddings if y == year]
        title = f"Tessera Embedding Coverage - Year {year}"
    else:
        # Get unique tile locations across all years
        tile_set = set((lat, lon) for _, lat, lon in available_embeddings)
        tiles = list(tile_set)
        title = "Tessera Embedding Coverage - All Available Years"
    
    if not progress_callback:
        print(f"Found {len(tiles)} tiles to visualize")
    
    # Create figure and axis
    if progress_callback:
        progress_callback(0, 100, "Creating figure...")
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    
    # Plot world map
    if show_countries:
        if progress_callback:
            progress_callback(10, 100, "Plotting world map...")
        world.plot(ax=ax, color='lightgray', edgecolor='darkgray', linewidth=0.5)
    
    # Create rectangles for each tile (more accurate representation)
    if progress_callback:
        progress_callback(20, 100, f"Creating {len(tiles)} tile rectangles...")
    
    rectangles = []
    total_tiles = len(tiles)
    
    for i, (lat, lon) in enumerate(tiles):
        # Update progress every 100 tiles or at the end
        if progress_callback and (i % 100 == 0 or i == total_tiles - 1):
            progress = 20 + int((i / total_tiles) * 50)  # 20% to 70%
            progress_callback(progress, 100, f"Processing tile {i+1}/{total_tiles}...")
        
        # Get tile bounds using the helper function
        west, south, east, north = get_tile_bounds(lat, lon)
        
        # Apply size multiplier if needed
        if tile_size != 1.0:
            center_lon, center_lat = lon, lat
            half_width = (east - west) / 2 * tile_size
            half_height = (north - south) / 2 * tile_size
            west = center_lon - half_width
            east = center_lon + half_width
            south = center_lat - half_height
            north = center_lat + half_height
        
        # Create rectangle patch
        rect = mpatches.Rectangle(
            (west, south),
            east - west,
            north - south,
            linewidth=0,
            facecolor=tile_color,
            alpha=tile_alpha
        )
        rectangles.append(rect)
    
    # Add all rectangles as a collection for better performance
    if progress_callback:
        progress_callback(70, 100, "Adding tiles to map...")
    collection = PatchCollection(rectangles, match_original=True)
    ax.add_collection(collection)
    
    # Set axis properties
    if progress_callback:
        progress_callback(75, 100, "Setting up map properties...")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add statistics text
    if progress_callback:
        progress_callback(80, 100, "Adding statistics...")
    stats_text = f"Total tiles: {len(tiles):,}"
    if year is None:
        years = sorted(set(y for y, _, _ in available_embeddings))
        if years:
            stats_text += f"\nYears: {min(years)}-{max(years)}"
    
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add legend
    if progress_callback:
        progress_callback(85, 100, "Adding legend...")
    legend_elements = [
        mpatches.Patch(color=tile_color, alpha=tile_alpha, label='Available tiles'),
    ]
    if show_countries:
        legend_elements.append(
            mpatches.Patch(color='lightgray', label='Land masses')
        )
    ax.legend(handles=legend_elements, loc='lower left', fontsize=10)
    
    # Save figure
    if progress_callback:
        progress_callback(90, 100, "Saving image to disk...")
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    if progress_callback:
        progress_callback(100, 100, "Done!")
    else:
        print(f"Coverage map saved to: {output_path}")
    return output_path