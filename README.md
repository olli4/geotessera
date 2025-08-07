# GeoTessera

Python library interface to the Tessera geofoundation model embeddings.

## Overview

GeoTessera provides access to geospatial embeddings from the [Tessera foundation model](https://github.com/ucam-eo/tessera), which processes Sentinel-1 and Sentinel-2 satellite imagery to generate 128-channel representation maps at 10m resolution. The embeddings compress a full year of temporal-spectral features into useful representations for geospatial analysis tasks.

## Data Coverage

![My Real-time Map](map.png)

## Features

- **Flexible Registry Sources**: Use local registry files, remote downloads, or auto-cloned repositories
- **Efficient Block-Based Loading**: Lazy loading of registry data using 5×5 degree geographic blocks
- **Environment Variable Support**: Configure registry location via `TESSERA_REGISTRY_DIR`
- **Auto-Updating Manifests**: Automatically clone and update registry manifests from GitHub
- **Geographic Data Access**: Download geospatial embeddings for specific coordinates
- **Comprehensive CLI**: List tiles, create visualizations, serve interactive maps
- **Built-in Caching**: Efficient local caching of downloaded files with Pooch
- **Registry Management Tools**: Generate and maintain registry files for data maintainers
- **Robust Error Handling**: Fatal errors for missing tiles ensure data integrity

## Installation

```bash
pip install geotessera
```

## Configuration

GeoTessera supports multiple configuration options for both data caching and registry management.

### Data Cache Configuration

Files are cached in the system's default cache directory (`~/.cache/geotessera` on Unix-like systems). You can customize this:

```bash
# Set custom cache directory
export TESSERA_DATA_DIR=/path/to/your/cache/directory

# Or set for a single command
TESSERA_DATA_DIR=/tmp/tessera geotessera info
```

### Registry Configuration

GeoTessera can load registry files from multiple sources, checked in this priority order:

1. **Explicit registry directory** (via `--registry-dir` or constructor parameter)
2. **Environment variable** (`TESSERA_REGISTRY_DIR`)
3. **Auto-cloned manifests repository** (default behavior)

#### Using Environment Variable

```bash
# Point to local registry directory
export TESSERA_REGISTRY_DIR=/path/to/tessera-manifests

# Use with any command
geotessera info
```

#### Using Auto-Cloned Repository

By default, GeoTessera automatically clones the [tessera-manifests](https://github.com/ucam-eo/tessera-manifests) repository to your cache directory. This can be configured:

```python
from geotessera import GeoTessera

# Use default auto-cloning
tessera = GeoTessera()

# Auto-update to latest manifests
tessera = GeoTessera(auto_update=True)

# Use custom manifest repository
tessera = GeoTessera(
    manifests_repo_url="https://github.com/your-org/custom-manifests.git"
)
```

## Usage

### Command Line Interface

All CLI commands support registry configuration options:

```bash
# Global options available for all commands:
# --registry-dir PATH          Use local registry directory
# --auto-update               Update manifests to latest version
# --manifests-repo-url URL    Custom manifests repository URL
```

#### Basic Commands

Make this `uv --from git+https://github.com/ucam-eo/geotessera@main` if you don't want to clone this repository.

```bash
# List available embeddings
uvx geotessera list-embeddings --limit 10

# Show dataset information  
uvx geotessera info

# Generate world map showing embedding coverage
uvx geotessera map --output coverage_map.png
```

#### Using Local Registry

```bash
# Use local registry directory
uvx geotessera \
  --registry-dir /path/to/tessera-manifests info

# Auto-update manifests before use
uvx geotessera \
  --auto-update info
```

#### Visualization Commands

```bash
# Create false-color visualization for a region
uvx geotessera visualize \
  --topojson example/CB.geojson --output cambridge_viz.tiff

# Serve interactive web map
uvx geotessera serve --geojson example/CB.geojson --open

# Serve with custom band selection (e.g., bands 30, 60, 90)
uvx geotessera serve \
  --geojson example/CB.geojson --bands 30 60 90 --open
```

If you have the repository checked out, use `--from .` instead.

### Python API

```python
from geotessera import GeoTessera

# Initialize with default settings (auto-clone manifests)
tessera = GeoTessera()

# Use local registry directory
tessera = GeoTessera(
    version="v1",
    registry_dir="/path/to/tessera-manifests"
)

# Auto-update manifests to latest version
tessera = GeoTessera(
    version="v1", 
    auto_update=True
)

# Use custom manifests repository
tessera = GeoTessera(
    version="v1",
    manifests_repo_url="https://github.com/your-org/custom-manifests.git"
)

# Download and get dequantized embedding for specific coordinates
embedding = tessera.fetch_embedding(lat=52.05, lon=0.15, year=2024)
print(f"Embedding shape: {embedding.shape}")  # (height, width, 128)

# List available embeddings for exploration
for year, lat, lon in tessera.list_available_embeddings():
    print(f"Year {year}: ({lat:.2f}, {lon:.2f})")

# Get available years
years = tessera.get_available_years()
print(f"Available years: {years}")

# Find tiles that intersect with a geometry
from shapely.geometry import box
geometry = box(-0.2, 51.9, 0.3, 52.3)  # Cambridge area
tiles = tessera.find_tiles_for_geometry(geometry, year=2024)
print(f"Found {len(tiles)} tiles for the geometry")

# Extract embeddings at specific points
points = [(52.2053, 0.1218), (52.1951, 0.1313)]  # Cambridge locations
embeddings_df = tessera.extract_points(points, year=2024, include_coords=True)
print(f"Extracted embeddings shape: {embeddings_df.shape}")

# Get tile metadata
bounds = tessera.get_tile_bounds(lat=52.05, lon=0.15)
crs = tessera.get_tile_crs(lat=52.05, lon=0.15)
transform = tessera.get_tile_transform(lat=52.05, lon=0.15)
print(f"Tile bounds: {bounds}")
print(f"Tile CRS: {crs}")

# Export single tile as GeoTIFF
tessera.export_single_tile_as_tiff(
    lat=52.05, lon=0.15, 
    output_path="tile.tiff",
    year=2024,
    bands=[0, 1, 2]  # RGB visualization
)

# Merge embeddings for a region
bounds = (-0.2, 51.9, 0.3, 52.3)  # (west, south, east, north)
tessera.merge_embeddings_for_region(
    bounds=bounds,
    output_path="region.tiff",
    target_crs="EPSG:4326",
    bands=[0, 1, 2],
    year=2024
)
```

## Registry Architecture

GeoTessera uses a block-based registry system for efficient data access to the needed tiles:

### Block-Based Organization

- **5×5 degree geographic blocks**: Registry files are organized into blocks to enable lazy loading
- **Embeddings**: `embeddings_YYYY_lonX_latY.txt` (e.g., `embeddings_2024_lon-5_lat50.txt`)
- **Landmasks**: `landmasks_lonX_latY.txt` (e.g., `landmasks_lon0_lat50.txt`)
- **Lazy Loading**: Only loads registry files for geographic regions being accessed

### Registry Directory Structure

```
tessera-manifests/
├── registry/
│   ├── embeddings/
│   │   ├── embeddings_2024_lon-180_lat-90.txt
│   │   ├── embeddings_2024_lon-175_lat-90.txt
│   │   └── ...
│   └── landmasks/
│       ├── landmasks_lon-180_lat-90.txt
│       ├── landmasks_lon-175_lat-90.txt
│       └── ...
```

### Registry File Format

Registry files use Pooch-compatible format for data integrity:

```
# Format: filepath checksum
v1/2024/grid_0.15_52.05/embedding.npy 2a1c8d7e9f3b5a6c8e7d9f2a1c8d7e9f3b5a6c8e7d9f2a1c8d7e9f3b5a6c8e7d
v1/2024/grid_0.15_52.05/embedding_scales.npy 5f9e2d1a8c6b9e3f7d2a8c6b9e3f7d2a8c6b9e3f7d2a8c6b9e3f7d2a8c6b9
```

## Registry Management (Data Maintainers)

GeoTessera includes tools for generating and maintaining registry files. The registry system uses block-based organization for efficient access to large datasets.

### Registry Generation Workflow

```bash
# 1. Generate SHA256 checksums for data files
uvx --from git+https://github.com/ucam-eo/geotessera@main geotessera-registry hash /path/to/v1

# 2. Scan checksums and create block-based pooch registries  
uvx --from git+https://github.com/ucam-eo/geotessera@main geotessera-registry scan /path/to/v1

# 3. List generated registry files
uvx --from git+https://github.com/ucam-eo/geotessera@main geotessera-registry list /path/to/v1
```

### Expected Data Structure

The registry tools expect this directory structure:

```
v1/
├── global_0.1_degree_representation/  # Embedding .npy files by year
│   ├── 2024/
│   │   ├── grid_0.15_52.05/
│   │   │   ├── embedding.npy
│   │   │   ├── embedding_scales.npy  
│   │   │   └── SHA256               # Generated by hash command
│   │   └── ...
│   └── ...
└── global_0.1_degree_tiff_all/        # Landmask .tiff files
    ├── grid_0.15_52.05.tiff
    ├── SHA256SUM                      # Generated by hash command
    └── ...
```

### Registry Commands

```bash
# Generate SHA256 checksums (parallel processing)
geotessera-registry hash /path/to/data
# - Creates SHA256 files in each grid subdirectory
# - Creates SHA256SUM file for TIFF files using chunked processing

# Generate block-based pooch registries from checksums
geotessera-registry scan /path/to/data [--registry-dir /output/path]
# - Reads SHA256 files and creates registry/embeddings/ files
# - Reads SHA256SUM and creates registry/landmasks/ files
# - Organizes into 5×5 degree blocks for efficient loading

# List existing registry files with entry counts
geotessera-registry list /path/to/registry
```

## Error Handling

GeoTessera now provides robust error handling to ensure data integrity:

- **Fatal Errors**: Missing embedding tiles throw exceptions instead of silent warnings
- **Registry Validation**: Missing registry files cause fatal errors during initialization
- **Checksum Verification**: All downloaded files are verified against SHA256 checksums
- **Clear Error Messages**: Descriptive errors help diagnose configuration issues

## About Tessera

Tessera is a foundation model for Earth observation developed by the University of Cambridge. It learns temporal-spectral features from multi-source satellite data to enable advanced geospatial analysis including land classification and canopy height prediction.

For more information about the Tessera project, visit: https://github.com/ucam-eo/tessera

## Related Tools

For a higher-level, interactive interface for point-and-click land cover classification built on top of `GeoTessera`, check out the [Interactive Map Classifier](https://github.com/ucam-eo/tessera-interactive-map) as an example.
