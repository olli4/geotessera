# GeoTessera

Python library for accessing and working with Tessera geospatial foundation model embeddings.

## Overview

GeoTessera provides access to geospatial embeddings from the [Tessera
foundation model](https://github.com/ucam-eo/tessera), which processes
Sentinel-1 and Sentinel-2 satellite imagery to generate 128-channel
representation maps at 10m resolution. These embeddings compress a full year of
temporal-spectral features into dense representations optimized for downstream
geospatial analysis tasks. Read more details about [the model](https://github.com/ucam-eo/tessera).

![Coverage map](https://raw.githubusercontent.com/ucam-eo/tessera-coverage-map/refs/heads/main/map.png)

## Table of Contents

- [Installation](#installation)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Python API](#python-api)
- [CLI Reference](#cli-reference)
- [Complete Workflows](#complete-workflows)
- [Registry System](#registry-system)
- [Data Organization](#data-organization)
- [Contributing](#contributing)

## Installation

```bash
pip install geotessera
```

For development:
```bash
git clone https://github.com/ucam-eo/geotessera
cd geotessera
pip install -e .
```

## Architecture

### Core Concepts

GeoTessera is built around a simple two-step workflow:

1. **Retrieve embeddings**: Fetch raw numpy arrays for a geographic bounding box
2. **Export to desired format**: Save as raw numpy arrays or convert to georeferenced GeoTIFF files

### Coordinate System and Tile Grid

The Tessera embeddings use a **0.1-degree grid system**:

- **Tile size**: Each tile covers 0.1° × 0.1° (approximately 11km × 11km at the equator)
- **Tile naming**: Tiles are named by their **center coordinates** (e.g., `grid_0.15_52.05`)
- **Tile bounds**: A tile at center (lon, lat) covers:
  - Longitude: [lon - 0.05°, lon + 0.05°]
  - Latitude: [lat - 0.05°, lat + 0.05°]
- **Resolution**: 10m per pixel (variable number of pixels per tile depending on latitude)

### File Structure and Downloads

When you request embeddings, GeoTessera downloads several files via Pooch:

#### Embedding Files (via `fetch_embedding`)
1. **Quantized embeddings** (`grid_X.XX_Y.YY.npy`):
   - Shape: `(height, width, 128)` 
   - Data type: int8 (quantized for storage efficiency)
   - Contains the compressed embedding values

2. **Scale files** (`grid_X.XX_Y.YY_scales.npy`):
   - Shape: `(height, width)` or `(height, width, 128)`
   - Data type: float32
   - Contains scale factors for dequantization

3. **Dequantization**: `final_embedding = quantized_embedding * scales`

#### Landmask Files (for GeoTIFF export)
When exporting to GeoTIFF, additional landmask files are fetched:
- **Landmask tiles** (`grid_X.XX_Y.YY.tiff`):
  - Provide UTM projection information
  - Define precise geospatial transforms
  - Contain land/water masks

### Data Flow

```
User Request (lat/lon bbox)
    ↓
Registry Lookup (find available tiles)
    ↓
Download Files (via Pooch with caching)
    ├── embedding.npy (quantized)
    └── embedding_scales.npy
    ↓
Dequantization (multiply arrays)
    ↓
Output Format
    ├── NumPy arrays → Direct analysis
    └── GeoTIFF → GIS integration
```

## Quick Start

### Check Available Data

Before downloading, check what data is available:

```bash
# Generate a coverage map showing all available tiles
geotessera coverage --output coverage_map.png

# Generate a coverage map for the UK
geotessera coverage --country uk

# View coverage for a specific year
geotessera coverage --year 2024 --output coverage_2024.png

# Customize the visualization
geotessera coverage --year 2024 --tile-color blue --tile-alpha 0.3 --dpi 150
```

### Download Embeddings

Download embeddings as either numpy arrays or GeoTIFF files:

```bash
# Download as GeoTIFF (default, with georeferencing)
geotessera download \
  --bbox "-0.2,51.4,0.1,51.6" \
  --year 2024 \
  --output ./london_tiffs

# Download as raw numpy arrays (with metadata JSON)
geotessera download \
  --bbox "-0.2,51.4,0.1,51.6" \
  --format npy \
  --year 2024 \
  --output ./london_arrays

# Download using a GeoJSON/Shapefile region
geotessera download \
  --region-file cambridge.geojson \
  --format tiff \
  --year 2024 \
  --output ./cambridge_tiles

# Download specific bands only
geotessera download \
  --bbox "-0.2,51.4,0.1,51.6" \
  --bands "0,1,2" \
  --year 2024 \
  --output ./london_rgb
```

### Create Visualizations

Generate web maps from downloaded GeoTIFFs:

```bash
# Create an interactive web map
geotessera visualize \
  ./london_tiffs \
  --type web \
  --output ./london_web

# Create an RGB mosaic
geotessera visualize \
  ./london_tiffs \
  --type rgb \
  --bands "30,60,90" \
  --output ./london_rgb

# Serve the web map locally
geotessera serve ./london_web --open
```

## Python API

### Core Methods

The library provides two main methods for retrieving embeddings:

```python
from geotessera import GeoTessera

# Initialize the client
gt = GeoTessera()

# Method 1: Fetch a single tile
embedding, crs, transform = gt.fetch_embedding(lat=52.05, lon=0.15, year=2024)
print(f"Shape: {embedding.shape}")  # e.g., (1200, 1200, 128)
print(f"CRS: {crs}")  # Coordinate reference system from landmask

# Method 2: Fetch all tiles in a bounding box
bbox = (-0.2, 51.4, 0.1, 51.6)  # (min_lon, min_lat, max_lon, max_lat)
embeddings = gt.fetch_embeddings(bbox, year=2024)

for tile_lat, tile_lon, embedding_array, crs, transform in embeddings:
    print(f"Tile ({tile_lat}, {tile_lon}): {embedding_array.shape}")
```

### Export Formats

#### Export as GeoTIFF

```python
# Export embeddings for a region as individual GeoTIFF files
files = gt.export_embedding_geotiffs(
    bbox=(-0.2, 51.4, 0.1, 51.6),
    output_dir="./output",
    year=2024,
    bands=None,  # Export all 128 bands (default)
    compress="lzw"  # Compression method
)

print(f"Created {len(files)} GeoTIFF files")

# Export specific bands only (e.g., first 3 for RGB visualization)
files = gt.export_embedding_geotiffs(
    bbox=(-0.2, 51.4, 0.1, 51.6),
    output_dir="./rgb_output",
    year=2024,
    bands=[0, 1, 2]  # Only export first 3 bands
)
```

#### Work with NumPy Arrays

```python
# Fetch and process embeddings directly
embeddings = gt.fetch_embeddings(bbox, year=2024)

for lat, lon, embedding, crs, transform in embeddings:
    # Compute statistics
    mean_values = np.mean(embedding, axis=(0, 1))  # Mean per channel
    std_values = np.std(embedding, axis=(0, 1))    # Std per channel
    
    # Extract specific pixels
    center_pixel = embedding[embedding.shape[0]//2, embedding.shape[1]//2, :]
    
    # Apply custom processing
    processed = your_analysis_function(embedding)
```

### Visualization Functions

```python
from geotessera.visualization import (
    create_rgb_mosaic_from_geotiffs,
    create_coverage_summary_map,
    visualize_global_coverage,
    geotiff_to_web_tiles
)

# Create an RGB mosaic from multiple GeoTIFF files
create_rgb_mosaic_from_geotiffs(
    geotiff_paths=["tile1.tif", "tile2.tif"],
    output_path="mosaic.tif",
    bands=(0, 1, 2),  # RGB bands
    normalize=True  # Normalize to 0-255
)

# Generate web tiles for interactive maps
geotiff_to_web_tiles(
    geotiff_path="mosaic.tif",
    output_dir="./web_tiles",
    zoom_levels=(8, 15)
)

# Create a global coverage visualization
visualize_global_coverage(
    tessera_client=gt,
    output_path="global_coverage.png",
    year=2024,  # Or None for all years
    width_pixels=2000,
    tile_color="red",
    tile_alpha=0.6
)
```

## CLI Reference

### download

Download embeddings for a region in your preferred format:

```bash
geotessera download [OPTIONS]

Options:
  -o, --output PATH         Output directory [required]
  --bbox TEXT              Bounding box: 'min_lon,min_lat,max_lon,max_lat'
  --region-file PATH       GeoJSON/Shapefile to define region
  -f, --format TEXT        Output format: 'tiff' or 'npy' (default: tiff)
  --year INT               Year of embeddings (default: 2024)
  --bands TEXT             Comma-separated band indices (default: all 128)
  --compress TEXT          Compression for TIFF format (default: lzw)
  --list-files             List all created files with details
  -v, --verbose            Verbose output
```

Output formats:
- **tiff**: Georeferenced GeoTIFF files with UTM projection
- **npy**: Raw numpy arrays with metadata.json file

### visualize

Create visualizations from GeoTIFF files:

```bash
geotessera visualize INPUT_PATH [OPTIONS]

Options:
  -o, --output PATH        Output directory [required]
  --type TEXT              Visualization type: rgb, web, coverage
  --bands TEXT             Comma-separated band indices for RGB
  --normalize              Normalize bands
  --min-zoom INT           Min zoom for web tiles (default: 8)
  --max-zoom INT           Max zoom for web tiles (default: 15)
  --force                  Force regeneration of tiles
```

### coverage

Generate a world map showing data availability:

```bash
geotessera coverage [OPTIONS]

Options:
  -o, --output PATH        Output PNG file (default: tessera_coverage.png)
  --year INT               Specific year to visualize
  --tile-color TEXT        Color for tiles (default: red)
  --tile-alpha FLOAT       Transparency 0-1 (default: 0.6)
  --tile-size FLOAT        Size multiplier (default: 1.0)
  --dpi INT                Output resolution (default: 100)
  --width INT              Figure width in inches (default: 20)
  --height INT             Figure height in inches (default: 10)
  --no-countries           Don't show country boundaries
```

### serve

Serve web visualizations locally:

```bash
geotessera serve DIRECTORY [OPTIONS]

Options:
  -p, --port INT           Port number (default: 8000)
  --open/--no-open         Auto-open browser (default: open)
  --html TEXT              Specific HTML file to serve
```

### info

Display information about GeoTIFF files or the library:

```bash
geotessera info [OPTIONS]

Options:
  --geotiffs PATH          Analyze GeoTIFF files/directory
  --dataset-version TEXT   Tessera dataset version
  -v, --verbose            Verbose output
```

## Registry System

### Overview

GeoTessera uses a [registry system](https://github.com/ucam-eo/tessera-manifests) to efficiently manage and access the large Tessera dataset:

- **Block-based organization**: Registry divided into 5×5 degree geographic blocks
- **Lazy loading**: Only loads registry blocks for the region you're accessing
- **Automatic caching**: Downloads are cached locally using Pooch
- **Integrity checking**: SHA256 checksums ensure data integrity

### Registry Sources

The registry can be loaded from multiple sources (in priority order):

1. **Local directory** (via `--registry-dir` or `registry_dir` parameter)
2. **Environment variable** (`TESSERA_REGISTRY_DIR`)
3. **Auto-cloned repository** (default, from GitHub)

```python
# Use local registry
gt = GeoTessera(registry_dir="/path/to/tessera-manifests")

# Use auto-updating registry
gt = GeoTessera(auto_update=True)

# Use custom manifest repository
gt = GeoTessera(
    manifests_repo_url="https://github.com/your-org/custom-manifests.git"
)
```

### Registry Structure

```
tessera-manifests/
└── registry/
    ├── embeddings/
    │   ├── embeddings_2024_lon-5_lat50.txt    # 5×5° block
    │   ├── embeddings_2024_lon0_lat50.txt
    │   └── ...
    └── landmasks/
        ├── landmasks_lon-5_lat50.txt
        ├── landmasks_lon0_lat50.txt
        └── ...
```

Each registry file contains:
```
# Pooch registry format
filepath SHA256checksum
2024/grid_0.15_52.05/grid_0.15_52.05.npy sha256:abc123...
2024/grid_0.15_52.05/grid_0.15_52.05_scales.npy sha256:def456...
```

### How Registry Loading Works

1. **Request tiles for bbox** → Determine which 5×5° blocks overlap
2. **Load block registries** → Parse only the needed registry files
3. **Find available tiles** → List tiles within the requested region
4. **Fetch via Pooch** → Download with caching and integrity checks

## Data Organization

### Tessera Data Structure

```
Remote Server (dl-2.tessera.wiki)
├── v1/                              # Dataset version
│   ├── 2024/                        # Year
│   │   ├── grid_0.15_52.05/         # Tile (named by center coords)
│   │   │   ├── grid_0.15_52.05.npy              # Quantized embeddings
│   │   │   └── grid_0.15_52.05_scales.npy       # Scale factors
│   │   └── ...
│   └── landmasks/
│       ├── grid_0.15_52.05.tiff     # Landmask with projection info
│       └── ...
```

### Local Cache Structure

```
~/.cache/geotessera/                 # Default cache location
├── tessera-manifests/                # Auto-cloned registry
│   └── registry/
├── pooch/                            # Downloaded data files
│   ├── grid_0.15_52.05.npy
│   ├── grid_0.15_52.05_scales.npy
│   └── ...
```

### Coordinate Reference Systems

- **Embeddings**: Stored in simple arrays, referenced by center coordinates
- **GeoTIFF exports**: Use UTM projection from corresponding landmask tiles
- **Web visualizations**: Reprojected to Web Mercator (EPSG:3857)

## Environment Variables

```bash
# Set custom cache directory for downloaded files
export TESSERA_DATA_DIR=/path/to/cache

# Use local registry directory
export TESSERA_REGISTRY_DIR=/path/to/tessera-manifests

# Configure per-command
TESSERA_DATA_DIR=/tmp/cache geotessera download ...
```

## Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for details.
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use Tessera in your research, please cite the [arXiv paper](https://arxiv.org/abs/2506.20380):

```bibtex
@misc{feng2025tesseratemporalembeddingssurface,
      title={TESSERA: Temporal Embeddings of Surface Spectra for Earth Representation and Analysis}, 
      author={Zhengpeng Feng and Clement Atzberger and Sadiq Jaffer and Jovana Knezevic and Silja Sormunen and Robin Young and Madeline C Lisaius and Markus Immitzer and David A. Coomes and Anil Madhavapeddy and Andrew Blake and Srinivasan Keshav},
      year={2025},
      eprint={2506.20380},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.20380}, 
}
```

## Links

- [Tessera Foundation Model](https://github.com/ucam-eo/tessera)
- [Tessera Interactive Notebook](https://github.com/ucam-eo/tessera-interactive-notebook)
- [Documentation](https://geotessera.readthedocs.io/)
- [PyPI Package](https://pypi.org/project/geotessera/)
- [Issue Tracker](https://github.com/ucam-eo/geotessera/issues)
