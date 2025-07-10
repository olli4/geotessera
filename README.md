# GeoTessera

Python library interface to the Tessera geofoundation model embeddings.

## Overview

GeoTessera provides access to geospatial embeddings from the [Tessera foundation model](https://github.com/ucam-eo/tessera), which processes Sentinel-1 and Sentinel-2 satellite imagery to generate 128-channel representation maps at 10m resolution. The embeddings compress a full year of temporal-spectral features into useful representations for geospatial analysis tasks.

## Data Coverage

![My Real-time Map](map.png)

## Features

- Download geospatial embeddings for specific coordinates
- List available embedding tiles
- Visualize embedding data as RGB composite images
- Built-in caching for efficient data management
- Command-line interface for easy access
- Registry management tools for data maintainers

## Installation

```bash
pip install git+https://github.com/ucam-eo/geotessera
```

## Usage

### Command Line Interface

Use `uvx` to run the CLI without installation:

```bash
# List available embeddings
uvx --from git+https://github.com/ucam-eo/geotessera@main geotessera list-embeddings --limit 10

# Show dataset information
uvx --from git+https://github.com/ucam-eo/geotessera@main geotessera info

# Generate a world map showing embedding coverage
uvx --from git+https://github.com/ucam-eo/geotessera@main geotessera map --output coverage_map.png

# Create a false-color visualization for a region
uvx --from git+https://github.com/ucam-eo/geotessera@main geotessera visualize --topojson example/CB.geojson --output cambridge_viz.tiff

# Serve an interactive web map with Leaflet.js
uvx --from git+https://github.com/ucam-eo/geotessera@main geotessera serve --geojson example/CB.geojson --open

# Serve with custom band selection (e.g., bands 30, 60, 90)
uvx --from git+https://github.com/ucam-eo/geotessera@main geotessera serve --geojson example/CB.geojson --bands 30 60 90 --open
```

If you have the repository checked out, then use `--from .` instead.

### Python API

```python
from geotessera import GeoTessera

# Initialize client
tessera = GeoTessera(version="v1")

# Download and get dequantized embedding for specific coordinates
embedding = tessera.get_embedding(lat=52.05, lon=0.15, year=2024)
print(f"Embedding shape: {embedding.shape}")  # (height, width, 128)
```

## Registry Management (Data Maintainers)

GeoTessera includes a separate tool for managing the registry files used by the package. This tool is intended for data maintainers who need to generate or update the Pooch registry files that track available embeddings.

### Using geotessera-registry

```bash
# List existing registry files
uvx --from git+https://github.com/ucam-eo/geotessera@main geotessera-registry list /path/to/data

# Generate/update registry files for all years
uvx --from git+https://github.com/ucam-eo/geotessera@main geotessera-registry update /path/to/data

# Update incrementally (only process new files)
uvx --from git+https://github.com/ucam-eo/geotessera@main geotessera-registry update /path/to/data --incremental

# Generate with custom worker count and create master registry index
uvx --from git+https://github.com/ucam-eo/geotessera@main geotessera-registry update /path/to/data --workers 8 --generate-master
```

The `--generate-master` flag creates a `registry.txt` file that lists all available registry files without hashes, serving as a master index.

## About Tessera

Tessera is a foundation model for Earth observation developed by the University of Cambridge. It learns temporal-spectral features from multi-source satellite data to enable advanced geospatial analysis including land classification and canopy height prediction.

For more information about the Tessera project, visit: https://github.com/ucam-eo/tessera
