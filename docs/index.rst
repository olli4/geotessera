GeoTessera Documentation
========================

GeoTessera provides access to open geospatial embeddings from the `Tessera foundation model <https://github.com/ucam-eo/tessera>`_
(`paper <https://arxiv.org/abs/2506.20380>`_). Tessera processes Sentinel-1 and
Sentinel-2 satellite imagery to generate 128-channel representation maps at 10m
resolution, compressing a full year of temporal-spectral features into dense
representations optimized for downstream geospatial analysis tasks.

Overview
--------

GeoTessera is built around a two-step workflow:

1. **Retrieve embeddings**: Fetch raw numpy arrays with CRS/transform information for a geographic bounding box
2. **Export to desired format**: Save as raw numpy arrays or convert to georeferenced GeoTIFF files with preserved projections

Key Features
------------

* **Global Coverage**: Access embeddings for any terrestrial location worldwide where data exists
* **Flexible Formats**: Export as numpy arrays for analysis or GeoTIFF for GIS integration
* **Projection Preservation**: Native UTM projections preserved from landmask tiles
* **High Resolution**: 10m spatial resolution
* **Temporal Compression**: Full year of satellite observations in each embedding
* **Multi-spectral**: Combines Sentinel-1 SAR and Sentinel-2 optical data
* **Country Support**: Download by country name or custom regions
* **Efficient Registry**: Block-based lazy loading of only required data
* **Easy Access**: Python API and CLI with automatic caching

Installation
------------

Install GeoTessera using pip::

    pip install geotessera

For development installation::

    git clone https://github.com/ucam-eo/geotessera
    cd geotessera
    pip install -e .

Quick Start
-----------

Check data availability first::

    # Generate a coverage map
    geotessera coverage --output coverage_map.png
    
    # View coverage for a specific year
    geotessera coverage --year 2024

    # Check coverage for a single country
    geotessera coverage --country uk

Download embeddings in your preferred format::

    # Download as GeoTIFF (georeferenced, for GIS)
    geotessera download --bbox "-0.2,51.4,0.1,51.6" --year 2024 --output ./london_tiffs --bands 1,2,3
    
    # Download as numpy arrays (for analysis, all 128 bands)
    geotessera download --bbox "-0.2,51.4,0.1,51.6" --format npy --year 2024 --output ./london_arrays

    # Download by country name
    geotessera download --country "United Kingdom" --year 2024 --output ./uk_tiles

    # Download tiles from a region file
    geotessera download --region-file example/CB.geojson --format npy --year 2024 --output ./cambridge


Python API usage::

    from geotessera import GeoTessera
    
    # Initialize client
    gt = GeoTessera()
    
    # Method 1: Fetch a single tile with CRS information
    embedding, crs, transform = gt.fetch_embedding(lon=0.15, lat=52.05, year=2024)
    print(f"Shape: {embedding.shape}")  # e.g., (1200, 1200, 128)
    print(f"CRS: {crs}")  # UTM projection
    
    # Method 2: Fetch all tiles in a bounding box
    bbox = (-0.2, 51.4, 0.1, 51.6)  # (min_lon, min_lat, max_lon, max_lat)
    tiles = gt.fetch_embeddings(bbox, year=2024)
    
    for tile_lon, tile_lat, embedding, crs, transform in tiles:
        print(f"Tile ({tile_lon}, {tile_lat}): {embedding.shape}")
    
    # Export as GeoTIFF files with preserved UTM projections
    files = gt.export_embedding_geotiffs(
        bbox=bbox,
        output_dir="./output",
        year=2024,
        bands=[0, 1, 2]  # Export first 3 bands only
    )

Create web visualizations::

    # Create interactive web map from GeoTIFFs
    geotessera visualize ./london_tiffs --type web --output ./london_web
    geotessera serve ./london_web --open

Architecture Overview
---------------------

Coordinate System and Tile Grid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Tessera embeddings use a **0.1-degree grid system**:

* **Tile size**: Each tile covers 0.1° × 0.1° (approximately 11km × 11km at the equator)
* **Tile naming**: Tiles are named by their **center coordinates** (e.g., ``grid_0.15_52.05``)
* **Tile bounds**: A tile at center (lon, lat) covers [lon ± 0.05°, lat ± 0.05°]
* **Resolution**: 10m per pixel (variable pixels per tile depending on latitude)

File Structure and Downloads
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When you request embeddings, GeoTessera downloads several files via Pooch:

**Embedding Files** (via ``fetch_embedding``):

1. **Quantized embeddings** (``grid_X.XX_Y.YY.npy``):
   
   * Shape: ``(height, width, 128)``
   * Data type: int8 (quantized for storage efficiency)
   * Contains the compressed embedding values

2. **Scale files** (``grid_X.XX_Y.YY_scales.npy``):
   
   * Shape: ``(height, width)`` or ``(height, width, 128)``
   * Data type: float32
   * Contains scale factors for dequantization

3. **Dequantization**: ``final_embedding = quantized_embedding * scales``

**Landmask Files** (with CRS and masks for GeoTIFF export):

* **Landmask tiles** (``grid_X.XX_Y.YY.tiff``):
  
  * Provide UTM projection information
  * Define precise geospatial transforms
  * Contain land/water masks

The geotessera CLI can also export these into GeoTIFF format with each band
dequantised into 128-bands and with the GeoTIFF CRS metadata intact.

Data Flow
~~~~~~~~~

::

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

Registry System
~~~~~~~~~~~~~~~

GeoTessera uses a hosted, versioned registry system for efficient data access:

* **Block-based organization**: Registry divided into 5×5 degree geographic blocks
* **Lazy loading**: Only loads registry blocks for the region you're accessing  
* **Automatic caching**: Downloads are cached locally using Pooch
* **Integrity checking**: SHA256 checksums ensure data integrity

The registry can be loaded from multiple sources:

1. **Local directory** (via ``--registry-dir`` or ``registry_dir`` parameter)
2. **Environment variable** (``TESSERA_REGISTRY_DIR``)
3. **Auto-cloned repository** (default, from GitHub)

Understanding Tessera Embeddings
--------------------------------

Each embedding tile:

* Covers a 0.1° × 0.1° area (approximately 11km × 11km at equator)
* Contains 128 channels of learned features per pixel
* Represents patterns from a full year of satellite observations
* Is stored in quantized format for efficient transmission and storage

The 128 channels capture various environmental features learned by the
Tessera foundation model, including vegetation patterns, water bodies,
urban structures, and seasonal changes.

Data Organization
-----------------

**Remote Server Structure**::

    dl-2.tessera.wiki/
    ├── v1/                              # Dataset version
    │   ├── 2024/                        # Year
    │   │   ├── grid_0.15_52.05/         # Tile (named by center coords)
    │   │   │   ├── grid_0.15_52.05.npy              # Quantized embeddings
    │   │   │   └── grid_0.15_52.05_scales.npy       # Scale factors
    │   │   └── ...
    │   └── landmasks/
    │       ├── grid_0.15_52.05.tiff     # Landmask with projection info
    │       └── ...

**Local Cache Structure**::

    ~/.cache/geotessera/                 # Default cache location
    ├── tessera-manifests/               # Auto-cloned registry
    │   └── registry/
    ├── pooch/                           # Downloaded data files
    │   ├── grid_0.15_52.05.npy
    │   ├── grid_0.15_52.05_scales.npy
    │   └── ...

Embeddings are organized by:

* **Year**: 2017-2024 (depending on availability)
* **Location**: Global 0.1-degree grid system
* **Format**: NumPy arrays with shape (height, width, 128)

Environment Configuration
-------------------------

You can configure GeoTessera using environment variables::

    # Set custom cache directory for downloaded files
    export TESSERA_DATA_DIR=/path/to/cache
    
    # Use local registry directory  
    export TESSERA_REGISTRY_DIR=/path/to/tessera-manifests
    
    # Or configure per-command
    TESSERA_DATA_DIR=/tmp/cache geotessera download ...

Documentation Sections
-----------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide:
   
   quickstart
   architecture
   tutorials
   cli_reference

.. toctree::
   :maxdepth: 2
   :caption: API Reference:
   
   modules

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources:
   
   GitHub Repository <https://github.com/ucam-eo/geotessera>
   Tessera Model <https://github.com/ucam-eo/tessera>
   Issue Tracker <https://github.com/ucam-eo/geotessera/issues>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
