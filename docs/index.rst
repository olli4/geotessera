GeoTessera Documentation
========================

GeoTessera provides Python access to pre-computed geospatial embeddings from the 
`Tessera foundation model <https://github.com/ucam-eo/tessera>`_. Tessera processes 
Sentinel-1 and Sentinel-2 satellite imagery to generate 128-dimensional representation 
maps at 10m spatial resolution, compressing a full year of temporal-spectral features 
into dense geospatial embeddings.

Key Features
------------

* **Global Coverage**: Access embeddings for any location worldwide (where data exists)
* **High Resolution**: 10m spatial resolution preserving fine-grained details
* **Temporal Compression**: Full year of satellite observations in each embedding
* **Multi-spectral**: Combines Sentinel-1 SAR and Sentinel-2 optical data
* **Efficient Storage**: Quantized embeddings with automatic dequantization
* **Easy Access**: Simple Python API with automatic data fetching and caching

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

Basic usage example::

    from geotessera import GeoTessera
    
    # Initialize client
    gt = GeoTessera()
    
    # Fetch embeddings for a location (Cambridge, UK)
    embedding = gt.fetch_embedding(lat=52.2053, lon=0.1218)
    print(f"Embedding shape: {embedding.shape}")  # (height, width, 128)
    
    # Create false-color visualization
    gt.export_single_tile_as_tiff(
        lat=52.20, lon=0.10,
        output_path="cambridge.tiff",
        bands=[10, 20, 30]  # Select 3 channels for RGB
    )

Command Line Usage
------------------

GeoTessera includes a comprehensive CLI::

    # List available embeddings
    geotessera list-embeddings --limit 10
    
    # Create visualization for a region
    geotessera visualize --region region.json --output viz.tiff
    
    # Launch interactive web map
    geotessera serve --region boundary.json --open

Understanding Tessera Embeddings
--------------------------------

Each embedding tile:

* Covers a 0.1° × 0.1° area (approximately 11km × 11km at equator)
* Contains 128 channels of learned features per pixel
* Represents patterns from a full year of satellite observations
* Is stored in quantized format for efficient transmission

The 128 channels capture various environmental features learned by the
Tessera foundation model, including vegetation patterns, water bodies,
urban structures, and seasonal changes.

Data Organization
-----------------

Embeddings are organized by:

* **Year**: 2017-2024 (depending on availability)
* **Location**: Global 0.1-degree grid system
* **Format**: NumPy arrays with shape (height, width, 128)

Files are fetched on-demand from the Tessera servers via
HTTPS and cached locally for subsequent use.

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

