geotessera package
==================

The GeoTessera package provides streamlined access to Tessera geospatial embeddings through a clean Python API and comprehensive CLI.

Package Overview
----------------

.. automodule:: geotessera
   :members:
   :show-inheritance:
   :undoc-members:

Core Modules
------------

geotessera.core module
~~~~~~~~~~~~~~~~~~~~~~

The main interface for accessing Tessera embeddings. Contains the primary GeoTessera class with methods for fetching embeddings with CRS information and exporting to various formats while preserving native UTM projections.

.. automodule:: geotessera.core
   :members:
   :show-inheritance:
   :undoc-members:

geotessera.registry module
~~~~~~~~~~~~~~~~~~~~~~~~~~

Registry management for efficient data discovery and access. Handles the block-based registry system, lazy loading, and metadata management.

.. automodule:: geotessera.registry
   :members:
   :show-inheritance:
   :undoc-members:

geotessera.visualization module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Visualization utilities for creating maps, web tiles, and interactive visualizations from GeoTIFF files.

.. automodule:: geotessera.visualization
   :members:
   :show-inheritance:
   :undoc-members:

Command Line Interface
----------------------

geotessera.cli module
~~~~~~~~~~~~~~~~~~~~~

Comprehensive command-line interface providing download, visualization, and serving capabilities.

.. automodule:: geotessera.cli
   :members:
   :show-inheritance:
   :undoc-members:

Key Classes and Functions
-------------------------

GeoTessera Class
~~~~~~~~~~~~~~~~

The main interface for accessing Tessera embeddings:

.. autoclass:: geotessera.GeoTessera
   :members:
   :show-inheritance:

Key methods:

* :meth:`~geotessera.GeoTessera.fetch_embedding` - Fetch a single embedding tile with CRS and transform
* :meth:`~geotessera.GeoTessera.fetch_embeddings` - Fetch multiple tiles in a bounding box with projection info
* :meth:`~geotessera.GeoTessera.export_embedding_geotiff` - Export single embedding as GeoTIFF with native UTM
* :meth:`~geotessera.GeoTessera.export_embedding_geotiffs` - Export multiple embeddings as GeoTIFF files

Registry Class
~~~~~~~~~~~~~~

Manages data discovery and access:

.. autoclass:: geotessera.registry.Registry
   :members:
   :show-inheritance:

Key methods:

* :meth:`~geotessera.registry.Registry.get_available_embeddings` - List all available tiles
* :meth:`~geotessera.registry.Registry.ensure_all_blocks_loaded` - Load complete registry for coverage maps
* :meth:`~geotessera.registry.Registry.load_blocks_for_region` - Load registry for specific region

Visualization Functions
~~~~~~~~~~~~~~~~~~~~~~~

Main visualization functions:

.. autofunction:: geotessera.visualization.visualize_global_coverage

.. autofunction:: geotessera.visualization.create_rgb_mosaic_from_geotiffs

.. autofunction:: geotessera.visualization.geotiff_to_web_tiles

.. autofunction:: geotessera.visualization.create_coverage_summary_map

Utility Functions
~~~~~~~~~~~~~~~~~

Registry utility functions:

.. autofunction:: geotessera.registry.get_tile_bounds

.. autofunction:: geotessera.registry.world_to_tile_coords

.. autofunction:: geotessera.registry.get_block_coordinates

Examples
--------

Basic Usage
~~~~~~~~~~~

Initialize and fetch embeddings::

    from geotessera import GeoTessera
    
    # Initialize client
    gt = GeoTessera()
    
    # Fetch single tile with CRS information
    embedding, crs, transform = gt.fetch_embedding(lon=0.15, lat=52.05, year=2024)
    print(f"CRS: {crs}")  # Native UTM projection
    
    # Fetch region with projection info
    bbox = (-0.2, 51.4, 0.1, 51.6)
    tiles = gt.fetch_embeddings(bbox, year=2024)
    
    for tile_lon, tile_lat, embedding, crs, transform in tiles:
        print(f"Tile ({tile_lon}, {tile_lat}): {embedding.shape}, CRS: {crs}")

Export to GeoTIFF
~~~~~~~~~~~~~~~~~

Export embeddings for GIS use with preserved projections::

    # Export all bands with native UTM projections
    files = gt.export_embedding_geotiffs(
        bbox=bbox,
        output_dir="./output",
        year=2024
    )
    
    # Export specific bands
    rgb_files = gt.export_embedding_geotiffs(
        bbox=bbox,
        output_dir="./rgb_output", 
        year=2024,
        bands=[0, 1, 2]  # Each tile preserves its native UTM projection
    )
    
    # Export single tile  
    single_file = gt.export_embedding_geotiff(
        lon=0.15, lat=52.05,
        output_path="./single_tile.tif",
        year=2024,
        bands=[10, 20, 30]
    )

Create Visualizations
~~~~~~~~~~~~~~~~~~~~~

Generate visualizations::

    from geotessera.visualization import (
        create_rgb_mosaic_from_geotiffs,
        visualize_global_coverage
    )
    
    # Create RGB mosaic
    create_rgb_mosaic_from_geotiffs(
        geotiff_paths=rgb_files,
        output_path="mosaic.tif",
        bands=(0, 1, 2),
        normalize=True
    )
    
    # Create coverage map
    visualize_global_coverage(
        tessera_client=gt,
        output_path="coverage.png",
        year=2024
    )