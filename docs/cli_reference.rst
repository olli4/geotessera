CLI Reference
=============

GeoTessera provides a comprehensive command-line interface for downloading, visualizing, and serving Tessera embeddings.

Global Options
--------------

All commands support these global options::

    --dataset-version TEXT    Tessera dataset version (default: v1)
    --cache-dir PATH         Custom cache directory
    --registry-dir PATH      Custom registry directory
    --verbose, -v            Enable verbose output
    --help                   Show help message

Environment Variables
---------------------

Configure GeoTessera using environment variables:

.. code-block:: bash

    # Set custom cache directory
    export TESSERA_DATA_DIR=/path/to/cache
    
    # Use local registry directory
    export TESSERA_REGISTRY_DIR=/path/to/tessera-manifests
    
    # Use per-command
    TESSERA_DATA_DIR=/tmp/cache geotessera download ...

Commands
--------

download
~~~~~~~~

Download embeddings for a region in numpy or GeoTIFF format.

**Usage**::

    geotessera download [OPTIONS]

**Required Options**:

* ``-o, --output PATH`` - Output directory [required]

**Region Definition** (one required):

* ``--bbox TEXT`` - Bounding box: 'min_lon,min_lat,max_lon,max_lat'
* ``--region-file PATH`` - GeoJSON/Shapefile to define region

**Format Options**:

* ``-f, --format TEXT`` - Output format: 'tiff' or 'npy' (default: tiff)
* ``--bands TEXT`` - Comma-separated band indices (default: all 128)
* ``--compress TEXT`` - Compression for TIFF format (default: lzw)

**Data Selection**:

* ``--year INT`` - Year of embeddings (default: 2024)

**Other Options**:

* ``--list-files`` - List all created files with details
* ``-v, --verbose`` - Verbose output

**Examples**::

    # Download as GeoTIFF (georeferenced, for GIS)
    geotessera download \
        --bbox "-0.2,51.4,0.1,51.6" \
        --year 2024 \
        --output ./london_tiffs

    # Download as numpy arrays (for analysis)
    geotessera download \
        --bbox "-0.2,51.4,0.1,51.6" \
        --format npy \
        --year 2024 \
        --output ./london_arrays

    # Download specific bands only
    geotessera download \
        --bbox "-0.2,51.4,0.1,51.6" \
        --bands "0,1,2,10,20,30" \
        --year 2024 \
        --output ./london_subset

    # Download using a region file
    geotessera download \
        --region-file cambridge.geojson \
        --format tiff \
        --year 2024 \
        --output ./cambridge_tiles

**Output Formats**:

**TIFF Format** (``--format tiff``):
    - Creates georeferenced GeoTIFF files
    - Each tile preserves its native UTM projection
    - Suitable for GIS software (QGIS, ArcGIS, etc.)
    - Supports compression (lzw, deflate, none)
    - Files named by tile coordinates (e.g., ``grid_51.45_-0.05.tif``)

**NPY Format** (``--format npy``):
    - Creates raw numpy arrays (.npy files)
    - Includes metadata.json with tile information
    - Suitable for direct analysis in Python
    - Smaller file sizes than GeoTIFF
    - Files named by coordinates (e.g., ``embedding_51.45_-0.05.npy``)

visualize
~~~~~~~~~

Create visualizations from GeoTIFF files.

**Usage**::

    geotessera visualize INPUT_PATH [OPTIONS]

**Required Arguments**:

* ``INPUT_PATH`` - Path to GeoTIFF file or directory containing GeoTIFFs

**Required Options**:

* ``-o, --output PATH`` - Output directory

**Visualization Options**:

* ``--type TEXT`` - Visualization type: rgb, web, coverage (default: rgb)
* ``--bands TEXT`` - Comma-separated band indices for RGB (default: "0,1,2")
* ``--normalize`` - Normalize bands to 0-255 range

**Web Tile Options**:

* ``--min-zoom INT`` - Minimum zoom level for web tiles (default: 8)
* ``--max-zoom INT`` - Maximum zoom level for web tiles (default: 15)
* ``--initial-zoom INT`` - Initial zoom level for viewer (default: 10)
* ``--force`` - Force regeneration of existing tiles

**Examples**::

    # Create RGB visualization from first 3 bands
    geotessera visualize \
        ./london_tiffs \
        --type rgb \
        --output ./london_rgb

    # Create RGB with custom bands
    geotessera visualize \
        ./london_tiffs \
        --type rgb \
        --bands "30,60,90" \
        --normalize \
        --output ./london_custom_rgb

    # Generate interactive web map
    geotessera visualize \
        ./london_tiffs \
        --type web \
        --min-zoom 8 \
        --max-zoom 15 \
        --output ./london_web

    # Force regeneration of web tiles
    geotessera visualize \
        ./london_tiffs \
        --type web \
        --force \
        --output ./london_web

    # Create coverage map from GeoTIFFs
    geotessera visualize \
        ./london_tiffs \
        --type coverage \
        --output ./london_coverage

**Visualization Types**:

**RGB** (``--type rgb``):
    - Creates RGB composite images
    - Merges multiple GeoTIFF tiles into a single mosaic
    - Output: Single GeoTIFF file (``rgb_mosaic.tif``)
    - Use ``--bands`` to specify which channels to use as R, G, B
    - Use ``--normalize`` to stretch values to 0-255 range

**Web** (``--type web``):
    - Generates web map tiles for interactive viewing
    - Creates Leaflet-compatible tile pyramid
    - Output: Directory with tiles and HTML viewer
    - Use ``--min-zoom`` and ``--max-zoom`` to control detail levels
    - Automatically creates ``viewer.html`` for viewing

**Coverage** (``--type coverage``):
    - Creates HTML map showing tile coverage
    - Shows spatial extent of available data
    - Output: Interactive HTML map
    - Useful for understanding data distribution

serve
~~~~~

Serve web visualizations locally with a built-in HTTP server.

**Usage**::

    geotessera serve DIRECTORY [OPTIONS]

**Required Arguments**:

* ``DIRECTORY`` - Directory containing web visualization files

**Options**:

* ``-p, --port INT`` - Port number for web server (default: 8000)
* ``--open/--no-open`` - Auto-open browser (default: open)
* ``--html TEXT`` - Specific HTML file to serve

**Examples**::

    # Serve web visualization and open browser
    geotessera serve ./london_web --open

    # Serve on specific port
    geotessera serve ./london_web --port 8080

    # Serve specific HTML file
    geotessera serve ./visualizations --html coverage.html

    # Serve without auto-opening browser
    geotessera serve ./london_web --no-open

**Notes**:
    - The server automatically finds HTML files (index.html, viewer.html, etc.)
    - Use Ctrl+C to stop the server
    - The server serves all files in the directory
    - Required for viewing Leaflet-based web maps

coverage
~~~~~~~~

Generate a world map showing Tessera embedding coverage.

**Usage**::

    geotessera coverage [OPTIONS]

**Output Options**:

* ``-o, --output PATH`` - Output PNG file path (default: tessera_coverage.png)

**Data Selection**:

* ``--year INT`` - Specific year to visualize (default: all years)

**Visualization Options**:

* ``--tile-color TEXT`` - Color for tile rectangles (default: red)
* ``--tile-alpha FLOAT`` - Transparency of tiles 0.0-1.0 (default: 0.6)
* ``--tile-size FLOAT`` - Size multiplier for tiles (default: 1.0)

**Map Options**:

* ``--width INT`` - Figure width in inches (default: 20)
* ``--height INT`` - Figure height in inches (default: 10)
* ``--dpi INT`` - Output resolution in dots per inch (default: 100)
* ``--no-countries`` - Don't show country boundaries

**Examples**::

    # Generate global coverage map
    geotessera coverage --output global_coverage.png

    # Show coverage for specific year
    geotessera coverage \
        --year 2024 \
        --output coverage_2024.png

    # Customize visualization
    geotessera coverage \
        --year 2024 \
        --tile-color blue \
        --tile-alpha 0.3 \
        --tile-size 1.2 \
        --dpi 150 \
        --width 24 \
        --height 12 \
        --output high_res_coverage.png

    # Map without country boundaries
    geotessera coverage \
        --no-countries \
        --tile-color green \
        --output coverage_clean.png

**Output**:
    - High-resolution PNG world map
    - Red rectangles show available tile locations
    - Each rectangle represents one 0.1° × 0.1° tile
    - Statistics shown in top-left corner
    - Legend indicates available tiles and land masses

info
~~~~

Display information about GeoTIFF files or the library.

**Usage**::

    geotessera info [OPTIONS]

**Options**:

* ``--geotiffs PATH`` - Analyze GeoTIFF files/directory
* ``--dataset-version TEXT`` - Tessera dataset version (default: v1)
* ``-v, --verbose`` - Verbose output

**Examples**::

    # Show library information
    geotessera info

    # Analyze GeoTIFF files
    geotessera info --geotiffs ./london_tiffs

    # Analyze single GeoTIFF file
    geotessera info --geotiffs ./london_tiffs/grid_51.45_-0.05.tif

    # Verbose library info
    geotessera info --verbose

**Output for Library Info**:
    - GeoTessera version
    - Available years in dataset
    - Registry information
    - Loaded blocks count

**Output for GeoTIFF Analysis**:
    - Total files analyzed
    - Years covered
    - Coordinate reference systems used
    - Bounding box of all files
    - Band count statistics
    - Individual tile information (with ``--verbose``)

Common Workflows
----------------

Basic Download and View
~~~~~~~~~~~~~~~~~~~~~~~

Complete workflow from download to visualization::

    # 1. Check data availability
    geotessera coverage --year 2024 --output coverage.png

    # 2. Download data
    geotessera download \
        --bbox "-0.2,51.4,0.1,51.6" \
        --year 2024 \
        --output ./london_data

    # 3. Create web visualization  
    geotessera visualize \
        ./london_data \
        --type web \
        --output ./london_web

    # 4. Serve and view
    geotessera serve ./london_web --open

Analysis Workflow
~~~~~~~~~~~~~~~~~

Download for analysis purposes::

    # 1. Download as numpy arrays
    geotessera download \
        --bbox "-0.1,52.0,0.1,52.2" \
        --format npy \
        --year 2024 \
        --output ./cambridge_analysis

    # 2. Check what was downloaded
    geotessera info --geotiffs ./cambridge_analysis

    # 3. Process in Python
    python your_analysis_script.py

    # 4. Export results as GeoTIFF for visualization
    geotessera download \
        --bbox "-0.1,52.0,0.1,52.2" \
        --format tiff \
        --bands "0,1,2" \
        --year 2024 \
        --output ./cambridge_viz

    # 5. Create web map
    geotessera visualize \
        ./cambridge_viz \
        --type web \
        --output ./cambridge_web

GIS Workflow
~~~~~~~~~~~~

Prepare data for GIS software::

    # 1. Download with specific bands for analysis
    geotessera download \
        --region-file study_area.geojson \
        --bands "10,20,30,40,50" \
        --format tiff \
        --compress lzw \
        --year 2024 \
        --output ./gis_data

    # 2. Create RGB composite for visualization
    geotessera visualize \
        ./gis_data \
        --type rgb \
        --bands "0,1,2" \
        --normalize \
        --output ./gis_rgb

    # 3. Analyze files before importing to GIS
    geotessera info --geotiffs ./gis_data --verbose

    # Files are now ready for QGIS, ArcGIS, etc.

Troubleshooting
---------------

Common Issues and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**"No tiles found in region"**:
    - Check coverage map first: ``geotessera coverage --year 2024``
    - Verify bounding box format: ``min_lon,min_lat,max_lon,max_lat``
    - Try a different year or larger region

**Slow downloads**:
    - Files are cached after first download
    - Use ``--verbose`` to see download progress
    - Check network connection

**Web visualization not working**:
    - Use ``geotessera serve`` instead of opening HTML directly
    - Check that tiles directory was created
    - Try ``--force`` to regenerate tiles

**Memory issues with large regions**:
    - Download smaller regions at a time
    - Use ``--bands`` to download only needed channels
    - Use ``npy`` format for smaller file sizes

**Permission errors**:
    - Check write permissions for output directory
    - Try using a different output directory
    - Set custom cache directory: ``--cache-dir /tmp/geotessera``

**GeoTIFF projection issues**:
    - Files use UTM projection (varies by location)
    - Most GIS software handles reprojection automatically
    - Use ``geotessera info --geotiffs`` to check CRS

Getting Help
~~~~~~~~~~~~

For additional help::

    # Command-specific help
    geotessera download --help
    geotessera visualize --help

    # Version information
    geotessera --version

    # Library information
    geotessera info --verbose

**Resources**:
    - GitHub Issues: https://github.com/ucam-eo/geotessera/issues
    - Documentation: https://geotessera.readthedocs.io/
    - Examples: See tutorials section of documentation