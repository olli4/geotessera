Architecture Guide
=================

This guide explains the internal architecture of GeoTessera and how the various components work together to provide efficient access to Tessera embeddings.

Overview
--------

GeoTessera is designed around a simple but powerful architecture that optimizes for:

- **Efficient data access**: Only download what you need
- **Scalability**: Handle large datasets with lazy loading
- **Flexibility**: Support both analysis and GIS workflows
- **Reliability**: Ensure data integrity with checksums

Core Architecture
-----------------

The library follows a layered architecture:

.. code-block::

    User Interface Layer
    ├── CLI Commands (geotessera download, visualize, etc.)
    └── Python API (GeoTessera class)
            ↓
    Core Processing Layer  
    ├── GeoTessera class (main interface)
    ├── Registry (data discovery and metadata)
    └── Visualization (rendering and web maps)
            ↓
    Data Access Layer
    ├── Pooch (download and caching)
    ├── Rasterio (GeoTIFF I/O)
    └── GeoPandas (geospatial operations)
            ↓
    Storage Layer
    ├── Remote servers (dl-2.tessera.wiki)
    └── Local cache (~/.cache/geotessera)

Coordinate System and Grid
--------------------------

Understanding the Tessera Grid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Tessera embeddings are organized on a **0.1-degree grid system**:

**Grid Properties**:

- **Grid spacing**: 0.1° latitude × 0.1° longitude
- **Tile naming**: Named by center coordinates (e.g., ``grid_0.15_52.05``)
- **Coverage**: Each tile spans from (center - 0.05°) to (center + 0.05°)
- **Resolution**: Approximately 11km × 11km at the equator

**Coordinate Calculations**::

    # For a tile at center coordinates (lon, lat)
    west = lon - 0.05
    east = lon + 0.05  
    south = lat - 0.05
    north = lat + 0.05

**Grid Alignment**:

Tile centers are aligned to 0.1-degree boundaries::

    # Valid tile centers (examples)
    valid_centers = [
        (0.05, 52.05),   # Northwest Europe
        (0.15, 52.05),   # Adjacent tile
        (-0.05, 51.95),  # Southwest tile
    ]
    
    # Invalid centers (not on grid)
    invalid_centers = [
        (0.07, 52.03),   # Off-grid
        (0.1, 52.1),     # Off by 0.05°
    ]

Resolution and Pixel Density
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The number of pixels per tile varies with latitude due to the Earth's curvature:

.. code-block:: python

    import math
    
    def pixels_per_tile(latitude, resolution_meters=10):
        """Calculate approximate pixels per tile at given latitude."""
        # Earth circumference at equator (meters)
        earth_circumference = 40075000
        
        # Degrees per meter at equator
        degrees_per_meter = 360 / earth_circumference
        
        # Adjust for latitude (longitude only)
        lon_degrees_per_meter = degrees_per_meter / math.cos(math.radians(latitude))
        lat_degrees_per_meter = degrees_per_meter
        
        # Tile size in meters
        tile_width_meters = 0.1 / lon_degrees_per_meter
        tile_height_meters = 0.1 / lat_degrees_per_meter
        
        # Pixels in tile
        pixels_width = int(tile_width_meters / resolution_meters)
        pixels_height = int(tile_height_meters / resolution_meters)
        
        return pixels_width, pixels_height
    
    # Examples
    eq_pixels = pixels_per_tile(0)      # ~(1111, 1111) at equator
    uk_pixels = pixels_per_tile(52)     # ~(1823, 1111) in UK
    arctic_pixels = pixels_per_tile(80) # ~(6389, 1111) near poles

Data Format and Storage
-----------------------

Quantization System
~~~~~~~~~~~~~~~~~~~

Tessera embeddings are stored using a quantization system for efficiency:

**Storage Format**:

1. **Quantized embeddings** (``grid_X.XX_Y.YY.npy``):
   
   - Data type: ``int8`` (values -128 to 127)
   - Shape: ``(height, width, 128)``
   - Storage efficient: ~1MB per tile vs ~64MB unquantized

2. **Scale factors** (``grid_X.XX_Y.YY_scales.npy``):
   
   - Data type: ``float32``
   - Shape: ``(height, width)`` or ``(height, width, 128)``
   - Contains dequantization multipliers

**Dequantization Process**::

    import numpy as np
    
    # Load quantized data and scales
    quantized = np.load("grid_0.15_52.05.npy")         # int8
    scales = np.load("grid_0.15_52.05_scales.npy")     # float32
    
    # Dequantize
    if scales.ndim == 2:
        # Broadcast 2D scales to 3D
        scales = scales[..., np.newaxis]
    
    embedding = quantized.astype(np.float32) * scales
    
    # Result: (height, width, 128) float32 array

This process is handled automatically by ``GeoTessera.fetch_embedding()``.

Metadata and Projections
~~~~~~~~~~~~~~~~~~~~~~~~

**Landmask Files** (``grid_X.XX_Y.YY.tiff``):

- Provide UTM projection information for each tile
- Define precise geospatial transforms
- Used for georeferencing when exporting to GeoTIFF
- Contain binary land/water masks

**Projection Selection**:

Each tile uses an appropriate UTM zone based on its location::

    def get_utm_zone(longitude):
        """Get UTM zone number for a longitude."""
        return int((longitude + 180) / 6) + 1
    
    def get_utm_epsg(longitude, latitude):
        """Get EPSG code for UTM projection."""
        zone = get_utm_zone(longitude)
        
        if latitude >= 0:
            # Northern hemisphere
            return f"EPSG:{32600 + zone}"
        else:
            # Southern hemisphere  
            return f"EPSG:{32700 + zone}"
    
    # Example: London at (0.15, 52.05)
    epsg = get_utm_epsg(0.15, 52.05)  # "EPSG:32631" (UTM Zone 31N)

Registry System
---------------

Block-Based Organization
~~~~~~~~~~~~~~~~~~~~~~~~

The registry uses a **5×5 degree block system** for efficient data discovery:

**Block Structure**:

.. code-block::

    Registry Blocks (5° × 5°)
    ├── Block (-5°, 50°) to (0°, 55°)     # Western Europe
    │   ├── embeddings_2024_lon-5_lat50.txt
    │   └── landmasks_lon-5_lat50.txt
    ├── Block (0°, 50°) to (5°, 55°)      # Central Europe  
    │   ├── embeddings_2024_lon0_lat50.txt
    │   └── landmasks_lon0_lat50.txt
    └── ...

**Block Coordinate Calculation**::

    def get_block_coordinates(lon, lat):
        """Get the 5x5 degree block coordinates for a point."""
        # Round down to nearest 5-degree boundary
        block_lon = int(lon // 5) * 5
        block_lat = int(lat // 5) * 5
        return block_lon, block_lat
    
    # Examples
    london_block = get_block_coordinates(0.15, 52.05)    # (0, 50)
    paris_block = get_block_coordinates(2.35, 48.86)     # (0, 45)  
    sydney_block = get_block_coordinates(151.21, -33.87) # (150, -35)

Registry File Format
~~~~~~~~~~~~~~~~~~~~

Each registry file uses the Pooch format::

    # Format: filepath checksum
    2024/grid_0.15_52.05/grid_0.15_52.05.npy sha256:abc123def456...
    2024/grid_0.15_52.05/grid_0.15_52.05_scales.npy sha256:def456abc123...
    landmasks/grid_0.15_52.05.tiff sha256:789abc456def...

**Registry Loading Process**:

1. **Determine required blocks** for the requested bounding box
2. **Load block registry files** (only the needed ones)
3. **Parse available tiles** within the requested region
4. **Cache registry data** for subsequent requests

Lazy Loading Strategy
~~~~~~~~~~~~~~~~~~~~~

GeoTessera uses lazy loading to minimize memory usage and startup time:

.. code-block:: python

    class Registry:
        def __init__(self):
            self._loaded_blocks = set()      # Track loaded blocks
            self._available_embeddings = []  # Cached tile list
        
        def load_blocks_for_region(self, bbox, year):
            """Load only the blocks needed for this region."""
            required_blocks = self._get_blocks_in_bbox(bbox)
            
            for block_coords in required_blocks:
                if (year, *block_coords) not in self._loaded_blocks:
                    self._load_block_registry(year, block_coords)
                    self._loaded_blocks.add((year, *block_coords))
        
        def ensure_all_blocks_loaded(self):
            """Load all blocks for global operations (coverage maps)."""
            # Only called when needed for complete coverage

Registry Sources
~~~~~~~~~~~~~~~~

The registry can be loaded from multiple sources:

**1. Auto-cloned Repository** (default)::

    ~/.cache/geotessera/tessera-manifests/
    └── registry/
        ├── embeddings/
        └── landmasks/

**2. Environment Variable**::

    export TESSERA_REGISTRY_DIR=/path/to/tessera-manifests
    geotessera download ...

**3. Explicit Parameter**::

    from geotessera import GeoTessera
    gt = GeoTessera(registry_dir="/path/to/tessera-manifests")

**4. Remote Fallback**:

If no local registry is available, individual registry files are downloaded on-demand.

Data Access Layer
-----------------

Pooch Integration
~~~~~~~~~~~~~~~~~

GeoTessera uses `Pooch <https://www.fatiando.org/pooch/>`_ for robust data downloading:

**Features**:

- **Automatic caching**: Files cached after first download
- **Integrity checking**: SHA256 verification
- **Progress bars**: Visual download feedback  
- **Retry logic**: Handles network issues
- **Concurrent downloads**: Parallel fetching when possible

**Cache Structure**::

    ~/.cache/geotessera/
    ├── tessera-manifests/           # Registry repository
    ├── pooch/                       # Downloaded embeddings
    │   ├── 2024/
    │   │   └── grid_0.15_52.05/
    │   │       ├── grid_0.15_52.05.npy
    │   │       └── grid_0.15_52.05_scales.npy
    │   └── landmasks/
    │       └── grid_0.15_52.05.tiff
    └── geodatasets/                 # World map data

**Download Process**::

    # Simplified download workflow
    def fetch_embedding(lat, lon, year):
        # 1. Ensure registry block is loaded
        registry.ensure_block_loaded(year, lon, lat)
        
        # 2. Get file paths from registry
        embedding_path, scales_path = get_tile_paths(lat, lon, year)
        
        # 3. Download files via Pooch (cached)
        embedding_file = pooch.fetch(embedding_path)
        scales_file = pooch.fetch(scales_path)
        
        # 4. Load and dequantize
        quantized = np.load(embedding_file)
        scales = np.load(scales_file)
        return quantized.astype(np.float32) * scales

Caching Strategy
~~~~~~~~~~~~~~~~

**Cache Hierarchy**:

1. **Memory cache**: Recently accessed embeddings kept in RAM
2. **Disk cache**: Downloaded files persist across sessions
3. **Registry cache**: Loaded registry data cached in memory

**Cache Management**::

    # Cache locations (configurable)
    data_cache = os.environ.get('TESSERA_DATA_DIR', 
                               platformdirs.user_cache_dir('geotessera'))
    
    # Automatic cleanup (if needed)
    def cleanup_cache(max_size_gb=10):
        """Remove oldest files if cache exceeds size limit."""
        # Implementation would check file sizes and modification times

GeoTIFF Export Process
~~~~~~~~~~~~~~~~~~~~~~

When exporting to GeoTIFF, additional processing occurs:

**Export Workflow**:

1. **Fetch embedding data** (quantized + scales)
2. **Fetch landmask tile** for projection information
3. **Extract UTM projection** from landmask
4. **Apply dequantization** to embedding data
5. **Select bands** (if specified)
6. **Write GeoTIFF** with proper geotransform and CRS
7. **Apply compression** (LZW, DEFLATE, etc.)

**Projection Inheritance**::

    import rasterio
    
    def export_geotiff(embedding, landmask_path, output_path, bands=None):
        # Read projection from landmask
        with rasterio.open(landmask_path) as landmask:
            crs = landmask.crs
            transform = landmask.transform
            
        # Select bands
        if bands:
            embedding = embedding[:, :, bands]
            
        # Write GeoTIFF
        with rasterio.open(output_path, 'w',
                          driver='GTiff',
                          height=embedding.shape[0],
                          width=embedding.shape[1], 
                          count=embedding.shape[2],
                          dtype=embedding.dtype,
                          crs=crs,
                          transform=transform,
                          compress='lzw') as dst:
            
            for i in range(embedding.shape[2]):
                dst.write(embedding[:, :, i], i + 1)

Performance Considerations
--------------------------

Memory Management
~~~~~~~~~~~~~~~~~

**Large Region Handling**:

When processing large regions, GeoTessera uses several strategies:

- **Tile-by-tile processing**: Process one tile at a time to limit memory usage
- **Band selection**: Only load required bands to reduce memory footprint  
- **Generator patterns**: Use generators for large tile collections
- **Progress callbacks**: Provide feedback for long operations

**Example Memory-Efficient Processing**::

    def process_large_region(bbox, year, bands=None):
        """Process a large region without loading all tiles into memory."""
        gt = GeoTessera()
        
        # Get tile list (metadata only)
        tiles = gt.registry.load_blocks_for_region(bbox, year)
        
        for tile_lat, tile_lon in tiles:
            # Process one tile at a time
            embedding = gt.fetch_embedding(tile_lat, tile_lon, year)
            
            # Apply band selection early
            if bands:
                embedding = embedding[:, :, bands]
                
            # Process this tile
            result = process_single_tile(embedding)
            
            # Save or accumulate results
            save_tile_result(result, tile_lat, tile_lon)
            
            # Free memory
            del embedding

Network Optimization
~~~~~~~~~~~~~~~~~~~~

**Concurrent Downloads**:

For multiple tiles, downloads can be parallelized::

    import concurrent.futures
    
    def download_tiles_parallel(tile_coords, year, max_workers=4):
        """Download multiple tiles in parallel."""
        gt = GeoTessera()
        
        def download_single(coords):
            lat, lon = coords
            return gt.fetch_embedding(lat, lon, year)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            embeddings = list(executor.map(download_single, tile_coords))
        
        return embeddings

**Cache Efficiency**:

- **Pre-warming**: Download commonly used tiles in advance
- **Batch processing**: Group requests by geographic region
- **Size limits**: Respect server rate limits

Future Extensions
~~~~~~~~~~~~~~~~~

The architecture supports future enhancements:

- **Temporal queries**: Multi-year analysis
- **Cloud optimization**: Direct cloud storage access
- **ML integration**: TensorFlow/PyTorch data loaders
- **Real-time updates**: Live data ingestion
- **Distributed processing**: Dask/Ray integration
