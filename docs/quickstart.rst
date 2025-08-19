Quick Start Guide
=================

This guide will get you up and running with GeoTessera quickly.

Installation
------------

Install GeoTessera using pip::

    pip install geotessera

Verify the installation::

    geotessera --help

Step 1: Check Data Availability
--------------------------------

Before downloading embeddings, it's useful to check what data is available for your region of interest.

Generate a global coverage map::

    geotessera coverage --output global_coverage.png

This creates a world map showing all available embedding tiles as red rectangles.

For a specific year::

    geotessera coverage --year 2024 --output coverage_2024.png

You can customize the visualization::

    geotessera coverage \
        --year 2024 \
        --tile-color blue \
        --tile-alpha 0.3 \
        --dpi 150 \
        --width 24 \
        --height 12 \
        --output high_res_coverage.png

Step 2: Download Embeddings
----------------------------

GeoTessera supports two output formats:

- **tiff**: Georeferenced GeoTIFF files (default, best for GIS)
- **npy**: Raw numpy arrays with metadata (best for analysis)

Download as GeoTIFF (Recommended for GIS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Download embeddings for London as GeoTIFF files::

    geotessera download \
        --bbox "-0.2,51.4,0.1,51.6" \
        --year 2024 \
        --output ./london_tiles

This downloads all 128 bands with LZW compression.

Download specific bands only::

    geotessera download \
        --bbox "-0.2,51.4,0.1,51.6" \
        --bands "0,1,2" \
        --year 2024 \
        --output ./london_rgb

Download by country name::

    geotessera download \
        --country "United Kingdom" \
        --year 2024 \
        --output ./uk_tiles

    # Or use short country codes
    geotessera download \
        --country "GB" \
        --year 2024 \
        --output ./uk_tiles

Download using a region file::

    # Create a GeoJSON file defining your region
    cat > cambridge.json << EOF
    {
      "type": "Polygon",
      "coordinates": [[
        [-0.2, 51.9], [0.3, 51.9], [0.3, 52.3], [-0.2, 52.3], [-0.2, 51.9]
      ]]
    }
    EOF
    
    geotessera download \
        --region-file cambridge.json \
        --year 2024 \
        --output ./cambridge_tiles

Download as NumPy Arrays (For Analysis)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Download raw numpy arrays with metadata::

    geotessera download \
        --bbox "-0.2,51.4,0.1,51.6" \
        --format npy \
        --year 2024 \
        --output ./london_arrays

This creates:

- Individual ``.npy`` files for each tile (e.g., ``embedding_51.45_-0.15.npy``)
- A ``metadata.json`` file with tile coordinates, shapes, and band information

Step 3: Work with the Data
---------------------------

Python API Examples
~~~~~~~~~~~~~~~~~~~

Initialize the client::

    from geotessera import GeoTessera
    import numpy as np
    
    gt = GeoTessera()

Fetch a single embedding tile with CRS information::

    # Fetch embedding for Cambridge, UK (note: lon, lat order)
    embedding, crs, transform = gt.fetch_embedding(lon=0.15, lat=52.05, year=2024)
    print(f"Shape: {embedding.shape}")  # e.g., (1200, 1200, 128)
    print(f"Data type: {embedding.dtype}")  # float32
    print(f"CRS: {crs}")  # UTM projection from landmask
    print(f"Transform: {transform}")  # Geospatial transform
    print(f"Value range: [{embedding.min():.2f}, {embedding.max():.2f}]")

Fetch multiple tiles in a bounding box::

    bbox = (-0.2, 51.4, 0.1, 51.6)  # (min_lon, min_lat, max_lon, max_lat)
    tiles = gt.fetch_embeddings(bbox, year=2024)
    
    for tile_lon, tile_lat, embedding_array, crs, transform in tiles:
        print(f"Tile ({tile_lon}, {tile_lat}): {embedding_array.shape}")
        print(f"  CRS: {crs}")
        
        # Compute basic statistics
        mean_values = np.mean(embedding_array, axis=(0, 1))  # Mean per channel
        print(f"  Mean of first 5 channels: {mean_values[:5]}")

Export embeddings to GeoTIFF::

    files = gt.export_embedding_geotiffs(
        bbox=bbox,
        output_dir="./output",
        year=2024,
        bands=[10, 30, 50],  # Custom band selection
        compress="lzw"
    )
    print(f"Created {len(files)} GeoTIFF files")

Working with Downloaded NumPy Arrays
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Load and analyze downloaded numpy arrays::

    import json
    import numpy as np
    
    # Load metadata
    with open("london_arrays/metadata.json", "r") as f:
        metadata = json.load(f)
    
    print(f"Downloaded {len(metadata['tiles'])} tiles")
    print(f"Bounding box: {metadata['bbox']}")
    print(f"Year: {metadata['year']}")
    
    # Load and process each tile
    for tile_info in metadata["tiles"]:
        lat, lon = tile_info["lat"], tile_info["lon"]
        filename = tile_info["filename"]
        
        # Load the numpy array
        embedding = np.load(f"london_arrays/{filename}")
        
        # Perform analysis
        print(f"Tile ({lat}, {lon}):")
        print(f"  Shape: {embedding.shape}")
        print(f"  Mean per band (first 5): {np.mean(embedding, axis=(0,1))[:5]}")
        
        # Extract center pixel features
        center_pixel = embedding[embedding.shape[0]//2, embedding.shape[1]//2, :]
        print(f"  Center pixel features (first 5): {center_pixel[:5]}")

Step 4: Create Visualizations
------------------------------

Create an RGB Mosaic
~~~~~~~~~~~~~~~~~~~~~

From GeoTIFF files::

    geotessera visualize \
        ./london_tiles \
        --type rgb \
        --bands "30,60,90" \
        --normalize \
        --output ./london_rgb

This creates an RGB visualization using bands 30, 60, and 90 as red, green, and blue channels.

Create Interactive Web Maps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate web tiles and an interactive map::

    geotessera visualize \
        ./london_tiles \
        --type web \
        --min-zoom 8 \
        --max-zoom 15 \
        --output ./london_web

Serve the interactive map locally::

    geotessera serve ./london_web --open

This starts a local web server and opens the interactive map in your browser.

Advanced Options
~~~~~~~~~~~~~~~~

Force regeneration of web tiles::

    geotessera visualize \
        ./london_tiles \
        --type web \
        --force \
        --output ./london_web

Use specific bands for RGB::

    geotessera visualize \
        ./london_tiles \
        --type rgb \
        --bands "0,1,2" \
        --output ./london_rgb_012

Step 5: Advanced Workflows
---------------------------

Python Analysis Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~

Complete analysis workflow::

    from geotessera import GeoTessera
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Initialize client
    gt = GeoTessera()
    
    # Define region of interest
    bbox = (-0.15, 52.15, 0.0, 52.25)  # Cambridge area
    
    # Fetch embeddings
    embeddings = gt.fetch_embeddings(bbox, year=2024)
    
    # Analyze each tile
    results = []
    for lon, lat, embedding, crs, transform in embeddings:
        # Compute statistics
        mean_per_band = np.mean(embedding, axis=(0, 1))
        std_per_band = np.std(embedding, axis=(0, 1))
        
        results.append({
            'lat': lat,
            'lon': lon,
            'mean_band_50': mean_per_band[50],
            'std_band_50': std_per_band[50],
            'total_variance': np.var(embedding),
            'crs': str(crs)
        })
    
    # Print results
    for result in results:
        print(f"Tile ({result['lat']:.2f}, {result['lon']:.2f}): "
              f"Band 50 mean={result['mean_band_50']:.3f}, "
              f"variance={result['total_variance']:.3f}")
    
    # Export interesting tiles as GeoTIFF
    threshold = np.median([r['mean_band_50'] for r in results])
    selected_tiles = [r for r in results if r['mean_band_50'] > threshold]
    
    print(f"Exporting {len(selected_tiles)} tiles above threshold")
    
    files = []
    for tile in selected_tiles:
        file = gt.export_embedding_geotiffs(
            bbox=(tile['lon']-0.05, tile['lat']-0.05, 
                  tile['lon']+0.05, tile['lat']+0.05),
            output_dir="./selected_tiles",
            year=2024,
            bands=[40, 50, 60]  # Bands around band 50
        )
        files.extend(file)

Mixed Format Workflow
~~~~~~~~~~~~~~~~~~~~~

Use both numpy and GeoTIFF formats in the same workflow::

    from geotessera import GeoTessera
    from geotessera.visualization import create_rgb_mosaic_from_geotiffs
    
    gt = GeoTessera()
    bbox = (-0.1, 51.5, 0.0, 51.55)
    
    # Step 1: Analyze with numpy arrays
    print("Analyzing embeddings...")
    tiles = gt.fetch_embeddings(bbox, year=2024)
    
    # Custom analysis to select interesting tiles
    selected_coords = []
    for lon, lat, embedding, crs, transform in tiles:
        # Example: select tiles with high variance in band 64
        band_64_var = np.var(embedding[:, :, 64])
        if band_64_var > 0.5:  # Threshold
            selected_coords.append((lon, lat))
    
    print(f"Selected {len(selected_coords)} interesting tiles")
    
    # Step 2: Export selected tiles as GeoTIFF  
    all_files = []
    for lon, lat in selected_coords:
        files = gt.export_embedding_geotiffs(
            bbox=(lon-0.05, lat-0.05, lon+0.05, lat+0.05),
            output_dir="./interesting_tiles",
            year=2024,
            bands=[60, 64, 68]  # Bands around interesting band 64
        )
        all_files.extend(files)
    
    # Step 3: Create combined visualization
    create_rgb_mosaic_from_geotiffs(
        geotiff_paths=all_files,
        output_path="interesting_mosaic.tif",
        bands=(0, 1, 2),  # Use the 3 exported bands
        normalize=True
    )
    
    print("Created mosaic of interesting tiles: interesting_mosaic.tif")

Next Steps
----------

- Read the :doc:`architecture` section to understand how GeoTessera works internally
- Check the :doc:`tutorials` for more detailed examples
- Browse the :doc:`cli_reference` for all available command options
- Explore the :doc:`modules` for complete API documentation

Common Issues
-------------

**No tiles found in region**:
   Check the coverage map first. The region might not have available data.

**Slow downloads**:
   Files are cached after first download. Subsequent access will be much faster.

**Memory issues with large regions**:
   Process tiles individually or use smaller bounding boxes.

**Projection issues in GIS software**:
   GeoTIFF files use UTM projections. Most GIS software will handle this automatically.