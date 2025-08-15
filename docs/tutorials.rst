Tutorials
=========

This section contains comprehensive tutorials for different use cases with GeoTessera.

Tutorial 1: Basic Data Analysis
-------------------------------

This tutorial covers the fundamentals of downloading and analyzing Tessera embeddings using Python.

Setup
~~~~~

First, install GeoTessera and import required libraries::

    pip install geotessera matplotlib jupyter

    # In Python/Jupyter
    import numpy as np
    import matplotlib.pyplot as plt
    from geotessera import GeoTessera
    import json

Initialize the Client
~~~~~~~~~~~~~~~~~~~~~

Create a GeoTessera client::

    # Initialize with default settings
    gt = GeoTessera()
    
    # Check available years
    years = gt.get_available_years()
    print(f"Available years: {years}")

Explore Data Availability
~~~~~~~~~~~~~~~~~~~~~~~~~

Before downloading, check what data is available::

    # Generate a coverage map
    from geotessera.visualization import visualize_global_coverage
    
    visualize_global_coverage(
        tessera_client=gt,
        output_path="global_coverage.png",
        year=2024,
        figsize=(15, 8),
        tile_color="blue",
        tile_alpha=0.4
    )
    
    print("Coverage map saved to global_coverage.png")

Download and Analyze a Single Tile
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Start with a single tile for Cambridge, UK::

    # Download embedding for Cambridge
    lat, lon = 52.05, 0.15
    year = 2024
    
    embedding = gt.fetch_embedding(lat=lat, lon=lon, year=year)
    
    print(f"Embedding shape: {embedding.shape}")
    print(f"Data type: {embedding.dtype}")
    print(f"Value range: [{embedding.min():.3f}, {embedding.max():.3f}]")
    print(f"Memory usage: {embedding.nbytes / 1024**2:.1f} MB")

Basic Statistical Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~

Compute basic statistics for the embedding::

    # Per-channel statistics
    mean_per_channel = np.mean(embedding, axis=(0, 1))
    std_per_channel = np.std(embedding, axis=(0, 1))
    
    print("First 10 channels:")
    for i in range(10):
        print(f"  Channel {i:2d}: mean={mean_per_channel[i]:6.3f}, "
              f"std={std_per_channel[i]:6.3f}")
    
    # Spatial statistics
    center_pixel = embedding[embedding.shape[0]//2, embedding.shape[1]//2, :]
    corner_pixel = embedding[0, 0, :]
    
    print(f"\nCenter pixel (first 5 channels): {center_pixel[:5]}")
    print(f"Corner pixel (first 5 channels): {corner_pixel[:5]}")

Visualize Individual Channels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create visualizations of different channels::

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Visualize channels 0, 10, 20, 30, 60, 100
    channels_to_plot = [0, 10, 20, 30, 60, 100]
    
    for i, channel in enumerate(channels_to_plot):
        ax = axes[i]
        im = ax.imshow(embedding[:, :, channel], cmap='viridis')
        ax.set_title(f'Channel {channel}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('cambridge_channels.png', dpi=150, bbox_inches='tight')
    plt.show()

Multi-Tile Regional Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Download and analyze multiple tiles for a region::

    # Define bounding box for Cambridge area
    bbox = (0.0, 52.0, 0.3, 52.2)  # (min_lon, min_lat, max_lon, max_lat)
    
    # Fetch all tiles in the region
    embeddings = gt.fetch_embeddings(bbox, year=2024)
    
    print(f"Found {len(embeddings)} tiles in the region")
    
    # Analyze each tile
    tile_stats = []
    for tile_lat, tile_lon, embedding in embeddings:
        stats = {
            'lat': tile_lat,
            'lon': tile_lon,
            'mean_all_channels': np.mean(embedding),
            'std_all_channels': np.std(embedding),
            'channel_50_mean': np.mean(embedding[:, :, 50]),
            'channel_50_std': np.std(embedding[:, :, 50]),
        }
        tile_stats.append(stats)
        
        print(f"Tile ({tile_lat:.2f}, {tile_lon:.2f}): "
              f"overall_mean={stats['mean_all_channels']:.3f}, "
              f"ch50_mean={stats['channel_50_mean']:.3f}")

Save Analysis Results
~~~~~~~~~~~~~~~~~~~~~

Save the analysis results for later use::

    # Save tile statistics
    with open('cambridge_analysis.json', 'w') as f:
        json.dump(tile_stats, f, indent=2)
    
    # Save raw embeddings for further analysis
    for i, (tile_lat, tile_lon, embedding) in enumerate(embeddings):
        filename = f'cambridge_tile_{tile_lat:.2f}_{tile_lon:.2f}.npy'
        np.save(filename, embedding)
        print(f"Saved {filename}")

Tutorial 2: GIS Integration Workflow
------------------------------------

This tutorial shows how to work with GeoTIFF exports for GIS software integration.

Export Embeddings as GeoTIFF
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Export a region as georeferenced GeoTIFF files::

    from geotessera import GeoTessera
    
    gt = GeoTessera()
    
    # Define region (London area)
    bbox = (-0.2, 51.4, 0.1, 51.6)
    year = 2024
    
    # Export all bands
    all_files = gt.export_embedding_geotiffs(
        bbox=bbox,
        output_dir="./london_full",
        year=year,
        compress="lzw"
    )
    
    print(f"Exported {len(all_files)} GeoTIFF files")
    
    # Export RGB subset for visualization
    rgb_files = gt.export_embedding_geotiffs(
        bbox=bbox,
        output_dir="./london_rgb",
        year=year,
        bands=[30, 60, 90],  # Custom RGB bands
        compress="lzw"
    )
    
    print(f"Exported {len(rgb_files)} RGB GeoTIFF files")

Inspect GeoTIFF Metadata
~~~~~~~~~~~~~~~~~~~~~~~~

Check the georeferencing information::

    import rasterio
    
    # Inspect the first file
    sample_file = all_files[0]
    
    with rasterio.open(sample_file) as src:
        print(f"File: {sample_file}")
        print(f"Shape: {src.shape}")
        print(f"Bands: {src.count}")
        print(f"Data type: {src.dtypes[0]}")
        print(f"CRS: {src.crs}")
        print(f"Transform: {src.transform}")
        print(f"Bounds: {src.bounds}")
        
        # Read a sample of the data
        sample_data = src.read(1)  # Read first band
        print(f"Data range: [{sample_data.min():.3f}, {sample_data.max():.3f}]")

Create RGB Composite
~~~~~~~~~~~~~~~~~~~~

Create an RGB visualization from the exported bands::

    from geotessera.visualization import create_rgb_mosaic_from_geotiffs
    
    # Create RGB mosaic from the 3-band files
    mosaic_file = create_rgb_mosaic_from_geotiffs(
        geotiff_paths=rgb_files,
        output_path="london_rgb_mosaic.tif",
        bands=(0, 1, 2),  # Use all 3 exported bands as RGB
        normalize=True
    )
    
    print(f"Created RGB mosaic: {mosaic_file}")

Generate Web Tiles
~~~~~~~~~~~~~~~~~~

Create interactive web tiles from the GeoTIFF::

    from geotessera.visualization import geotiff_to_web_tiles, create_simple_web_viewer
    
    # Generate web tiles
    tiles_dir = "./london_web_tiles"
    geotiff_to_web_tiles(
        geotiff_path=mosaic_file,
        output_dir=tiles_dir,
        zoom_levels=(8, 15)
    )
    
    # Create a simple web viewer
    create_simple_web_viewer(
        tiles_dir=tiles_dir,
        output_html="london_map.html",
        center_lat=51.5,
        center_lon=-0.05,
        zoom=10,
        title="London Tessera Embeddings"
    )
    
    print("Web tiles created. Open london_map.html in a browser.")

QGIS Integration
~~~~~~~~~~~~~~~

Tips for using the GeoTIFF files in QGIS:

1. **Loading files**: Drag and drop GeoTIFF files directly into QGIS
2. **Projection**: Files use UTM projection - QGIS will handle reprojection automatically
3. **Styling**: Use single-band pseudocolor for individual channels
4. **RGB composites**: Use the RGB mosaic files for natural color visualization
5. **Analysis**: Use QGIS raster calculator for band math operations

Example QGIS workflow::

    # In QGIS Python console
    from qgis.core import QgsRasterLayer
    
    # Load a GeoTIFF
    layer = QgsRasterLayer('/path/to/london_full/grid_51.45_-0.05.tif', 'Tessera Embedding')
    QgsProject.instance().addMapLayer(layer)
    
    # Set single-band pseudocolor for channel 50
    from qgis.core import QgsColorRampShader, QgsSingleBandPseudoColorRenderer
    
    renderer = QgsSingleBandPseudoColorRenderer(layer.dataProvider(), 50)  # Channel 50
    shader = QgsColorRampShader()
    # Configure color ramp...
    layer.setRenderer(renderer)

Tutorial 3: Large-Scale Analysis
--------------------------------

This tutorial covers working with large regions and multiple years of data.

Memory-Efficient Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

When working with large regions, process tiles individually::

    from geotessera import GeoTessera
    import numpy as np
    
    gt = GeoTessera()
    
    # Large region (entire southern England)
    bbox = (-3.0, 50.0, 2.0, 53.0)
    year = 2024
    
    def process_large_region_efficiently(bbox, year, analysis_func):
        """Process a large region without loading all tiles into memory."""
        
        # Get list of available tiles (metadata only)
        embeddings = gt.fetch_embeddings(bbox, year)
        total_tiles = len(embeddings)
        
        print(f"Processing {total_tiles} tiles...")
        
        results = []
        for i, (tile_lat, tile_lon, embedding) in enumerate(embeddings):
            # Process one tile at a time
            result = analysis_func(embedding, tile_lat, tile_lon)
            results.append(result)
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{total_tiles} tiles")
            
            # Free memory
            del embedding
        
        return results
    
    def vegetation_analysis(embedding, lat, lon):
        """Example analysis function for vegetation detection."""
        # Hypothetical vegetation channels (example)
        veg_channels = [20, 25, 30, 35, 40]
        
        # Compute vegetation index
        veg_data = embedding[:, :, veg_channels]
        veg_index = np.mean(veg_data, axis=2)
        
        return {
            'lat': lat,
            'lon': lon,
            'mean_vegetation': float(np.mean(veg_index)),
            'max_vegetation': float(np.max(veg_index)),
            'vegetation_pixels': int(np.sum(veg_index > 0.5))
        }
    
    # Run the analysis
    results = process_large_region_efficiently(bbox, year, vegetation_analysis)
    
    # Save results
    with open('vegetation_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)

Batch Export for Multiple Regions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Export multiple regions efficiently::

    def batch_export_regions(regions_config, base_output_dir):
        """Export multiple regions as GeoTIFF files."""
        import os
        from pathlib import Path
        
        gt = GeoTessera()
        
        for region_name, config in regions_config.items():
            print(f"Processing region: {region_name}")
            
            output_dir = Path(base_output_dir) / region_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                files = gt.export_embedding_geotiffs(
                    bbox=config['bbox'],
                    output_dir=str(output_dir),
                    year=config['year'],
                    bands=config.get('bands', None),
                    compress="lzw"
                )
                
                print(f"  Exported {len(files)} files to {output_dir}")
                
                # Create metadata file
                metadata = {
                    'region': region_name,
                    'bbox': config['bbox'],
                    'year': config['year'],
                    'files': files,
                    'band_count': len(config.get('bands', list(range(128))))
                }
                
                with open(output_dir / 'metadata.json', 'w') as f:
                    json.dump(metadata, f, indent=2)
                    
            except Exception as e:
                print(f"  Error processing {region_name}: {e}")
    
    # Define regions to process
    regions = {
        'london': {
            'bbox': (-0.3, 51.3, 0.2, 51.7),
            'year': 2024,
            'bands': [10, 20, 30, 40, 50]  # Subset of bands
        },
        'cambridge': {
            'bbox': (-0.2, 52.0, 0.3, 52.3),
            'year': 2024,
            'bands': None  # All bands
        },
        'oxford': {
            'bbox': (-1.4, 51.6, -1.1, 51.9),
            'year': 2024,
            'bands': [0, 1, 2]  # RGB only
        }
    }
    
    # Run batch export
    batch_export_regions(regions, "./batch_exports")

Multi-Year Comparison
~~~~~~~~~~~~~~~~~~~~

Compare embeddings across different years::

    def compare_years(lat, lon, years):
        """Compare a single location across multiple years."""
        gt = GeoTessera()
        
        yearly_data = {}
        for year in years:
            try:
                embedding = gt.fetch_embedding(lat=lat, lon=lon, year=year)
                
                # Compute summary statistics
                yearly_data[year] = {
                    'mean_per_channel': np.mean(embedding, axis=(0, 1)).tolist(),
                    'std_per_channel': np.std(embedding, axis=(0, 1)).tolist(),
                    'overall_mean': float(np.mean(embedding)),
                    'overall_std': float(np.std(embedding))
                }
                
                print(f"Year {year}: mean={yearly_data[year]['overall_mean']:.3f}")
                
            except Exception as e:
                print(f"Year {year}: Data not available ({e})")
                yearly_data[year] = None
        
        return yearly_data
    
    # Compare Cambridge across years
    cambridge_comparison = compare_years(
        lat=52.05, lon=0.15, 
        years=[2020, 2021, 2022, 2023, 2024]
    )
    
    # Save comparison
    with open('cambridge_temporal_comparison.json', 'w') as f:
        json.dump(cambridge_comparison, f, indent=2)
    
    # Plot temporal trends
    valid_years = [year for year, data in cambridge_comparison.items() if data is not None]
    overall_means = [cambridge_comparison[year]['overall_mean'] for year in valid_years]
    
    plt.figure(figsize=(10, 6))
    plt.plot(valid_years, overall_means, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Year')
    plt.ylabel('Mean Embedding Value')
    plt.title('Temporal Trend - Cambridge (52.05°N, 0.15°E)')
    plt.grid(True, alpha=0.3)
    plt.savefig('cambridge_temporal_trend.png', dpi=150, bbox_inches='tight')
    plt.show()

Tutorial 4: Custom Analysis Workflows
-------------------------------------

Advanced analysis techniques and custom workflows.

Principal Component Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reduce dimensionality of the 128-channel embeddings::

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    
    def perform_pca_analysis(embeddings_list, n_components=10):
        """Perform PCA on a collection of embedding tiles."""
        
        # Reshape all embeddings to 2D (pixels x channels)
        all_pixels = []
        tile_info = []
        
        for tile_lat, tile_lon, embedding in embeddings_list:
            # Reshape from (H, W, 128) to (H*W, 128)
            pixels = embedding.reshape(-1, embedding.shape[-1])
            all_pixels.append(pixels)
            
            # Track which pixels belong to which tile
            n_pixels = pixels.shape[0]
            tile_info.extend([(tile_lat, tile_lon)] * n_pixels)
        
        # Combine all pixels
        X = np.vstack(all_pixels)
        print(f"Total pixels for PCA: {X.shape[0]:,}")
        print(f"Feature dimensions: {X.shape[1]}")
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        # Print explained variance
        print(f"\nExplained variance by component:")
        for i, var in enumerate(pca.explained_variance_ratio_):
            print(f"  PC{i+1}: {var:.3f} ({var*100:.1f}%)")
        print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.3f}")
        
        return X_pca, pca, scaler, tile_info
    
    # Example usage
    gt = GeoTessera()
    bbox = (-0.1, 51.9, 0.1, 52.1)  # Small region around Cambridge
    embeddings = gt.fetch_embeddings(bbox, year=2024)
    
    X_pca, pca, scaler, tile_info = perform_pca_analysis(embeddings, n_components=5)
    
    # Visualize first two principal components
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5, s=1)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA: First Two Components')
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
             pca.explained_variance_ratio_, 'bo-')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('PCA Explained Variance')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pca_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

Clustering Analysis
~~~~~~~~~~~~~~~~~~

Identify similar regions using clustering::

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    
    def cluster_embedding_tiles(embeddings_list, n_clusters=5):
        """Cluster embedding tiles based on their mean features."""
        
        # Extract mean features for each tile
        tile_features = []
        tile_coords = []
        
        for tile_lat, tile_lon, embedding in embeddings_list:
            # Use mean values across spatial dimensions
            mean_features = np.mean(embedding, axis=(0, 1))
            tile_features.append(mean_features)
            tile_coords.append((tile_lat, tile_lon))
        
        X = np.array(tile_features)
        print(f"Clustering {X.shape[0]} tiles with {X.shape[1]} features")
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        print(f"Silhouette score: {silhouette_avg:.3f}")
        
        # Organize results
        clusters = {}
        for i, (coords, label) in enumerate(zip(tile_coords, cluster_labels)):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append({
                'coords': coords,
                'features': tile_features[i]
            })
        
        return clusters, kmeans, scaler
    
    # Perform clustering
    clusters, kmeans, scaler = cluster_embedding_tiles(embeddings, n_clusters=3)
    
    # Visualize clusters
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Geographic distribution of clusters
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for cluster_id, tiles in clusters.items():
        lats = [tile['coords'][0] for tile in tiles]
        lons = [tile['coords'][1] for tile in tiles]
        ax1.scatter(lons, lats, c=colors[cluster_id], 
                   label=f'Cluster {cluster_id} ({len(tiles)} tiles)',
                   alpha=0.7, s=50)
    
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title('Geographic Distribution of Clusters')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cluster characteristics (first 10 channels)
    for cluster_id, tiles in clusters.items():
        mean_features = np.mean([tile['features'] for tile in tiles], axis=0)
        ax2.plot(range(10), mean_features[:10], 'o-', 
                label=f'Cluster {cluster_id}', color=colors[cluster_id])
    
    ax2.set_xlabel('Channel Index')
    ax2.set_ylabel('Mean Feature Value')
    ax2.set_title('Cluster Characteristics (Channels 0-9)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('clustering_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

Time Series Analysis
~~~~~~~~~~~~~~~~~~~

Analyze temporal patterns in multi-year data::

    def analyze_temporal_patterns(lat, lon, years, channels_of_interest):
        """Analyze temporal patterns for specific channels at a location."""
        
        gt = GeoTessera()
        temporal_data = {}
        
        for year in years:
            try:
                embedding = gt.fetch_embedding(lat=lat, lon=lon, year=year)
                
                # Extract data for channels of interest
                year_data = {}
                for channel in channels_of_interest:
                    channel_data = embedding[:, :, channel]
                    year_data[f'channel_{channel}'] = {
                        'mean': float(np.mean(channel_data)),
                        'std': float(np.std(channel_data)),
                        'min': float(np.min(channel_data)),
                        'max': float(np.max(channel_data))
                    }
                
                temporal_data[year] = year_data
                
            except Exception as e:
                print(f"Year {year}: {e}")
                continue
        
        return temporal_data
    
    # Analyze temporal patterns for interesting channels
    channels_of_interest = [10, 30, 50, 70, 90]  # Example channels
    years_to_analyze = [2020, 2021, 2022, 2023, 2024]
    
    temporal_results = analyze_temporal_patterns(
        lat=52.05, lon=0.15, 
        years=years_to_analyze,
        channels_of_interest=channels_of_interest
    )
    
    # Plot temporal trends
    fig, axes = plt.subplots(len(channels_of_interest), 1, 
                           figsize=(12, 3*len(channels_of_interest)))
    
    if len(channels_of_interest) == 1:
        axes = [axes]
    
    for i, channel in enumerate(channels_of_interest):
        ax = axes[i]
        
        valid_years = []
        means = []
        stds = []
        
        for year in sorted(temporal_results.keys()):
            if f'channel_{channel}' in temporal_results[year]:
                valid_years.append(year)
                means.append(temporal_results[year][f'channel_{channel}']['mean'])
                stds.append(temporal_results[year][f'channel_{channel}']['std'])
        
        if valid_years:
            ax.errorbar(valid_years, means, yerr=stds, 
                       marker='o', capsize=5, capthick=2)
            ax.set_title(f'Channel {channel} - Temporal Trend')
            ax.set_ylabel('Mean Value')
            ax.grid(True, alpha=0.3)
            
            if i == len(channels_of_interest) - 1:
                ax.set_xlabel('Year')
    
    plt.tight_layout()
    plt.savefig('temporal_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

Save Results and Create Report
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate a comprehensive analysis report::

    def create_analysis_report(results_dict, output_file):
        """Create a comprehensive analysis report."""
        
        report = {
            'analysis_date': str(datetime.now()),
            'geotessera_version': gt.version,
            'summary': {
                'total_tiles_analyzed': len(results_dict.get('tiles', [])),
                'regions_covered': list(results_dict.keys()),
                'years_analyzed': sorted(set(
                    year for region_data in results_dict.values() 
                    if isinstance(region_data, dict)
                    for year in region_data.get('years', [])
                ))
            },
            'detailed_results': results_dict
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Analysis report saved to: {output_file}")
        
        # Create summary statistics
        print("\n=== ANALYSIS SUMMARY ===")
        print(f"Total tiles analyzed: {report['summary']['total_tiles_analyzed']}")
        print(f"Regions covered: {', '.join(report['summary']['regions_covered'])}")
        print(f"Years analyzed: {', '.join(map(str, report['summary']['years_analyzed']))}")
    
    # Compile all results
    all_results = {
        'pca_analysis': {
            'explained_variance': pca.explained_variance_ratio_.tolist(),
            'n_components': len(pca.explained_variance_ratio_)
        },
        'clustering': {
            'n_clusters': len(clusters),
            'cluster_sizes': {str(k): len(v) for k, v in clusters.items()},
            'silhouette_score': float(silhouette_score(X_scaled, cluster_labels))
        },
        'temporal_analysis': temporal_results
    }
    
    create_analysis_report(all_results, 'comprehensive_analysis_report.json')

This comprehensive tutorial set covers the major use cases for GeoTessera, from basic data exploration to advanced machine learning workflows. Each tutorial builds upon the previous ones, providing a complete learning path for users.