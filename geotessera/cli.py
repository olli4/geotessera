"""Command-line interface for GeoTessera."""
import argparse
import sys
from pathlib import Path
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import datetime
from .core import GeoTessera


def list_embeddings_command(args):
    """Handle the list-embeddings command."""
    tessera = GeoTessera(version=args.version)
    
    embeddings = list(tessera.list_available_embeddings())
    print(f"Available embeddings ({len(embeddings)} total):")
    
    if args.limit:
        embeddings = embeddings[:args.limit]
    
    for year, lat, lon in embeddings:
        print(f"  - Year {year}: ({lat:.2f}, {lon:.2f})")
    
    total_count = tessera.count_available_embeddings()
    if args.limit and args.limit < total_count:
        print(f"  ... and {total_count - args.limit} more")


def info_command(args):
    """Handle the info command."""
    tessera = GeoTessera(version=args.version)
    
    print(f"GeoTessera Dataset Information")
    print(f"Version: {tessera.version}")
    print(f"Base URL: {tessera._pooch.base_url}")
    print(f"Cache directory: {tessera._pooch.path}")
    print(f"Total embeddings: {tessera.count_available_embeddings()}")
    print(f"Internal land masks: {tessera._count_available_landmasks()}")
    print(f"Land mask Base URL: {tessera._landmask_pooch.base_url if tessera._landmask_pooch else 'Not loaded'}")


def map_command(args):
    """Handle the map command to generate a world map with all available embedding grid points."""
    tessera = GeoTessera(version=args.version)
    
    print("Generating coverage map from embedding registry data...")
    
    # Get all available embeddings from the library
    embeddings = list(tessera.list_available_embeddings())
    
    if not embeddings:
        print("No embeddings available. Check registry loading.")
        return
    
    # Extract unique coordinates (lat, lon) from embeddings
    coordinates = set()
    for year, lat, lon in embeddings:
        coordinates.add((lat, lon))
    
    print(f"Found {len(coordinates)} unique embedding grid points")
    
    # Convert to DataFrame
    coords_list = list(coordinates)
    df = pd.DataFrame(coords_list, columns=['Latitude', 'Longitude'])
    
    # Convert pandas DataFrame to GeoDataFrame
    print("Creating GeoDataFrame...")
    gdf = geopandas.GeoDataFrame(
        df, geometry=geopandas.points_from_xy(df.Longitude, df.Latitude)
    )
    gdf.set_crs("EPSG:4326", inplace=True)  # Set the coordinate system to WGS84
    
    # Plot the map
    # Check if world map shapefile exists
    world_map_path = Path('world_map/ne_110m_admin_0_countries.shp')
    if not world_map_path.exists():
        print("Using built-in world map.")
        world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    else:
        world = geopandas.read_file(world_map_path)
    
    # Create the plotting area
    print("Creating map...")
    fig, ax = plt.subplots(1, 1, figsize=(20, 12))
    
    # Plot the world map base layer
    world.plot(ax=ax, color='lightgray', edgecolor='black', linewidth=0.5)
    
    # Plot grid points on the map with smaller markers
    gdf.plot(ax=ax, marker='o', color='red', markersize=2, alpha=0.8, label='Available Embeddings')
    
    # Add a title with the current time and grid count
    current_time_utc = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    ax.set_title(f'GeoTessera Embedding Coverage Map\nTotal Embedding Grid Points: {len(df)}\nLast Updated: {current_time_utc}',
                 fontdict={'fontsize': '18', 'fontweight': 'bold'})
    
    # Add legend
    ax.legend(loc='lower left', frameon=True, fancybox=True, shadow=True, fontsize=12)
    
    # Set axis labels
    ax.set_xlabel('Longitude', fontsize=14)
    ax.set_ylabel('Latitude', fontsize=14)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Save the map as an image file
    plt.tight_layout()
    output_path = args.output
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Map saved to {output_path} with {len(df)} embedding grid points!")
    
    # Close the plot to free memory
    plt.close()


def merge_command(args):
    """Handle the merge command to merge land mask tiles for a region."""
    tessera = GeoTessera(version=args.version)
    
    # Parse bounds
    bounds = (args.min_lon, args.min_lat, args.max_lon, args.max_lat)
    
    print(f"Merging land mask tiles for region: {bounds}")
    print(f"Target CRS: {args.target_crs}")
    print("Note: This creates a binary land/water mask for coordinate alignment.")
    
    # Merge land mask tiles
    try:
        output_path = tessera.merge_landmasks_for_region(
            bounds=bounds,
            output_path=args.output,
            target_crs=args.target_crs
        )
        print(f"Successfully merged land mask to: {output_path}")
    except Exception as e:
        print(f"Error merging land mask tiles: {e}")
        sys.exit(1)


def visualize_command(args):
    """Handle the visualize command using TopoJSON files."""
    tessera = GeoTessera(version=args.version)
    
    if not args.topojson:
        print("Error: --topojson is required for visualization")
        sys.exit(1)
    
    print(f"Analyzing TopoJSON file: {args.topojson}")
    
    # Read the TopoJSON to get bounds
    try:
        import geopandas as gpd
        gdf = gpd.read_file(args.topojson)
        bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
        min_lon, min_lat, max_lon, max_lat = bounds
        
        print(f"TopoJSON bounds: ({min_lon:.4f}, {min_lat:.4f}, {max_lon:.4f}, {max_lat:.4f})")
        
        # Merge embedding tiles for this region
        normalize = not args.no_normalize
        output_path = tessera.merge_embeddings_for_region(
            bounds=(min_lon, min_lat, max_lon, max_lat),
            output_path=args.output,
            target_crs=args.target_crs,
            bands=args.bands,
            normalize=normalize,
            year=args.year
        )
        
        print(f"Created merged embedding visualization for TopoJSON region: {output_path}")
        
    except Exception as e:
        print(f"Error processing TopoJSON file: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="GeoTessera - Access geospatial embeddings and create land masks for alignment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available embeddings
  geotessera list-embeddings --limit 10
  
  # Get information about the dataset
  geotessera info
  
  # Generate a world map showing all available embedding grid points
  geotessera map --output coverage_map.png
  
  # Create a false-color visualization from embeddings for a region
  geotessera visualize --topojson region.geojson --output region_viz.tiff --bands 0 1 2
  
  # Create a land mask for coordinate alignment (internal use)
  geotessera merge --min-lon 0.0 --min-lat 52.0 --max-lon 1.0 --max-lat 53.0 --output landmask.tiff

Valid Target CRS Values:
  EPSG:4326     - WGS84 Geographic (lat/lon) - good for global/large areas
  EPSG:326XX    - UTM Northern Hemisphere (XX = zone 01-60, e.g., EPSG:32630)
  EPSG:327XX    - UTM Southern Hemisphere (XX = zone 01-60, e.g., EPSG:32730)  
  EPSG:3995     - Arctic Polar Stereographic (for areas north of 70°N)
  EPSG:3031     - Antarctic Polar Stereographic (for areas south of 70°S)

Note: The 'visualize' command creates false-color visualizations from numpy embeddings.
The 'merge' command creates binary land/water masks for internal coordinate alignment.
        """
    )
    
    parser.add_argument("--version", default="v1", help="Dataset version (default: v1)")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # List embeddings command
    list_parser = subparsers.add_parser("list-embeddings", help="List available embeddings")
    list_parser.add_argument("--limit", type=int, help="Limit number of results shown")
    list_parser.set_defaults(func=list_embeddings_command)
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show dataset information")
    info_parser.set_defaults(func=info_command)
    
    # Map command
    map_parser = subparsers.add_parser("map", help="Generate a world map showing all available embedding grid points")
    map_parser.add_argument("--output", type=str, default="embedding_coverage_map.png", help="Output map file path (default: embedding_coverage_map.png)")
    map_parser.set_defaults(func=map_command)
    
    # Merge command (internal land mask creation)
    merge_parser = subparsers.add_parser("merge", help="Create binary land mask for a region (internal coordinate alignment)")
    merge_parser.add_argument("--min-lon", type=float, required=True, help="Minimum longitude")
    merge_parser.add_argument("--min-lat", type=float, required=True, help="Minimum latitude")
    merge_parser.add_argument("--max-lon", type=float, required=True, help="Maximum longitude")
    merge_parser.add_argument("--max-lat", type=float, required=True, help="Maximum latitude")
    merge_parser.add_argument("--output", type=str, default="landmask.tiff", help="Output land mask file path")
    merge_parser.add_argument("--target-crs", type=str, default="EPSG:4326", 
                             help="Target CRS (default: EPSG:4326). See help for valid values.")
    merge_parser.set_defaults(func=merge_command)
    
    # Visualize command (embedding visualization)
    viz_parser = subparsers.add_parser("visualize", help="Create false-color visualization from embeddings for a TopoJSON/GeoJSON region")
    viz_parser.add_argument("--topojson", type=str, required=True, help="TopoJSON/GeoJSON file to visualize embeddings for")
    viz_parser.add_argument("--output", type=str, default="region_visualization.tiff", help="Output visualization file path")
    viz_parser.add_argument("--target-crs", type=str, default="EPSG:4326", 
                           help="Target CRS (default: EPSG:4326). See help for valid values.")
    viz_parser.add_argument("--bands", type=int, nargs=3, default=[0, 1, 2], 
                           help="Three band indices to use for RGB visualization (default: 0 1 2)")
    viz_parser.add_argument("--no-normalize", action="store_true", 
                           help="Skip normalization of band values")
    viz_parser.add_argument("--year", type=int, default=2024, 
                           help="Year of embeddings to visualize (default: 2024)")
    viz_parser.set_defaults(func=visualize_command)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute the command
    args.func(args)


if __name__ == "__main__":
    main()