"""Command-line interface for GeoTessera."""
import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from .core import GeoTessera


def download_command(args):
    """Handle the download command."""
    tessera = GeoTessera(version=args.version)
    
    print(f"Downloading embedding for coordinates ({args.lat}, {args.lon})...")
    path = tessera.get_embedding_path(args.lat, args.lon, args.year)
    print(f"Downloaded to: {path}")


def list_command(args):
    """Handle the list command."""
    tessera = GeoTessera(version=args.version)
    
    embeddings = tessera.list_available_embeddings()
    print(f"Available embeddings ({len(embeddings)} total):")
    
    if args.limit:
        embeddings = embeddings[:args.limit]
    
    for embedding in embeddings:
        print(f"  - {embedding}")
    
    if args.limit and args.limit < len(tessera.list_available_embeddings()):
        print(f"  ... and {len(tessera.list_available_embeddings()) - args.limit} more")


def info_command(args):
    """Handle the info command."""
    tessera = GeoTessera(version=args.version)
    
    print(f"GeoTessera Dataset Information")
    print(f"Version: {tessera.version}")
    print(f"Base URL: {tessera._pooch.base_url}")
    print(f"Cache directory: {tessera._pooch.path}")
    print(f"Total embeddings: {len(tessera.list_available_embeddings())}")


def visualize_command(args):
    """Handle the visualize command."""
    tessera = GeoTessera(version=args.version)
    
    # If TopoJSON file is provided, visualize tiles for that region
    if args.topojson:
        print(f"Analyzing TopoJSON file: {args.topojson}")
        normalize = not args.no_normalize
        output_path = tessera.visualize_topojson_with_tiles(
            args.topojson, args.output, bands=args.bands, normalize=normalize
        )
        print(f"Created TopoJSON tile visualization: {output_path}")
        
        # Also print the tiles that were found
        tiles = tessera.get_tiles_for_topojson(args.topojson)
        print(f"\nFound {len(tiles)} overlapping tiles:")
        for lat, lon, tile_path in tiles:
            print(f"  - Tile at ({lat:.2f}, {lon:.2f}): {tile_path}")
        return
    
    # Check that lat/lon are provided for regular visualization
    if args.lat is None or args.lon is None:
        print("Error: --lat and --lon are required unless --topojson is used")
        return
    
    # Regular embedding visualization
    print(f"Fetching embedding for ({args.lat}, {args.lon})...")
    embedding_path = tessera.get_embedding_path(args.lat, args.lon, args.year)
    
    print(f"Loading embedding from {embedding_path}...")
    data = np.load(embedding_path, mmap_mode='r')
    
    print(f"Embedding shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    
    # Extract bands for visualization
    vis_data = data[:, :, list(args.bands)].copy()
    vis_data = vis_data.astype(np.float32)
    
    # Normalize if requested
    if not args.no_normalize:
        print("Normalizing bands...")
        for i in range(vis_data.shape[2]):
            channel = vis_data[:, :, i]
            min_val = np.min(channel)
            max_val = np.max(channel)
            if max_val > min_val:
                vis_data[:, :, i] = (channel - min_val) / (max_val - min_val)
    
    # Create visualization
    plt.figure(figsize=(10, 10))
    plt.imshow(vis_data)
    plt.title(f'GeoTessera Embedding\nLocation: ({args.lat}, {args.lon})\nBands: {args.bands}')
    plt.axis('off')
    
    # Save
    plt.savefig(args.output, bbox_inches='tight', dpi=300)
    print(f"Saved visualization to {args.output}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="GeoTessera - Access and visualize geospatial embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download an embedding for specific coordinates
  geotessera download --lat 52.05 --lon 0.15
  
  # List available embeddings
  geotessera list --limit 10
  
  # Get information about the dataset
  geotessera info
  
  # Visualize an embedding
  geotessera visualize --lat 52.05 --lon 0.15 --output output.png
        """
    )
    
    parser.add_argument("--version", default="v1", help="Dataset version (default: v1)")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Download command
    download_parser = subparsers.add_parser("download", help="Download an embedding")
    download_parser.add_argument("--lat", type=float, required=True, help="Latitude coordinate")
    download_parser.add_argument("--lon", type=float, required=True, help="Longitude coordinate")
    download_parser.add_argument("--year", type=int, default=2024, help="Year of embedding (default: 2024)")
    download_parser.set_defaults(func=download_command)
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available embeddings")
    list_parser.add_argument("--limit", type=int, help="Limit number of results shown")
    list_parser.set_defaults(func=list_command)
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show dataset information")
    info_parser.set_defaults(func=info_command)
    
    # Visualize command
    viz_parser = subparsers.add_parser("visualize", help="Visualize GeoTessera embeddings")
    viz_parser.add_argument("--lat", type=float, help="Latitude coordinate (required unless --topojson is used)")
    viz_parser.add_argument("--lon", type=float, help="Longitude coordinate (required unless --topojson is used)")
    viz_parser.add_argument("--year", type=int, default=2024, help="Year of embedding (default: 2024)")
    viz_parser.add_argument("--output", type=str, default="geotessera_vis.png", help="Output file path")
    viz_parser.add_argument("--bands", type=int, nargs=3, default=[0, 1, 2], help="Three band indices to visualize as RGB")
    viz_parser.add_argument("--no-normalize", action="store_true", help="Skip normalization of band values")
    viz_parser.add_argument("--topojson", type=str, help="TopoJSON file to overlay tiles for")
    viz_parser.set_defaults(func=visualize_command)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute the command
    args.func(args)


if __name__ == "__main__":
    main()