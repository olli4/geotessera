"""Core GeoTessera functionality for accessing geospatial embeddings."""
import os
from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple, Iterator
import importlib.resources
import pooch
import geopandas as gpd
import numpy as np


class GeoTessera:
    """Main class for accessing GeoTessera geospatial embeddings.
    
    This class provides methods to fetch and load geospatial embeddings
    from the GeoTessera dataset.
    
    Attributes:
        version: Version of the GeoTessera dataset to use (default: "v1")
        cache_dir: Directory to cache downloaded files
    """
    
    def __init__(self, version: str = "v1", cache_dir: Optional[Union[str, Path]] = None):
        """Initialize GeoTessera client.
        
        Args:
            version: Version of the dataset to use
            cache_dir: Directory to cache downloaded files. If None, uses system cache.
        """
        self.version = version
        self._cache_dir = cache_dir
        self._pooch = None
        self._available_embeddings = []
        self._initialize_pooch()
    
    def _initialize_pooch(self):
        """Initialize the Pooch downloader with registry."""
        cache_path = self._cache_dir if self._cache_dir else pooch.os_cache("geotessera")
        
        self._pooch = pooch.create(
            path=cache_path,
            base_url=f"https://dl-1.tessera.wiki/{self.version}/global_0.1_degree_representation/",
            version=self.version,
            registry=None,
        )
        
        # Load the registry file
        with importlib.resources.open_text("geotessera", "registry_2024.txt") as registry_file:
            self._pooch.load_registry(registry_file)
        
        # Parse and cache available embeddings
        self._parse_available_embeddings()
    
    def fetch_embedding(self, lat: float, lon: float, year: int = 2024, 
                       progressbar: bool = True) -> np.ndarray:
        """Fetch and dequantize embedding for a specific location.
        
        This method fetches both the quantized embedding and its scales,
        then performs dequantization by multiplying them together.
        
        Args:
            lat: Latitude coordinate
            lon: Longitude coordinate
            year: Year of the embedding (default: 2024)
            progressbar: Show download progress bar
            
        Returns:
            Dequantized embedding array with shape (height, width, channels)
        """
        # Format coordinates to match file naming convention
        grid_name = f"grid_{lon:.2f}_{lat:.2f}"
        
        # Fetch both the main embedding and scales files
        embedding_path = f"{year}/{grid_name}/{grid_name}.npy"
        scales_path = f"{year}/{grid_name}/{grid_name}_scales.npy"
        
        embedding_file = self._pooch.fetch(embedding_path, progressbar=progressbar)
        scales_file = self._pooch.fetch(scales_path, progressbar=progressbar)
        
        # Load both files
        embedding = np.load(embedding_file)  # shape: (height, width, channels)
        scales = np.load(scales_file)        # shape: (height, width)
        
        # Dequantize by multiplying embedding by scales across all channels
        # Broadcasting scales from (height, width) to (height, width, channels)
        dequantized = embedding.astype(np.float32) * scales[:, :, np.newaxis]
        
        return dequantized
    
    def get_embedding(self, lat: float, lon: float, year: int = 2024) -> np.ndarray:
        """Get the dequantized embedding for a specific location.
        
        Args:
            lat: Latitude coordinate
            lon: Longitude coordinate
            year: Year of the embedding (default: 2024)
            
        Returns:
            Dequantized embedding array with shape (height, width, channels)
        """
        return self.fetch_embedding(lat, lon, year, progressbar=True)
    
    def _parse_available_embeddings(self):
        """Parse registry to extract available embeddings as (year, lat, lon) tuples."""
        embeddings = []
        
        for file_path in self._pooch.registry.keys():
            # Only process .npy files that are not scale files
            if file_path.endswith('.npy') and not file_path.endswith('_scales.npy'):
                # Parse file path: e.g., "2024/grid_0.15_52.05/grid_0.15_52.05.npy"
                parts = file_path.split('/')
                if len(parts) >= 3:
                    year_str = parts[0]
                    grid_name = parts[1]  # e.g., "grid_0.15_52.05"
                    
                    try:
                        year = int(year_str)
                        
                        # Extract coordinates from grid name
                        if grid_name.startswith('grid_'):
                            coords = grid_name[5:].split('_')  # Remove "grid_" prefix
                            if len(coords) == 2:
                                lon = float(coords[0])
                                lat = float(coords[1])
                                embeddings.append((year, lat, lon))
                                
                    except (ValueError, IndexError):
                        continue
        
        # Sort by year, then lat, then lon for consistent ordering
        embeddings.sort(key=lambda x: (x[0], x[1], x[2]))
        self._available_embeddings = embeddings
    
    def list_available_embeddings(self) -> Iterator[Tuple[int, float, float]]:
        """List all available embeddings as (year, lat, lon) tuples.
        
        Returns:
            Iterator of tuples containing (year, latitude, longitude) for each available embedding
        """
        return iter(self._available_embeddings)
    
    def count_available_embeddings(self) -> int:
        """Get the total number of available embeddings.
        
        Returns:
            Total count of available embeddings
        """
        return len(self._available_embeddings)
    
    def get_registry_info(self) -> Dict[str, str]:
        """Get information about all files in the registry.
        
        Returns:
            Dictionary mapping file paths to their checksums
        """
        return dict(self._pooch.registry)
    
    def get_tiles_for_topojson(self, topojson_path: Union[str, Path]) -> List[Tuple[float, float, str]]:
        """Get all Tessera tiles that intersect with a TopoJSON file geometries.
        
        Args:
            topojson_path: Path to the TopoJSON file
            
        Returns:
            List of tuples containing (lat, lon, tile_path) for intersecting tiles
        """
        from shapely.geometry import box
        
        # Read the TopoJSON file
        gdf = gpd.read_file(topojson_path)
        
        # Create a unified geometry (union of all features)
        # This handles the convex hull of all shapes properly
        unified_geom = gdf.unary_union
        
        # Get the bounds to limit our search area
        bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
        min_lon, min_lat, max_lon, max_lat = bounds
        
        # Round to 0.1 degree grid (Tessera grid resolution) to get search bounds
        min_lon_grid = np.floor(min_lon * 10) / 10
        max_lon_grid = np.ceil(max_lon * 10) / 10
        min_lat_grid = np.floor(min_lat * 10) / 10
        max_lat_grid = np.ceil(max_lat * 10) / 10
        
        # Get all available tiles
        available_tiles = self.list_available_embeddings()
        
        # Filter tiles that actually intersect with the geometries
        overlapping_tiles = []
        
        for year, tile_lat, tile_lon in available_tiles:
            # First check if tile is within the bounding box (optimization)
            if (tile_lon >= min_lon_grid and tile_lon <= max_lon_grid and
                tile_lat >= min_lat_grid and tile_lat <= max_lat_grid):
                
                # Create a box representing the tile (0.1 degree grid)
                tile_box = box(tile_lon, tile_lat, tile_lon + 0.1, tile_lat + 0.1)
                
                # Check if the tile box intersects with the actual geometries
                # Conservative approach: if ANY part of the boundary is within the tile, include it
                if unified_geom.intersects(tile_box):
                    # Create the tile path for reference
                    tile_path = f"{year}/grid_{tile_lon:.2f}_{tile_lat:.2f}/grid_{tile_lon:.2f}_{tile_lat:.2f}.npy"
                    overlapping_tiles.append((tile_lat, tile_lon, tile_path))
        
        return overlapping_tiles
    
    def visualize_topojson_with_tiles(self, topojson_path: Union[str, Path], 
                                    output_path: str = "topojson_tiles.png",
                                    bands: List[int] = [0, 1, 2],
                                    normalize: bool = True) -> str:
        """Visualize a TopoJSON file with its overlapping Tessera tiles showing actual embeddings.
        
        Args:
            topojson_path: Path to the TopoJSON file
            output_path: Path for the output visualization
            bands: Three band indices to visualize as RGB
            normalize: Whether to normalize band values
            
        Returns:
            Path to the created visualization
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        
        # Read the TopoJSON file
        gdf = gpd.read_file(topojson_path)
        
        # Get overlapping tiles
        tiles = self.get_tiles_for_topojson(topojson_path)
        
        if not tiles:
            print("No overlapping tiles found")
            return output_path
        
        # Calculate grid dimensions
        tile_bounds = [
            min(lon for _, lon, _ in tiles),
            min(lat for lat, _, _ in tiles), 
            max(lon for _, lon, _ in tiles) + 0.1,
            max(lat for lat, _, _ in tiles) + 0.1
        ]
        
        # Create a dictionary to map coordinates to tile data
        tile_data_dict = {}
        
        # Download and process each tile
        print(f"Processing {len(tiles)} tiles...")
        for i, (lat, lon, tile_path) in enumerate(tiles):
            print(f"Processing tile {i+1}/{len(tiles)}: ({lat:.2f}, {lon:.2f})")
            
            try:
                # Download and dequantize the tile data using named arguments to avoid confusion
                data = self.fetch_embedding(lat=lat, lon=lon, progressbar=False)
                
                # Extract bands for visualization (data is already float32 after dequantization)
                vis_data = data[:, :, bands].copy()
                
                # Normalize if requested
                if normalize:
                    for j in range(vis_data.shape[2]):
                        channel = vis_data[:, :, j]
                        min_val = np.min(channel)
                        max_val = np.max(channel)
                        if max_val > min_val:
                            vis_data[:, :, j] = (channel - min_val) / (max_val - min_val)
                
                # Ensure we have valid RGB data in [0,1] range
                vis_data = np.clip(vis_data, 0, 1)
                
                # Store the processed tile data
                tile_data_dict[(lat, lon)] = vis_data
                
            except Exception as e:
                print(f"ERROR: Failed to download tile ({lat:.2f}, {lon:.2f}): {e}")
                tile_data_dict[(lat, lon)] = None
        
        # Create the composite image
        # Calculate number of tiles in each direction
        lon_min, lat_min, lon_max, lat_max = tile_bounds
        n_lon_tiles = int(round((lon_max - lon_min) / 0.1))
        n_lat_tiles = int(round((lat_max - lat_min) / 0.1))
        
        # Since tiles have variable dimensions, we'll use a different approach
        # Display each tile individually with proper geographic positioning
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Display each tile individually
        for (lat, lon), tile_data in tile_data_dict.items():
            if tile_data is not None:
                # Create extent for this tile (each tile covers 0.1 degrees)
                extent = [lon, lon + 0.1, lat, lat + 0.1]
                
                # Display the tile with proper extent
                ax.imshow(tile_data, extent=extent, origin='lower', alpha=0.8, interpolation='nearest')
                
                # Add tile boundary
                rect = Rectangle((lon, lat), 0.1, 0.1, 
                               linewidth=0.5, edgecolor='white', facecolor='none', alpha=0.9)
                ax.add_patch(rect)
            else:
                # For failed tiles, show a light gray placeholder with red border
                rect = Rectangle((lon, lat), 0.1, 0.1, 
                               linewidth=2, edgecolor='red', facecolor='lightgray', alpha=0.3)
                ax.add_patch(rect)
                
                # Add "N/A" text to indicate missing data
                ax.text(lon + 0.05, lat + 0.05, 'N/A', 
                       ha='center', va='center', fontsize=8, color='red', weight='bold')
        
        # Plot the geometries from TopoJSON on top
        gdf.plot(ax=ax, alpha=0.3, edgecolor='black', facecolor='none', linewidth=2)
        
        # Set extent with padding
        padding = 0.02  # 2% padding
        width = tile_bounds[2] - tile_bounds[0]
        height = tile_bounds[3] - tile_bounds[1]
        
        ax.set_xlim(tile_bounds[0] - width * padding, tile_bounds[2] + width * padding)
        ax.set_ylim(tile_bounds[1] - height * padding, tile_bounds[3] + height * padding)
        
        # Set title and labels
        ax.set_title(f'TopoJSON Regions with Tessera Tile Embeddings\n{len(tiles)} tiles (bands {bands})')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        
        # Equal aspect ratio for proper geographic display
        ax.set_aspect('equal')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def visualize_topojson_as_tiff(self, topojson_path: Union[str, Path], 
                                   output_path: str = "topojson_tiles.tiff",
                                   bands: List[int] = [0, 1, 2],
                                   normalize: bool = True) -> str:
        """Export a high-resolution TIFF of overlapping Tessera tiles with optional TopoJSON boundary.
        
        This creates a clean mosaic without legends, titles, or decorations - just the 
        false-color tile imagery and optionally the TopoJSON boundary overlay.
        
        Args:
            topojson_path: Path to the TopoJSON file
            output_path: Path for the output TIFF file
            bands: Three band indices to visualize as RGB
            normalize: Whether to normalize band values
            
        Returns:
            Path to the created TIFF file
        """
        try:
            from PIL import Image
            import rasterio
            from rasterio.transform import from_bounds
        except ImportError:
            raise ImportError("Please install rasterio and pillow for TIFF export: pip install rasterio pillow")
        
        # Read the TopoJSON file
        gdf = gpd.read_file(topojson_path)
        
        # Get overlapping tiles
        tiles = self.get_tiles_for_topojson(topojson_path)
        
        if not tiles:
            print("No overlapping tiles found")
            return output_path
        
        # Calculate bounding box for all tiles
        lon_min = min(lon for _, lon, _ in tiles)
        lat_min = min(lat for lat, _, _ in tiles)
        lon_max = max(lon for _, lon, _ in tiles) + 0.1
        lat_max = max(lat for lat, _, _ in tiles) + 0.1
        
        # Download and process each tile
        tile_data_dict = {}
        print(f"Processing {len(tiles)} tiles for TIFF export...")
        
        for i, (lat, lon, tile_path) in enumerate(tiles):
            print(f"Processing tile {i+1}/{len(tiles)}: ({lat:.2f}, {lon:.2f})")
            
            try:
                # Download and dequantize the tile data
                data = self.fetch_embedding(lat=lat, lon=lon, progressbar=False)
                
                # Extract bands for visualization
                vis_data = data[:, :, bands].copy()
                
                # Normalize if requested
                if normalize:
                    for j in range(vis_data.shape[2]):
                        channel = vis_data[:, :, j]
                        min_val = np.min(channel)
                        max_val = np.max(channel)
                        if max_val > min_val:
                            vis_data[:, :, j] = (channel - min_val) / (max_val - min_val)
                
                # Ensure we have valid RGB data in [0,1] range
                vis_data = np.clip(vis_data, 0, 1)
                
                # Store the processed tile data
                tile_data_dict[(lat, lon)] = vis_data
                
            except Exception as e:
                print(f"WARNING: Failed to download tile ({lat:.2f}, {lon:.2f}): {e}")
                tile_data_dict[(lat, lon)] = None
        
        # Determine the resolution based on the first valid tile
        tile_height, tile_width = None, None
        for (lat, lon), tile_data in tile_data_dict.items():
            if tile_data is not None:
                tile_height, tile_width = tile_data.shape[:2]
                break
        
        if tile_height is None:
            raise ValueError("No valid tiles were downloaded")
        
        # Calculate the size of the output mosaic
        # Each tile covers 0.1 degrees, calculate pixels per degree
        pixels_per_degree_lat = tile_height / 0.1
        pixels_per_degree_lon = tile_width / 0.1
        
        # Calculate output dimensions
        mosaic_width = int((lon_max - lon_min) * pixels_per_degree_lon)
        mosaic_height = int((lat_max - lat_min) * pixels_per_degree_lat)
        
        # Create the mosaic array
        mosaic = np.zeros((mosaic_height, mosaic_width, 3), dtype=np.float32)
        
        # Place each tile in the mosaic
        for (lat, lon), tile_data in tile_data_dict.items():
            if tile_data is not None:
                # Calculate pixel coordinates for this tile
                x_start = int((lon - lon_min) * pixels_per_degree_lon)
                y_start = int((lat_max - lat - 0.1) * pixels_per_degree_lat)  # Flip Y axis
                
                # Place tile in mosaic
                y_end = y_start + tile_height
                x_end = x_start + tile_width
                
                # Handle edge cases where tiles might extend beyond mosaic bounds
                y_end = min(y_end, mosaic_height)
                x_end = min(x_end, mosaic_width)
                
                tile_y_end = y_end - y_start
                tile_x_end = x_end - x_start
                
                mosaic[y_start:y_end, x_start:x_end] = tile_data[:tile_y_end, :tile_x_end]
        
        # Convert to uint8 for TIFF export
        mosaic_uint8 = (mosaic * 255).astype(np.uint8)
        
        # Create georeferencing transform
        transform = from_bounds(lon_min, lat_min, lon_max, lat_max, mosaic_width, mosaic_height)
        
        # Write the GeoTIFF
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=mosaic_height,
            width=mosaic_width,
            count=3,
            dtype='uint8',
            crs='EPSG:4326',  # WGS84
            transform=transform,
            compress='lzw'
        ) as dst:
            # Write RGB bands
            for i in range(3):
                dst.write(mosaic_uint8[:, :, i], i + 1)
        
        print(f"Exported high-resolution TIFF to {output_path}")
        print(f"Dimensions: {mosaic_width}x{mosaic_height} pixels")
        print(f"Geographic bounds: {lon_min:.4f}, {lat_min:.4f}, {lon_max:.4f}, {lat_max:.4f}")
        
        return output_path