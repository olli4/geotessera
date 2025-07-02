"""Core GeoTessera functionality for accessing geospatial embeddings."""
import os
from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple
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
    
    def fetch_embedding(self, lat: float, lon: float, year: int = 2024, 
                       progressbar: bool = True) -> str:
        """Fetch embedding file for a specific location.
        
        Args:
            lat: Latitude coordinate
            lon: Longitude coordinate
            year: Year of the embedding (default: 2024)
            progressbar: Show download progress bar
            
        Returns:
            Path to the downloaded file
        """
        # Format coordinates to match file naming convention
        grid_name = f"grid_{lon:.2f}_{lat:.2f}"
        file_path = f"{year}/{grid_name}/{grid_name}.npy"
        
        return self._pooch.fetch(file_path, progressbar=progressbar)
    
    def get_embedding_path(self, lat: float, lon: float, year: int = 2024) -> str:
        """Get the local file path for an embedding after downloading.
        
        Args:
            lat: Latitude coordinate
            lon: Longitude coordinate
            year: Year of the embedding (default: 2024)
            
        Returns:
            Path to the embedding file
        """
        return self.fetch_embedding(lat, lon, year, progressbar=True)
    
    def list_available_embeddings(self) -> List[str]:
        """List all available embeddings in the registry.
        
        Returns:
            List of available embedding files
        """
        return list(self._pooch.registry.keys())
    
    def get_registry_info(self) -> Dict[str, str]:
        """Get information about all files in the registry.
        
        Returns:
            Dictionary mapping file paths to their checksums
        """
        return dict(self._pooch.registry)
    
    def get_tiles_for_topojson(self, topojson_path: Union[str, Path]) -> List[Tuple[float, float, str]]:
        """Get all Tessera tiles that overlap with a TopoJSON file.
        
        Args:
            topojson_path: Path to the TopoJSON file
            
        Returns:
            List of tuples containing (lat, lon, tile_path) for overlapping tiles
        """
        # Read the TopoJSON file
        gdf = gpd.read_file(topojson_path)
        
        # Get the bounds of all geometries
        bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
        min_lon, min_lat, max_lon, max_lat = bounds
        
        # Round to 0.1 degree grid (Tessera grid resolution)
        min_lon_grid = np.floor(min_lon * 10) / 10
        max_lon_grid = np.ceil(max_lon * 10) / 10
        min_lat_grid = np.floor(min_lat * 10) / 10
        max_lat_grid = np.ceil(max_lat * 10) / 10
        
        # Get all available tiles
        available_tiles = self.list_available_embeddings()
        
        # Filter tiles that overlap with the TopoJSON bounds
        overlapping_tiles = []
        for tile in available_tiles:
            if tile.endswith('.npy') and not tile.endswith('_scales.npy'):
                # Parse tile filename to get coordinates
                # Format: 2024/grid_lon_lat/grid_lon_lat.npy
                parts = tile.split('/')
                if len(parts) >= 2:
                    grid_name = parts[1]  # e.g., "grid_-0.05_10.05"
                    if grid_name.startswith('grid_'):
                        coords = grid_name[5:].split('_')  # Remove "grid_" prefix
                        if len(coords) == 2:
                            try:
                                tile_lon = float(coords[0])
                                tile_lat = float(coords[1])
                                
                                # Check if tile overlaps with bounds
                                if (tile_lon >= min_lon_grid and tile_lon <= max_lon_grid and
                                    tile_lat >= min_lat_grid and tile_lat <= max_lat_grid):
                                    overlapping_tiles.append((tile_lat, tile_lon, tile))
                            except ValueError:
                                continue
        
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
                # Download the tile data using named arguments to avoid confusion
                embedding_path = self.fetch_embedding(lat=lat, lon=lon, progressbar=False)
                data = np.load(embedding_path, mmap_mode='r')
                
                # Extract bands for visualization (same logic as regular visualize)
                vis_data = data[:, :, bands].copy()
                vis_data = vis_data.astype(np.float32)
                
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
                print(f"Warning: Could not process tile ({lat:.2f}, {lon:.2f}): {e}")
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