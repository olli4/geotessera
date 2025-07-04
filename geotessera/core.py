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
        self._landmask_pooch = None
        self._available_embeddings = []
        self._available_landmasks = []
        self._initialize_pooch()
    
    def _initialize_pooch(self):
        """Initialize the Pooch downloader with registry."""
        cache_path = self._cache_dir if self._cache_dir else pooch.os_cache("geotessera")
        
        # Initialize main pooch for numpy embeddings
        self._pooch = pooch.create(
            path=cache_path,
            base_url=f"https://dl-1.tessera.wiki/{self.version}/global_0.1_degree_representation/",
            version=self.version,
            registry=None,
        )
        
        # Load the registry file for numpy embeddings
        with importlib.resources.open_text("geotessera", "registry_2024.txt") as registry_file:
            self._pooch.load_registry(registry_file)
        
        # Initialize land mask pooch for internal land/water mask files
        self._landmask_pooch = pooch.create(
            path=cache_path,
            base_url=f"https://dl-1.tessera.wiki/{self.version}/global_0.1_degree_tiff_all/",
            version=self.version,
            registry=None,
        )
        
        # Load land mask registry dynamically
        self._load_landmask_registry()
        
        # Parse and cache available embeddings
        self._parse_available_embeddings()
        self._parse_available_landmasks()
    
    def _load_landmask_registry(self):
        """Load the land mask registry file dynamically from the server.
        
        Note: These are internal land/water mask files used for coordinate
        alignment during numpy array merging operations, not user-facing TIFFs.
        """
        try:
            cache_path = self._cache_dir if self._cache_dir else pooch.os_cache("geotessera")
            
            # Use pooch.retrieve to get the registry file without known hash
            registry_file = pooch.retrieve(
                url=f"https://dl-1.tessera.wiki/{self.version}/global_0.1_degree_tiff_all/registry.txt",
                known_hash=None,
                fname="landmask_registry.txt",
                path=cache_path,
                progressbar=True
            )
            
            # Load the registry into the land mask pooch
            self._landmask_pooch.load_registry(registry_file)
            
        except Exception as e:
            print(f"Warning: Could not load land mask registry: {e}")
            # Continue without land mask support if registry loading fails
    
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
    
    def _fetch_landmask(self, lat: float, lon: float, progressbar: bool = True) -> str:
        """Fetch internal land mask file for a specific location.
        
        Note: This is an internal method for fetching land/water mask files
        used during numpy array merging operations. These are not user-facing TIFFs.
        
        Args:
            lat: Latitude coordinate
            lon: Longitude coordinate
            progressbar: Show download progress bar
            
        Returns:
            Path to the downloaded land mask file
        """
        if not self._landmask_pooch:
            raise RuntimeError("Land mask registry not loaded. Check initialization.")
        
        # Format coordinates to match file naming convention
        landmask_filename = f"grid_{lon:.2f}_{lat:.2f}.tiff"
        
        return self._landmask_pooch.fetch(landmask_filename, progressbar=progressbar)
    
    def _list_available_landmasks(self) -> Iterator[Tuple[float, float]]:
        """List all available internal land mask files as (lat, lon) tuples.
        
        Note: These are internal land/water mask files, not user-facing TIFFs.
        
        Returns:
            Iterator of tuples containing (latitude, longitude) for each available land mask
        """
        return iter(self._available_landmasks)
    
    def _count_available_landmasks(self) -> int:
        """Get the total number of available internal land mask files.
        
        Note: These are internal land/water mask files, not user-facing TIFFs.
        
        Returns:
            Total count of available land mask files
        """
        return len(self._available_landmasks)
    
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
    
    def _parse_available_landmasks(self):
        """Parse land mask registry to extract available land mask files as (lat, lon) tuples."""
        landmasks = []
        
        if not self._landmask_pooch or not self._landmask_pooch.registry:
            return
        
        for file_path in self._landmask_pooch.registry.keys():
            # Parse file path: e.g., "grid_0.15_52.05.tiff"
            if file_path.endswith('.tiff'):
                # Extract coordinates from filename
                filename = Path(file_path).name
                if filename.startswith('grid_'):
                    coords = filename[5:-5].split('_')  # Remove "grid_" prefix and ".tiff" suffix
                    if len(coords) == 2:
                        try:
                            lon = float(coords[0])
                            lat = float(coords[1])
                            landmasks.append((lat, lon))
                        except ValueError:
                            continue
        
        # Sort by lat, then lon for consistent ordering
        landmasks.sort(key=lambda x: (x[0], x[1]))
        self._available_landmasks = landmasks
    
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
                
                # Get actual tile dimensions
                tile_h, tile_w = tile_data.shape[:2]
                
                # Calculate end positions
                y_end = y_start + tile_h
                x_end = x_start + tile_w
                
                # Clip to mosaic bounds
                y_start_clipped = max(0, y_start)
                x_start_clipped = max(0, x_start)
                y_end_clipped = min(mosaic_height, y_end)
                x_end_clipped = min(mosaic_width, x_end)
                
                # Calculate tile region to copy
                tile_y_start = y_start_clipped - y_start
                tile_x_start = x_start_clipped - x_start
                tile_y_end = tile_y_start + (y_end_clipped - y_start_clipped)
                tile_x_end = tile_x_start + (x_end_clipped - x_start_clipped)
                
                # Place tile in mosaic if there's any overlap
                if y_end_clipped > y_start_clipped and x_end_clipped > x_start_clipped:
                    mosaic[y_start_clipped:y_end_clipped, x_start_clipped:x_end_clipped] = \
                        tile_data[tile_y_start:tile_y_end, tile_x_start:tile_x_end]
        
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
    
    def export_single_tile_as_tiff(self, lat: float, lon: float, output_path: str,
                                   year: int = 2024, bands: List[int] = [0, 1, 2],
                                   normalize: bool = True) -> str:
        """Export a single Tessera tile as a GeoTIFF.
        
        Args:
            lat: Latitude coordinate
            lon: Longitude coordinate
            output_path: Path for the output TIFF file
            year: Year of the embedding (default: 2024)
            bands: Three band indices to visualize as RGB
            normalize: Whether to normalize band values
            
        Returns:
            Path to the created TIFF file
        """
        try:
            import rasterio
            from rasterio.transform import from_bounds
        except ImportError:
            raise ImportError("Please install rasterio for TIFF export: pip install rasterio")
        
        # Fetch and dequantize the embedding
        data = self.fetch_embedding(lat=lat, lon=lon, year=year, progressbar=True)
        
        # Extract bands for visualization
        vis_data = data[:, :, bands].copy()
        
        # Normalize if requested
        if normalize:
            for i in range(vis_data.shape[2]):
                channel = vis_data[:, :, i]
                min_val = np.min(channel)
                max_val = np.max(channel)
                if max_val > min_val:
                    vis_data[:, :, i] = (channel - min_val) / (max_val - min_val)
        
        # Ensure we have valid RGB data in [0,1] range
        vis_data = np.clip(vis_data, 0, 1)
        
        # Convert to uint8 for TIFF export
        vis_data_uint8 = (vis_data * 255).astype(np.uint8)
        
        # Get dimensions
        height, width = vis_data.shape[:2]
        
        # Calculate geographic bounds (each tile covers 0.1 degrees)
        lon_min = lon
        lat_min = lat
        lon_max = lon + 0.1
        lat_max = lat + 0.1
        
        # Create georeferencing transform
        transform = from_bounds(lon_min, lat_min, lon_max, lat_max, width, height)
        
        # Write the GeoTIFF
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=3,
            dtype='uint8',
            crs='EPSG:4326',  # WGS84
            transform=transform,
            compress='lzw'
        ) as dst:
            # Write RGB bands
            for i in range(3):
                dst.write(vis_data_uint8[:, :, i], i + 1)
        
        print(f"Exported tile to {output_path}")
        print(f"Dimensions: {width}x{height} pixels")
        print(f"Geographic bounds: {lon_min:.4f}, {lat_min:.4f}, {lon_max:.4f}, {lat_max:.4f}")
        
        return output_path
    
    def merge_landmasks_for_region(self, bounds: Tuple[float, float, float, float], 
                              output_path: str, target_crs: str = "EPSG:4326") -> str:
        """Merge multiple internal land mask tiles for a region without coordinate skew.
        
        Note: This method uses internal land/water mask files for coordinate alignment
        during numpy array merging operations. The output is a binary land/water mask.
        
        This method uses proper coordinate reprojection to avoid skew issues
        when merging tiles from different UTM zones.
        
        Args:
            bounds: Tuple of (min_lon, min_lat, max_lon, max_lat) in WGS84
            output_path: Path for the output merged TIFF file
            target_crs: Target coordinate reference system (default: EPSG:4326)
            
        Returns:
            Path to the created merged TIFF file
        """
        try:
            import rasterio
            from rasterio.warp import calculate_default_transform, reproject
            from rasterio.enums import Resampling
            from rasterio.merge import merge
            from rasterio.transform import from_bounds
            import tempfile
            import shutil
        except ImportError:
            raise ImportError("Please install rasterio for TIFF merging: pip install rasterio")
        
        min_lon, min_lat, max_lon, max_lat = bounds
        
        # Find all land mask tiles that intersect with the bounds
        tiles_to_merge = []
        for lat, lon in self._list_available_landmasks():
            # Check if tile intersects with bounds (0.1 degree grid)
            tile_min_lon, tile_min_lat = lon, lat
            tile_max_lon, tile_max_lat = lon + 0.1, lat + 0.1
            
            if (tile_min_lon < max_lon and tile_max_lon > min_lon and
                tile_min_lat < max_lat and tile_max_lat > min_lat):
                tiles_to_merge.append((lat, lon))
        
        if not tiles_to_merge:
            raise ValueError("No land mask tiles found for the specified region")
        
        print(f"Found {len(tiles_to_merge)} land mask tiles to merge")
        
        # Download all required land mask tiles
        tile_paths = []
        for lat, lon in tiles_to_merge:
            try:
                tile_path = self._fetch_landmask(lat, lon, progressbar=True)
                tile_paths.append(tile_path)
            except Exception as e:
                print(f"Warning: Could not fetch land mask tile ({lat}, {lon}): {e}")
                continue
        
        if not tile_paths:
            raise ValueError("No land mask tiles could be downloaded")
        
        # Create temporary directory for reprojected tiles
        temp_dir = tempfile.mkdtemp(prefix="geotessera_merge_")
        
        try:
            # Reproject all tiles to target CRS if needed
            reprojected_paths = []
            
            for i, tile_path in enumerate(tile_paths):
                with rasterio.open(tile_path) as src:
                    if str(src.crs) != target_crs:
                        # Reproject to target CRS
                        reprojected_path = Path(temp_dir) / f"reprojected_{i}.tiff"
                        
                        # Calculate transform and dimensions for reprojection
                        transform, width, height = calculate_default_transform(
                            src.crs, target_crs, src.width, src.height, *src.bounds
                        )
                        
                        # Create reprojected raster
                        with rasterio.open(
                            reprojected_path,
                            'w',
                            driver='GTiff',
                            height=height,
                            width=width,
                            count=src.count,
                            dtype=src.dtypes[0],
                            crs=target_crs,
                            transform=transform,
                            compress='lzw'
                        ) as dst:
                            for band_idx in range(1, src.count + 1):
                                reproject(
                                    source=rasterio.band(src, band_idx),
                                    destination=rasterio.band(dst, band_idx),
                                    src_transform=src.transform,
                                    src_crs=src.crs,
                                    dst_transform=transform,
                                    dst_crs=target_crs,
                                    resampling=Resampling.nearest
                                )
                        
                        reprojected_paths.append(str(reprojected_path))
                    else:
                        reprojected_paths.append(tile_path)
            
            # Merge all reprojected tiles
            with rasterio.open(reprojected_paths[0]) as src:
                merged_array, merged_transform = merge([
                    rasterio.open(path) for path in reprojected_paths
                ])
                
                # Check if this appears to be a land/water mask (binary values)
                is_binary_mask = (merged_array.min() >= 0 and merged_array.max() <= 1 and 
                                 merged_array.dtype in ['uint8', 'int8'])
                
                if is_binary_mask:
                    print("Detected binary land/water mask - converting to visible format")
                    # Convert binary mask to visible grayscale (0->0, 1->255)
                    display_array = (merged_array * 255).astype('uint8')
                else:
                    display_array = merged_array
                
                # Write merged result
                with rasterio.open(
                    output_path,
                    'w',
                    driver='GTiff',
                    height=display_array.shape[1],
                    width=display_array.shape[2],
                    count=display_array.shape[0],
                    dtype=display_array.dtype,
                    crs=target_crs,
                    transform=merged_transform,
                    compress='lzw'
                ) as dst:
                    dst.write(display_array)
                
            print(f"Merged land mask saved to: {output_path}")
            return output_path
            
        finally:
            # Clean up temporary files
            shutil.rmtree(temp_dir)
    
    def merge_embeddings_for_region(self, bounds: Tuple[float, float, float, float], 
                                   output_path: str, target_crs: str = "EPSG:4326",
                                   bands: List[int] = [0, 1, 2], normalize: bool = True,
                                   year: int = 2024) -> str:
        """Merge multiple numpy embeddings for a region with proper coordinate alignment.
        
        This method follows the tessera-util approach: uses land mask TIFF files to get 
        proper coordinate transforms, creates temporary georeferenced TIFF files from 
        numpy embeddings, then merges them using rasterio.merge for perfect alignment.
        
        Args:
            bounds: Tuple of (min_lon, min_lat, max_lon, max_lat) in WGS84
            output_path: Path for the output merged TIFF file
            target_crs: Target coordinate reference system (default: EPSG:4326)
            bands: List of band indices to use for RGB visualization (default: [0,1,2])
            normalize: Whether to normalize band values (default: True)
            year: Year of embeddings to use (default: 2024)
            
        Returns:
            Path to the created merged TIFF file
        """
        try:
            import rasterio
            from rasterio.warp import calculate_default_transform, reproject
            from rasterio.enums import Resampling
            from rasterio.merge import merge
            import tempfile
            import shutil
        except ImportError:
            raise ImportError("Please install rasterio for embedding merging: pip install rasterio")
        
        min_lon, min_lat, max_lon, max_lat = bounds
        
        # Find all embedding tiles that intersect with the bounds
        tiles_to_merge = []
        for emb_year, lat, lon in self.list_available_embeddings():
            if emb_year != year:
                continue
                
            # Check if tile intersects with bounds (0.1 degree grid)
            tile_min_lon, tile_min_lat = lon, lat
            tile_max_lon, tile_max_lat = lon + 0.1, lat + 0.1
            
            if (tile_min_lon < max_lon and tile_max_lon > min_lon and
                tile_min_lat < max_lat and tile_max_lat > min_lat):
                tiles_to_merge.append((lat, lon))
        
        if not tiles_to_merge:
            raise ValueError(f"No embedding tiles found for the specified region in year {year}")
        
        print(f"Found {len(tiles_to_merge)} embedding tiles to merge for year {year}")
        
        # Create temporary directory for georeferenced TIFF files
        temp_dir = tempfile.mkdtemp(prefix="geotessera_embed_merge_")
        
        try:
            # Step 1: Create properly georeferenced temporary TIFF files
            temp_tiff_paths = []
            
            for lat, lon in tiles_to_merge:
                try:
                    # Get the numpy embedding
                    embedding = self.fetch_embedding(lat, lon, year, progressbar=True)
                    
                    # Get the corresponding land mask TIFF for coordinate information
                    landmask_path = self._fetch_landmask(lat, lon, progressbar=False)
                    
                    # Read coordinate information from the land mask TIFF
                    with rasterio.open(landmask_path) as landmask_src:
                        src_transform = landmask_src.transform
                        src_crs = landmask_src.crs
                        src_bounds = landmask_src.bounds
                        src_height, src_width = landmask_src.height, landmask_src.width
                    
                    # Extract and process the specified bands
                    if len(bands) == 3:
                        vis_data = embedding[:, :, bands].copy()
                    else:
                        raise ValueError("Exactly 3 bands must be specified for RGB visualization")
                    
                    # Normalize if requested
                    if normalize:
                        for i in range(3):
                            channel = vis_data[:, :, i]
                            min_val = np.min(channel)
                            max_val = np.max(channel)
                            if max_val > min_val:
                                vis_data[:, :, i] = (channel - min_val) / (max_val - min_val)
                    
                    # Ensure we have valid RGB data in [0,1] range and convert to uint8
                    vis_data = np.clip(vis_data, 0, 1)
                    vis_data_uint8 = (vis_data * 255).astype(np.uint8)
                    
                    # Create temporary georeferenced TIFF file
                    temp_tiff_path = Path(temp_dir) / f"embed_{lat:.2f}_{lon:.2f}.tiff"
                    
                    # Handle potential coordinate system differences and reprojection
                    if str(src_crs) != str(target_crs):
                        # Calculate transform for reprojection
                        dst_transform, dst_width, dst_height = calculate_default_transform(
                            src_crs, target_crs, src_width, src_height,
                            left=src_bounds.left, bottom=src_bounds.bottom,
                            right=src_bounds.right, top=src_bounds.top
                        )
                        
                        # Create reprojected array
                        dst_data = np.zeros((dst_height, dst_width, 3), dtype=np.uint8)
                        
                        # Reproject each band
                        for i in range(3):
                            reproject(
                                source=vis_data_uint8[:, :, i],
                                destination=dst_data[:, :, i],
                                src_transform=src_transform,
                                src_crs=src_crs,
                                dst_transform=dst_transform,
                                dst_crs=target_crs,
                                resampling=Resampling.bilinear  # Use bilinear for smoother results
                            )
                        
                        # Use reprojected data
                        final_data = dst_data
                        final_transform = dst_transform
                        final_crs = target_crs
                        final_height, final_width = dst_height, dst_width
                    else:
                        # Use original coordinate system
                        final_data = vis_data_uint8
                        final_transform = src_transform
                        final_crs = src_crs
                        final_height, final_width = vis_data_uint8.shape[:2]
                    
                    # Write georeferenced TIFF file
                    with rasterio.open(
                        temp_tiff_path,
                        'w',
                        driver='GTiff',
                        height=final_height,
                        width=final_width,
                        count=3,
                        dtype='uint8',
                        crs=final_crs,
                        transform=final_transform,
                        compress='lzw',
                        tiled=True,
                        blockxsize=256,
                        blockysize=256
                    ) as dst:
                        for i in range(3):
                            dst.write(final_data[:, :, i], i + 1)
                    
                    temp_tiff_paths.append(str(temp_tiff_path))
                    
                except Exception as e:
                    print(f"Warning: Could not process embedding tile ({lat}, {lon}): {e}")
                    continue
            
            if not temp_tiff_paths:
                raise ValueError("No embedding tiles could be processed")
            
            print(f"Created {len(temp_tiff_paths)} temporary georeferenced TIFF files")
            
            # Step 2: Use rasterio.merge to properly merge the georeferenced TIFF files
            print("Merging georeferenced TIFF files...")
            
            # Open all TIFF files for merging
            src_files = [rasterio.open(path) for path in temp_tiff_paths]
            
            try:
                # Merge the files
                merged_array, merged_transform = merge(src_files, method='first')
                
                # Write the merged result
                with rasterio.open(
                    output_path,
                    'w',
                    driver='GTiff',
                    height=merged_array.shape[1],
                    width=merged_array.shape[2],
                    count=merged_array.shape[0],
                    dtype=merged_array.dtype,
                    crs=target_crs,
                    transform=merged_transform,
                    compress='lzw'
                ) as dst:
                    dst.write(merged_array)
                
                print(f"Merged embedding visualization saved to: {output_path}")
                print(f"Dimensions: {merged_array.shape[2]}x{merged_array.shape[1]} pixels")
                
                return output_path
                
            finally:
                # Close all source files
                for src in src_files:
                    src.close()
            
        finally:
            # Clean up temporary files
            shutil.rmtree(temp_dir)
