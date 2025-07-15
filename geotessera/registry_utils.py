"""
Utilities for block-based registry management.

This module provides functions for organizing global grid data into 5x5 degree blocks
and managing registry files for efficient data access.
"""

import math
import re
from typing import Tuple, Optional


BLOCK_SIZE = 5  # 5x5 degree blocks


def get_block_coordinates(lon: float, lat: float) -> Tuple[int, int]:
    """
    Calculate the block coordinates for a given longitude and latitude.
    
    Args:
        lon: Longitude in decimal degrees
        lat: Latitude in decimal degrees
        
    Returns:
        tuple: (block_lon, block_lat) representing the lower-left corner of the block
    """
    block_lon = math.floor(lon / BLOCK_SIZE) * BLOCK_SIZE
    block_lat = math.floor(lat / BLOCK_SIZE) * BLOCK_SIZE
    return int(block_lon), int(block_lat)


def get_block_registry_filename(year: str, block_lon: int, block_lat: int) -> str:
    """
    Generate the registry filename for a specific block.
    
    Args:
        year: Year string (e.g., "2024")
        block_lon: Block longitude (lower-left corner)
        block_lat: Block latitude (lower-left corner)
        
    Returns:
        str: Registry filename like "registry_2024_lon-55_lat-25.txt"
    """
    # Format longitude and latitude to avoid negative zero
    lon_str = f"lon{block_lon}" if block_lon != 0 else "lon0"
    lat_str = f"lat{block_lat}" if block_lat != 0 else "lat0"
    return f"registry_{year}_{lon_str}_{lat_str}.txt"


def parse_grid_coordinates(filename: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract longitude and latitude from a grid filename.
    
    Args:
        filename: Grid filename like "grid_-50.55_-20.65"
        
    Returns:
        tuple: (lon, lat) as floats, or (None, None) if parsing fails
    """
    match = re.match(r'grid_(-?\d+\.\d+)_(-?\d+\.\d+)', filename)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None


def get_tile_name(lon: float, lat: float) -> str:
    """
    Generate the tile name for a specific coordinate.
    
    Args:
        lon: Longitude in decimal degrees
        lat: Latitude in decimal degrees
        
    Returns:
        str: Tile name like "grid_-50.55_-20.65"
    """
    return f"grid_{lon:.2f}_{lat:.2f}"


def get_registry_path_for_tile(registry_base_dir: str, year: str, lon: float, lat: float) -> str:
    """
    Get the full path to the registry file containing a specific tile.
    
    Args:
        registry_base_dir: Base directory for registry files (should be the parent directory)
        year: Year string (e.g., "2024")
        lon: Longitude in decimal degrees
        lat: Latitude in decimal degrees
        
    Returns:
        str: Full path to the registry file
    """
    import os
    block_lon, block_lat = get_block_coordinates(lon, lat)
    registry_filename = get_block_registry_filename(year, block_lon, block_lat)
    return os.path.join(registry_base_dir, "registry", registry_filename)


def get_tiles_registry_filename(block_lon: int, block_lat: int) -> str:
    """
    Generate the registry filename for a specific landmask tiles block.
    
    Args:
        block_lon: Block longitude (lower-left corner)
        block_lat: Block latitude (lower-left corner)
        
    Returns:
        str: Registry filename like "registry_tiles_lon-55_lat-25.txt"
    """
    # Format longitude and latitude to avoid negative zero
    lon_str = f"lon{block_lon}" if block_lon != 0 else "lon0"
    lat_str = f"lat{block_lat}" if block_lat != 0 else "lat0"
    return f"registry_tiles_{lon_str}_{lat_str}.txt"


def get_registry_path_for_tiles(registry_base_dir: str, lon: float, lat: float) -> str:
    """
    Get the full path to the tiles registry file containing a specific coordinate.
    
    Args:
        registry_base_dir: Base directory for registry files (should be the parent directory)
        lon: Longitude in decimal degrees
        lat: Latitude in decimal degrees
        
    Returns:
        str: Full path to the tiles registry file
    """
    import os
    block_lon, block_lat = get_block_coordinates(lon, lat)
    registry_filename = get_tiles_registry_filename(block_lon, block_lat)
    return os.path.join(registry_base_dir, "registry", registry_filename)


def get_all_blocks_in_range(min_lon: float, max_lon: float, 
                           min_lat: float, max_lat: float) -> list:
    """
    Get all block coordinates that intersect with a given bounding box.
    
    Args:
        min_lon: Minimum longitude
        max_lon: Maximum longitude
        min_lat: Minimum latitude
        max_lat: Maximum latitude
        
    Returns:
        list: List of (block_lon, block_lat) tuples
    """
    blocks = []
    
    # Get block coordinates for corners
    min_block_lon = math.floor(min_lon / BLOCK_SIZE) * BLOCK_SIZE
    max_block_lon = math.floor(max_lon / BLOCK_SIZE) * BLOCK_SIZE
    min_block_lat = math.floor(min_lat / BLOCK_SIZE) * BLOCK_SIZE
    max_block_lat = math.floor(max_lat / BLOCK_SIZE) * BLOCK_SIZE
    
    # Iterate through all blocks in range
    lon = min_block_lon
    while lon <= max_block_lon:
        lat = min_block_lat
        while lat <= max_block_lat:
            blocks.append((int(lon), int(lat)))
            lat += BLOCK_SIZE
        lon += BLOCK_SIZE
    
    return blocks