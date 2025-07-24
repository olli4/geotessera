"""Core module for accessing and working with Tessera geospatial embeddings.

This module provides the main GeoTessera class which interfaces with pre-computed
satellite embeddings from the Tessera foundation model. The embeddings compress
a full year of Sentinel-1 and Sentinel-2 observations into 128-dimensional
representation maps at 10m spatial resolution.

The module handles:
- Automatic data fetching and caching from remote servers
- Dequantization of compressed embeddings using scale factors
- Geographic tile discovery and intersection analysis
- Visualization and export of embeddings as GeoTIFF files
- Merging multiple tiles with proper coordinate alignment
"""

from pathlib import Path
from typing import Optional, Union, List, Tuple, Iterator, Dict
import os
import subprocess
import pooch
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.geometry

from .registry_utils import (
    get_block_coordinates,
    get_embeddings_registry_filename,
    get_landmasks_registry_filename,
)


# Base URL for Tessera data downloads
TESSERA_BASE_URL = "https://dl-2.tessera.wiki"


class GeoTessera:
    """Interface for accessing Tessera foundation model embeddings.

    GeoTessera provides access to pre-computed embeddings from the Tessera
    foundation model, which processes Sentinel-1 and Sentinel-2 satellite imagery
    to generate dense representation maps. Each embedding compresses a full year
    of temporal-spectral observations into 128 channels at 10m resolution.

    The embeddings are organized in a global 0.1-degree grid system, with each
    tile covering approximately 11km × 11km at the equator. Files are fetched
    on-demand and cached locally for efficient access.

    Attributes:
        version: Dataset version identifier (default: "v1")
        cache_dir: Local directory for caching downloaded files
        registry_dir: Local directory containing registry files (if None, downloads from remote)

    Example:
        >>> gt = GeoTessera()
        >>> # Fetch embeddings for Cambridge, UK
        >>> embedding = gt.fetch_embedding(lat=52.2053, lon=0.1218)
        >>> print(f"Shape: {embedding.shape}")  # (height, width, 128)
        >>> # Visualize as RGB composite
        >>> gt.visualize_embedding(embedding, bands=[10, 20, 30])
    """

    def __init__(
        self,
        version: str = "v1",
        cache_dir: Optional[Union[str, Path]] = None,
        registry_dir: Optional[Union[str, Path]] = None,
        auto_update: bool = False,
        manifests_repo_url: str = "https://github.com/ucam-eo/tessera-manifests.git",
    ):
        """Initialize GeoTessera client for accessing Tessera embeddings.

        Creates a client instance that can fetch and work with pre-computed
        satellite embeddings. Data is automatically cached locally after first
        download to improve performance.

        Args:
            version: Dataset version to use. Currently "v1" is available.
            cache_dir: Directory for caching downloaded files. If None, uses
                      the system's default cache directory (~/.cache/geotessera
                      on Unix-like systems).
            registry_dir: Local directory containing registry files. If provided,
                         registry files will be loaded from this directory instead
                         of being downloaded via pooch. Should point to directory
                         containing "registry" subdirectory with embeddings and
                         landmasks folders. If None, will check TESSERA_REGISTRY_DIR
                         environment variable, and if that's also not set, will
                         auto-clone the tessera-manifests repository.
            auto_update: If True, updates the tessera-manifests repository to
                        the latest version from upstream (main branch). Only
                        applies when using the auto-cloned manifests repository.
            manifests_repo_url: Git repository URL for tessera-manifests. Only used
                               when auto-cloning the manifests repository (when no
                               registry_dir is specified and TESSERA_REGISTRY_DIR is
                               not set). Defaults to the official repository.

        Raises:
            ValueError: If the specified version is not supported.

        Note:
            The client lazily loads registry files for each year as needed,
            improving startup performance when working with specific years.
        """
        self.version = version
        self._cache_dir = cache_dir
        self._auto_update = auto_update
        self._manifests_repo_url = manifests_repo_url
        self._registry_dir = self._resolve_registry_dir(registry_dir)
        self._pooch = None
        self._landmask_pooch = None
        self._available_embeddings = []
        self._available_landmasks = []
        self._loaded_blocks = (
            set()
        )  # Track which blocks have been loaded for embeddings
        self._loaded_tile_blocks = (
            set()
        )  # Track which blocks have been loaded for landmasks
        self._registry_base_dir = None  # Base directory for block registries
        self._registry_file = None  # Path to the master registry.txt file
        self._initialize_pooch()

    def _resolve_registry_dir(
        self, registry_dir: Optional[Union[str, Path]]
    ) -> Optional[str]:
        """Resolve the registry directory path from multiple sources.

        This method normalizes the registry directory path to always point to the
        directory containing the actual registry files (embeddings/, landmasks/).

        Priority order:
        1. Explicit registry_dir parameter
        2. TESSERA_REGISTRY_DIR environment variable
        3. Auto-clone tessera-manifests repository to cache dir

        Args:
            registry_dir: Directory containing registry files or parent directory
                         with 'registry' subdirectory

        Returns:
            Path to directory containing registry files, or None for remote-only mode
        """
        resolved_path = None

        # 1. Use explicit parameter if provided
        if registry_dir is not None:
            resolved_path = str(registry_dir)
        # 2. Check environment variable
        elif os.environ.get("TESSERA_REGISTRY_DIR"):
            resolved_path = os.environ.get("TESSERA_REGISTRY_DIR")
        # 3. Auto-clone tessera-manifests repository
        else:
            return (
                self._setup_tessera_manifests()
            )  # This already returns registry subdir

        # Normalize the path to point to the actual registry directory
        if resolved_path:
            registry_path = Path(resolved_path)

            # If the path contains a 'registry' subdirectory, use that
            if (registry_path / "registry").exists():
                return str(registry_path / "registry")
            # Otherwise assume the path already points to the registry directory
            else:
                return str(registry_path)

        return None

    def _setup_tessera_manifests(self) -> str:
        """Setup tessera-manifests repository in cache directory.

        Clones or updates the tessera-manifests repository from GitHub.

        Returns:
            Path to the tessera-manifests directory
        """
        cache_path = (
            self._cache_dir if self._cache_dir else pooch.os_cache("geotessera")
        )
        manifests_dir = Path(cache_path) / "tessera-manifests"

        if manifests_dir.exists():
            if self._auto_update:
                # Update existing repository
                try:
                    print(f"Updating tessera-manifests repository in {manifests_dir}")
                    subprocess.run(
                        ["git", "fetch", "origin"],
                        cwd=manifests_dir,
                        check=True,
                        capture_output=True,
                    )

                    subprocess.run(
                        ["git", "reset", "--hard", "origin/main"],
                        cwd=manifests_dir,
                        check=True,
                        capture_output=True,
                    )

                    print("✓ tessera-manifests updated to latest version")
                except subprocess.CalledProcessError as e:
                    print(f"Warning: Failed to update tessera-manifests: {e}")
        else:
            # Clone repository
            try:
                print(f"Cloning tessera-manifests repository to {manifests_dir}")
                subprocess.run(
                    ["git", "clone", self._manifests_repo_url, str(manifests_dir)],
                    check=True,
                    capture_output=True,
                )

                print("✓ tessera-manifests repository cloned successfully")
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to clone tessera-manifests repository: {e}")

        # Return the registry subdirectory path
        registry_dir = manifests_dir / "registry"
        return str(registry_dir)

    def _initialize_pooch(self):
        """Initialize Pooch data fetchers for embeddings and land masks.

        Sets up two Pooch instances:
        1. Main fetcher for numpy embedding files (.npy and _scales.npy)
        2. Land mask fetcher for GeoTIFF files containing binary land/water
           masks and coordinate reference system metadata

        Registry files are loaded lazily per year to improve performance.
        """
        cache_path = (
            self._cache_dir if self._cache_dir else pooch.os_cache("geotessera")
        )

        # Initialize main pooch for numpy embeddings
        self._pooch = pooch.create(
            path=cache_path,
            base_url=f"{TESSERA_BASE_URL}/{self.version}/global_0.1_degree_representation/",
            version=self.version,
            registry=None,
            env="TESSERA_DATA_DIR",
        )

        # Registry files will be loaded lazily when needed
        # This is handled by _ensure_year_loaded method

        # Initialize land mask pooch for landmask GeoTIFF files
        # These TIFFs serve dual purposes:
        # 1. Binary land/water distinction (pixel values 0=water, 1=land)
        # 2. Coordinate reference system metadata for proper georeferencing
        self._landmask_pooch = pooch.create(
            path=cache_path,
            base_url=f"{TESSERA_BASE_URL}/{self.version}/global_0.1_degree_tiff_all/",
            version=self.version,
            registry=None,
            env="TESSERA_DATA_DIR",  # CR:avsm FIXME this should be a separate subdir
        )

        # Load registry index for block-based registries
        self._load_registry_index()

        # Try to load tiles registry index
        self._load_tiles_registry_index()

    def _load_tiles_registry_index(self):
        """Load the registry index for block-based tile registries.

        Downloads and caches the registry index file that lists all
        available tile block registry files.
        """
        try:
            # The registry file should already be loaded by _load_registry_index
            # This method exists for compatibility but doesn't need to re-download
            pass

        except Exception as e:
            print(f"Warning: Could not load registry: {e}")
            # Continue without landmask support if registry loading fails

    def _load_registry_index(self):
        """Load the registry index for block-based registries.

        If registry_dir is provided, loads registry files from local directory.
        Otherwise downloads and caches the registry index file from remote.
        """
        if self._registry_dir:
            # Use local registry directory (already normalized to point to registry files)
            registry_path = Path(self._registry_dir)
            if not registry_path.exists():
                raise ValueError(f"Registry directory not found: {registry_path}")

            self._registry_base_dir = str(registry_path)

            # Look for master registry file in the local directory
            master_registry = registry_path / "registry.txt"
            if master_registry.exists():
                self._registry_file = str(master_registry)
            else:
                # No master registry file, we'll scan directories later
                self._registry_file = None
        else:
            # Original behavior: download from remote
            cache_path = (
                self._cache_dir if self._cache_dir else pooch.os_cache("geotessera")
            )
            self._registry_base_dir = cache_path

            # Download the master registry containing hashes of registry files
            self._registry_file = pooch.retrieve(
                url=f"{TESSERA_BASE_URL}/{self.version}/registry/registry.txt",
                known_hash=None,
                fname="registry.txt",
                path=cache_path,
                progressbar=True,
            )

    def _get_registry_hash(self, registry_filename: str) -> Optional[str]:
        """Get the hash for a specific registry file from the master registry.txt.

        Args:
            registry_filename: Name of the registry file to look up

        Returns:
            Hash string if found, None otherwise
        """
        try:
            if not self._registry_file or not Path(self._registry_file).exists():
                return None

            with open(self._registry_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        parts = line.split(" ", 1)
                        if len(parts) == 2 and parts[0] == registry_filename:
                            return parts[1]
            return None
        except Exception:
            return None

    def _ensure_block_loaded(self, year: int, lon: float, lat: float):
        """Ensure registry data for a specific block is loaded.

        Loads only the registry file containing the specific coordinates needed,
        providing efficient lazy loading of registry data.

        Args:
            year: Year to load (e.g., 2024)
            lon: Longitude in decimal degrees
            lat: Latitude in decimal degrees
        """
        block_lon, block_lat = get_block_coordinates(lon, lat)
        block_key = (year, block_lon, block_lat)

        if block_key in self._loaded_blocks:
            return

        registry_filename = get_embeddings_registry_filename(
            str(year), block_lon, block_lat
        )

        if self._registry_dir:
            # Load from local directory
            embeddings_dir = Path(self._registry_base_dir) / "embeddings"
            registry_file = embeddings_dir / registry_filename

            if not registry_file.exists():
                # Silently skip if file doesn't exist - it may not have data for this block
                return

            # Load the registry file directly (now in correct pooch format)
            self._pooch.load_registry(str(registry_file))
            self._loaded_blocks.add(block_key)
            self._parse_available_embeddings()
            return
        else:
            # Original behavior: download from remote
            # Get the hash from the master registry.txt file
            registry_hash = self._get_registry_hash(registry_filename)

            # Download the specific block registry file
            registry_url = (
                f"{TESSERA_BASE_URL}/{self.version}/registry/{registry_filename}"
            )
            registry_file = pooch.retrieve(
                url=registry_url,
                known_hash=registry_hash,
                fname=registry_filename,
                path=self._registry_base_dir,
                progressbar=False,  # Don't show progress for individual block downloads
            )

        # Load the registry into the pooch instance
        self._pooch.load_registry(registry_file)
        self._loaded_blocks.add(block_key)

        # Update available embeddings cache
        self._parse_available_embeddings()

    def _load_all_blocks(self):
        """Load all available block registries to build complete embedding list.

        This method is used when a complete listing of all embeddings is needed,
        such as for generating coverage maps. It scans the local registry directory
        or parses the master registry to find all block files and loads them.
        """
        try:
            if self._registry_dir:
                # Scan local embeddings directory for registry files
                embeddings_dir = Path(self._registry_base_dir) / "embeddings"
                if not embeddings_dir.exists():
                    print(f"Warning: Embeddings directory not found: {embeddings_dir}")
                    return

                # Find all embeddings registry files
                block_files = []
                for file_path in embeddings_dir.glob("embeddings_*.txt"):
                    if "_lon" in file_path.name and "_lat" in file_path.name:
                        block_files.append(file_path.name)

                print(f"Found {len(block_files)} block registry files to load")

                # Load each block registry
                for i, block_file in enumerate(block_files):
                    if (i + 1) % 100 == 0:  # Progress indicator every 100 blocks
                        print(f"Loading block registries: {i + 1}/{len(block_files)}")

                    try:
                        registry_file_path = embeddings_dir / block_file

                        # Load the registry file directly (now in correct pooch format)
                        self._pooch.load_registry(str(registry_file_path))

                        # Mark this block as loaded
                        # Parse filename format: embeddings_YYYY_lonXXX_latYYY.txt
                        # Examples: embeddings_2024_lon-15_lat10.txt, embeddings_2024_lon130_lat45.txt
                        parts = block_file.replace(".txt", "").split("_")
                        if len(parts) >= 4:
                            year = int(
                                parts[1]
                            )  # parts[0] is "embeddings", parts[1] is year

                            # Extract lon and lat values
                            lon_part = None
                            lat_part = None
                            for j, part in enumerate(parts):
                                if part.startswith("lon"):
                                    lon_part = part[3:]  # Remove 'lon' prefix
                                elif part.startswith("lat"):
                                    lat_part = part[3:]  # Remove 'lat' prefix

                            if lon_part and lat_part:
                                # Convert to block coordinates (assuming these are already block coordinates)
                                block_lon = int(lon_part)
                                block_lat = int(lat_part)
                                self._loaded_blocks.add((year, block_lon, block_lat))

                    except Exception as e:
                        print(
                            f"Warning: Failed to load block registry {block_file}: {e}"
                        )
                        continue

            else:
                # Original behavior: use master registry file
                if not self._registry_file or not Path(self._registry_file).exists():
                    print("Warning: Master registry not found")
                    return

                # Parse registry.txt to find all block registry files
                block_files = []
                with open(self._registry_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            parts = line.split(" ", 1)
                            if len(parts) == 2:
                                filename = parts[0]
                                # Look for embeddings registry files (format: embeddings_YYYY_lonXXX_latYYY.txt)
                                if (
                                    filename.startswith("embeddings_")
                                    and "_lon" in filename
                                    and "_lat" in filename
                                    and filename.endswith(".txt")
                                ):
                                    block_files.append(filename)

                print(f"Found {len(block_files)} block registry files to load")

                # Load each block registry
                for i, block_file in enumerate(block_files):
                    if (i + 1) % 100 == 0:  # Progress indicator every 100 blocks
                        print(f"Loading block registries: {i + 1}/{len(block_files)}")

                    try:
                        # Download the block registry file
                        registry_url = (
                            f"{TESSERA_BASE_URL}/{self.version}/registry/{block_file}"
                        )
                        registry_hash = self._get_registry_hash(block_file)

                        downloaded_file = pooch.retrieve(
                            url=registry_url,
                            known_hash=registry_hash,
                            fname=block_file,
                            path=self._registry_base_dir,
                            progressbar=False,  # Don't show progress for individual files
                        )

                        # Load the registry into the pooch instance
                        self._pooch.load_registry(downloaded_file)

                        # Mark this block as loaded
                        # Parse filename format: embeddings_YYYY_lonXXX_latYYY.txt
                        # Examples: embeddings_2024_lon-15_lat10.txt, embeddings_2024_lon130_lat45.txt
                        parts = block_file.replace(".txt", "").split("_")
                        if len(parts) >= 4:
                            year = int(
                                parts[1]
                            )  # parts[0] is "embeddings", parts[1] is year

                            # Extract lon and lat values
                            lon_part = None
                            lat_part = None
                            for j, part in enumerate(parts):
                                if part.startswith("lon"):
                                    lon_part = part[3:]  # Remove 'lon' prefix
                                elif part.startswith("lat"):
                                    lat_part = part[3:]  # Remove 'lat' prefix

                            if lon_part and lat_part:
                                # Convert to block coordinates (assuming these are already block coordinates)
                                block_lon = int(lon_part)
                                block_lat = int(lat_part)
                                self._loaded_blocks.add((year, block_lon, block_lat))

                    except Exception as e:
                        print(
                            f"Warning: Failed to load block registry {block_file}: {e}"
                        )
                        continue

            # Update available embeddings cache
            self._parse_available_embeddings()
            print(f"Loaded {len(self._available_embeddings)} total embeddings")

        except Exception as e:
            print(f"Error loading all blocks: {e}")

    def _load_blocks_for_region(
        self, bounds: Tuple[float, float, float, float], year: int
    ):
        """Load only the registry blocks needed for a specific region.

        This is much more efficient than loading all blocks globally when only
        working with a specific geographic region.

        Args:
            bounds: Geographic bounds as (min_lon, min_lat, max_lon, max_lat)
            year: Year of embeddings to load
        """
        from .registry_utils import get_all_blocks_in_range

        min_lon, min_lat, max_lon, max_lat = bounds

        # Get all blocks that intersect with the region
        required_blocks = get_all_blocks_in_range(min_lon, max_lon, min_lat, max_lat)

        print(
            f"Loading {len(required_blocks)} registry blocks for region bounds: "
            f"({min_lon:.4f}, {min_lat:.4f}, {max_lon:.4f}, {max_lat:.4f})"
        )

        # Load each required block
        blocks_loaded = 0
        for block_lon, block_lat in required_blocks:
            block_key = (year, block_lon, block_lat)

            if block_key not in self._loaded_blocks:
                # Use the center of the block to trigger loading
                center_lon = block_lon + 2.5  # Center of 5-degree block
                center_lat = block_lat + 2.5  # Center of 5-degree block

                try:
                    self._ensure_block_loaded(year, center_lon, center_lat)
                    blocks_loaded += 1
                except Exception as e:
                    print(
                        f"Warning: Failed to load block ({block_lon}, {block_lat}): {e}"
                    )

        print(
            f"Successfully loaded {blocks_loaded}/{len(required_blocks)} registry blocks"
        )

        # Update available embeddings cache
        self._parse_available_embeddings()

    def _ensure_tile_block_loaded(self, lon: float, lat: float):
        """Ensure registry data for a specific tile block is loaded.

        Loads only the registry file containing the specific coordinates needed
        for landmask tiles, providing efficient lazy loading.

        Args:
            lon: Longitude in decimal degrees
            lat: Latitude in decimal degrees
        """
        block_lon, block_lat = get_block_coordinates(lon, lat)
        block_key = (block_lon, block_lat)

        if block_key in self._loaded_tile_blocks:
            return

        registry_filename = get_landmasks_registry_filename(block_lon, block_lat)

        if self._registry_dir:
            # Load from local directory using block-based landmasks
            landmasks_dir = Path(self._registry_base_dir) / "landmasks"
            landmasks_registry_file = landmasks_dir / registry_filename

            if not landmasks_registry_file.exists():
                raise FileNotFoundError(
                    f"Landmasks registry file not found: {landmasks_registry_file}"
                )

            # Load the block-specific landmasks registry
            self._landmask_pooch.load_registry(str(landmasks_registry_file))
            self._parse_available_landmasks()

            # Mark this block as loaded
            self._loaded_tile_blocks.add(block_key)
            return
        else:
            # Original behavior: download from remote
            # Get the hash from the master registry.txt file
            registry_hash = self._get_registry_hash(registry_filename)

            # Download the specific tile block registry file
            registry_url = (
                f"{TESSERA_BASE_URL}/{self.version}/registry/{registry_filename}"
            )
            registry_file = pooch.retrieve(
                url=registry_url,
                known_hash=registry_hash,
                fname=registry_filename,
                path=self._registry_base_dir,
                progressbar=False,  # Don't show progress for individual block downloads
            )

        # Load the registry into the landmask pooch instance
        self._landmask_pooch.load_registry(registry_file)
        self._loaded_tile_blocks.add(block_key)

        # Update available landmasks cache
        self._parse_available_landmasks()

    def get_available_years(self) -> List[int]:
        """List all years with available Tessera embeddings.

        Returns the years that have been loaded in blocks, or the common
        range of years if no blocks have been loaded yet.

        Returns:
            List of years with available data, sorted in ascending order.

        Example:
            >>> gt = GeoTessera()
            >>> years = gt.get_available_years()
            >>> print(years)  # [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
        """
        loaded_years = {year for year, _, _ in self._loaded_blocks}
        if loaded_years:
            return sorted(loaded_years)
        else:
            # Return common range if no blocks loaded yet
            return list(range(2017, 2025))

    def fetch_embedding(
        self, lat: float, lon: float, year: int = 2024, progressbar: bool = True
    ) -> np.ndarray:
        """Fetch and dequantize Tessera embeddings for a geographic location.

        Downloads both the quantized embedding array and its corresponding scale
        factors, then performs dequantization by element-wise multiplication.
        The embeddings represent learned features from a full year of Sentinel-1
        and Sentinel-2 satellite observations.

        Args:
            lat: Latitude in decimal degrees. Will be rounded to nearest 0.1°
                 grid cell (e.g., 52.23 → 52.20).
            lon: Longitude in decimal degrees. Will be rounded to nearest 0.1°
                 grid cell (e.g., 0.17 → 0.15).
            year: Year of embeddings to fetch (2017-2024). Different years may
                  capture different environmental conditions.
            progressbar: Whether to display download progress. Set to False for
                        batch processing to reduce output verbosity.

        Returns:
            Dequantized embedding array of shape (height, width, 128) containing
            128-dimensional feature vectors for each 10m pixel. Typical tile
            dimensions are approximately 1100×1100 pixels.

        Raises:
            ValueError: If the requested tile is not available or year is invalid.
            IOError: If download fails after retries.

        Example:
            >>> gt = GeoTessera()
            >>> # Fetch embeddings for central London
            >>> embedding = gt.fetch_embedding(lat=51.5074, lon=-0.1278)
            >>> print(f"Tile shape: {embedding.shape}")
            >>> print(f"Feature dimensions: {embedding.shape[-1]} channels")

        Note:
            Files are cached after first download. Subsequent requests for the
            same tile will load from cache unless the cache is cleared.
        """
        # Ensure the registry for this coordinate block is loaded
        self._ensure_block_loaded(year, lon, lat)
        # Format coordinates to match file naming convention
        grid_name = f"grid_{lon:.2f}_{lat:.2f}"

        # Fetch both the main embedding and scales files
        embedding_path = f"{year}/{grid_name}/{grid_name}.npy"
        scales_path = f"{year}/{grid_name}/{grid_name}_scales.npy"

        embedding_file = self._pooch.fetch(embedding_path, progressbar=progressbar)
        scales_file = self._pooch.fetch(scales_path, progressbar=progressbar)

        # Load both files
        embedding = np.load(embedding_file)  # shape: (height, width, channels)
        scales = np.load(scales_file)  # shape: (height, width)

        # Dequantize by multiplying embedding by scales across all channels
        # Broadcasting scales from (height, width) to (height, width, channels)
        dequantized = embedding.astype(np.float32) * scales[:, :, np.newaxis]

        return dequantized

    def _fetch_landmask(self, lat: float, lon: float, progressbar: bool = True) -> str:
        """Download land mask GeoTIFF for coordinate reference information.

        Land mask files contain binary land/water data and crucial CRS metadata
        that defines the optimal projection for each tile. This metadata is used
        during tile merging to ensure proper geographic alignment.

        Args:
            lat: Latitude in decimal degrees (rounded to 0.1° grid).
            lon: Longitude in decimal degrees (rounded to 0.1° grid).
            progressbar: Whether to show download progress.

        Returns:
            Local file path to the cached land mask GeoTIFF.

        Raises:
            RuntimeError: If land mask registry was not loaded successfully.

        Note:
            This is an internal method used primarily during merge operations.
            End users typically don't need to call this directly.
        """
        if not self._landmask_pooch:
            raise RuntimeError("Land mask registry not loaded. Check initialization.")

        # Ensure the registry for this coordinate block is loaded
        self._ensure_tile_block_loaded(lon, lat)

        # Format coordinates to match file naming convention
        landmask_filename = f"grid_{lon:.2f}_{lat:.2f}.tiff"

        return self._landmask_pooch.fetch(landmask_filename, progressbar=progressbar)

    def _list_available_landmasks(self) -> Iterator[Tuple[float, float]]:
        """Iterate over available land mask tiles.

        Provides access to the catalog of land mask GeoTIFF files. Each file
        contains binary land/water classification and coordinate system metadata
        for its corresponding embedding tile.

        Returns:
            Iterator yielding (latitude, longitude) tuples for each available
            land mask, sorted by latitude then longitude.

        Note:
            Land masks are auxiliary data used primarily for coordinate alignment
            during tile merging operations.
        """
        return iter(self._available_landmasks)

    def _count_available_landmasks(self) -> int:
        """Count total number of available land mask files.

        Returns:
            Number of land mask GeoTIFF files in the registry.

        Note:
            Land mask availability may be limited compared to embedding tiles.
            Not all embedding tiles have corresponding land masks.
        """
        return len(self._available_landmasks)

    def _parse_available_embeddings(self):
        """Parse registry files to build index of available embedding tiles.

        Scans through loaded registry files to extract metadata about available
        tiles. Each tile is identified by year, latitude, and longitude. This
        method is called automatically when registry files are loaded.

        The index is stored as a sorted list of (year, lat, lon) tuples for
        efficient searching and iteration.
        """
        embeddings = []

        if self._pooch and self._pooch.registry:
            for file_path in self._pooch.registry.keys():
                # Only process .npy files that are not scale files
                if file_path.endswith(".npy") and not file_path.endswith("_scales.npy"):
                    # Parse file path: e.g., "2024/grid_0.15_52.05/grid_0.15_52.05.npy"
                    parts = file_path.split("/")
                    if len(parts) >= 3:
                        year_str = parts[0]
                        grid_name = parts[1]  # e.g., "grid_0.15_52.05"

                        try:
                            year = int(year_str)

                            # Extract coordinates from grid name
                            if grid_name.startswith("grid_"):
                                coords = grid_name[5:].split(
                                    "_"
                                )  # Remove "grid_" prefix
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
        """Parse land mask registry to index available GeoTIFF files.

        Land mask files serve dual purposes:
        1. Provide binary land/water classification (0=water, 1=land)
        2. Store coordinate reference system metadata for proper georeferencing

        This method builds an index of available land mask tiles as (lat, lon)
        tuples for efficient lookup during merge operations.
        """
        landmasks = []

        if not self._landmask_pooch or not self._landmask_pooch.registry:
            return

        for file_path in self._landmask_pooch.registry.keys():
            # Parse file path: e.g., "grid_0.15_52.05.tiff"
            if file_path.endswith(".tiff"):
                # Extract coordinates from filename
                filename = Path(file_path).name
                if filename.startswith("grid_"):
                    coords = filename[5:-5].split(
                        "_"
                    )  # Remove "grid_" prefix and ".tiff" suffix
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
        """Iterate over all available embedding tiles across all years.

        Provides an iterator over the complete catalog of available Tessera
        embeddings. Each tile covers a 0.1° × 0.1° area (approximately
        11km × 11km at the equator) and contains embeddings for one year.

        Returns:
            Iterator yielding (year, latitude, longitude) tuples for each
            available tile. Tiles are sorted by year, then latitude, then
            longitude.

        Example:
            >>> gt = GeoTessera()
            >>> # Count tiles in a specific region
            >>> uk_tiles = [(y, lat, lon) for y, lat, lon in gt.list_available_embeddings()
            ...             if 49 <= lat <= 59 and -8 <= lon <= 2]
            >>> print(f"UK tiles available: {len(uk_tiles)}")

        Note:
            On first call, this method will load registry files for all available
            years, which may take a few seconds.
        """
        # If no blocks have been loaded yet, load all available blocks
        if not self._loaded_blocks:
            self._load_all_blocks()

        return iter(self._available_embeddings)

    def count_available_embeddings(self) -> int:
        """Count total number of available embedding tiles across all years.

        Returns:
            Total number of available embedding tiles in the dataset.

        Example:
            >>> gt = GeoTessera()
            >>> total = gt.count_available_embeddings()
            >>> print(f"Total tiles available: {total:,}")
        """
        return len(self._available_embeddings)

    def get_tiles_for_topojson(
        self, topojson_path: Union[str, Path]
    ) -> List[Tuple[float, float, str]]:
        """Find all embedding tiles that intersect with region geometries.

        Analyzes a region file (GeoJSON, TopoJSON, Shapefile, GeoPackage) containing
        geographic features and identifies which Tessera embedding tiles overlap with
        those features. Uses improved geometry-based intersection without grid rounding
        that could miss edge tiles.

        Args:
            topojson_path: Path to a region file containing one or more geographic
                          features. Supports GeoJSON, TopoJSON, Shapefile (.shp),
                          and GeoPackage (.gpkg) formats.

        Returns:
            List of tuples containing (latitude, longitude, tile_path) for each
            tile that intersects with any geometry in the region file. The
            tile_path can be used with the Pooch fetcher.

        Example:
            >>> gt = GeoTessera()
            >>> # Find tiles covering a region (any supported format)
            >>> tiles = gt.get_tiles_for_topojson("boundary.shp")
            >>> print(f"Need {len(tiles)} tiles to cover the region")

        Note:
            This method now uses precise geometric intersection testing without
            grid rounding that could cause edge clipping issues. It returns tiles
            for all available years - use find_tiles_for_geometry() if you need
            year-specific filtering.
        """
        # Load region using general I/O utility
        gdf = gpd.read_file(topojson_path)

        # Ensure it's in the correct CRS
        if gdf.crs != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")

        # Create a unified geometry (union of all features)
        unified_geom = gdf.unary_union
        from shapely.geometry import box

        # Find intersecting tiles across all available years
        overlapping_tiles = []
        for year, lat, lon in self.list_available_embeddings():
            # Create tile bounding box (tile coordinates represent center, so extend ±0.05°)
            tile_box = box(lon - 0.05, lat - 0.05, lon + 0.05, lat + 0.05)

            # Check intersection with precise geometry testing
            if unified_geom.intersects(tile_box):
                tile_path = (
                    f"{year}/grid_{lon:.2f}_{lat:.2f}/grid_{lon:.2f}_{lat:.2f}.npy"
                )
                overlapping_tiles.append((lat, lon, tile_path))

        return overlapping_tiles

    def visualize_topojson_as_tiff(
        self,
        topojson_path: Union[str, Path],
        output_path: str = "topojson_tiles.tiff",
        bands: List[int] = [0, 1, 2],
        normalize: bool = True,
    ) -> str:
        """Create a GeoTIFF mosaic of embeddings covering a TopoJSON region.

        Generates a georeferenced TIFF image by mosaicking all Tessera tiles that
        intersect with the geometries in a TopoJSON file. The output is a clean
        satellite-style visualization without any overlays or decorations.

        Args:
            topojson_path: Path to TopoJSON file defining the region of interest.
            output_path: Output filename for the GeoTIFF (default: "topojson_tiles.tiff").
            bands: Three embedding channel indices to map to RGB. Default [0,1,2]
                   uses the first three channels. Try different combinations to
                   highlight different features.
            normalize: If True, normalizes each band to 0-1 range for better
                      contrast. If False, uses raw embedding values.

        Returns:
            Path to the created GeoTIFF file.

        Raises:
            ImportError: If rasterio is not installed.
            ValueError: If no tiles overlap with the TopoJSON region.

        Example:
            >>> gt = GeoTessera()
            >>> # Create false-color image of a national park
            >>> gt.visualize_topojson_as_tiff(
            ...     "park_boundary.json",
            ...     "park_tessera.tiff",
            ...     bands=[10, 20, 30]  # Custom band combination
            ... )

        Note:
            The output TIFF includes georeferencing information and can be
            opened in GIS software like QGIS or ArcGIS. Large regions may
            take significant time to process and require substantial memory.
        """
        try:
            import rasterio
            from rasterio.transform import from_bounds
        except ImportError:
            raise ImportError(
                "Please install rasterio and pillow for TIFF export: pip install rasterio pillow"
            )

        # Read the TopoJSON file
        gpd.read_file(topojson_path)

        # Get overlapping tiles
        tiles = self.get_tiles_for_topojson(topojson_path)

        if not tiles:
            print("No overlapping tiles found")
            return output_path

        # Calculate bounding box for all tiles
        lon_min = min(lon - 0.05 for _, lon, _ in tiles)
        lat_min = min(lat - 0.05 for lat, _, _ in tiles)
        lon_max = max(lon + 0.05 for _, lon, _ in tiles)
        lat_max = max(lat + 0.05 for lat, _, _ in tiles)

        # Download and process each tile
        tile_data_dict = {}
        print(f"Processing {len(tiles)} tiles for TIFF export...")

        for i, (lat, lon, tile_path) in enumerate(tiles):
            print(f"Processing tile {i + 1}/{len(tiles)}: ({lat:.2f}, {lon:.2f})")

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
                            vis_data[:, :, j] = (channel - min_val) / (
                                max_val - min_val
                            )

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
                y_start = int(
                    (lat_max - lat - 0.1) * pixels_per_degree_lat
                )  # Flip Y axis

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
                    mosaic[
                        y_start_clipped:y_end_clipped, x_start_clipped:x_end_clipped
                    ] = tile_data[tile_y_start:tile_y_end, tile_x_start:tile_x_end]

        # Convert to uint8 for TIFF export
        mosaic_uint8 = (mosaic * 255).astype(np.uint8)

        # Create georeferencing transform
        transform = from_bounds(
            lon_min, lat_min, lon_max, lat_max, mosaic_width, mosaic_height
        )

        # Write the GeoTIFF
        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=mosaic_height,
            width=mosaic_width,
            count=3,
            dtype="uint8",
            crs="EPSG:4326",  # WGS84
            transform=transform,
            compress="lzw",
        ) as dst:
            # Write RGB bands
            for i in range(3):
                dst.write(mosaic_uint8[:, :, i], i + 1)

        print(f"Exported high-resolution TIFF to {output_path}")
        print(f"Dimensions: {mosaic_width}x{mosaic_height} pixels")
        print(
            f"Geographic bounds: {lon_min:.4f}, {lat_min:.4f}, {lon_max:.4f}, {lat_max:.4f}"
        )

        return output_path

    def export_single_tile_as_tiff(
        self,
        lat: float,
        lon: float,
        output_path: str,
        year: int = 2024,
        bands: List[int] = [0, 1, 2],
        normalize: bool = True,
    ) -> str:
        """Export a single Tessera embedding tile as a georeferenced GeoTIFF.

        Creates a GeoTIFF file from a single embedding tile, selecting three
        channels to visualize as RGB. The output includes proper georeferencing
        metadata for use in GIS applications.

        Args:
            lat: Latitude of tile in decimal degrees (rounded to 0.1° grid).
            lon: Longitude of tile in decimal degrees (rounded to 0.1° grid).
            output_path: Filename for the output GeoTIFF.
            year: Year of embeddings to export (2017-2024).
            bands: Three channel indices to map to RGB. Each index must be
                   between 0-127. Different combinations highlight different
                   features (e.g., vegetation, water, urban areas).
            normalize: If True, stretches values to use full 0-255 range for
                      better visualization. If False, preserves relative values.

        Returns:
            Path to the created GeoTIFF file.

        Raises:
            ImportError: If rasterio is not installed.
            ValueError: If bands list doesn't contain exactly 3 indices.

        Example:
            >>> gt = GeoTessera()
            >>> # Export a tile over Paris with custom visualization
            >>> gt.export_single_tile_as_tiff(
            ...     lat=48.85, lon=2.35,
            ...     output_path="paris_2024.tiff",
            ...     bands=[25, 50, 75]  # Custom band selection
            ... )

        Note:
            Output files can be large (typically 10-50 MB per tile). The GeoTIFF
            uses LZW compression to reduce file size.
        """
        try:
            import rasterio
            from rasterio.transform import from_bounds
        except ImportError:
            raise ImportError(
                "Please install rasterio for TIFF export: pip install rasterio"
            )

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
        lon_min = lon - 0.05
        lat_min = lat - 0.05
        lon_max = lon + 0.05
        lat_max = lat + 0.05

        # Create georeferencing transform
        transform = from_bounds(lon_min, lat_min, lon_max, lat_max, width, height)

        # Write the GeoTIFF
        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=3,
            dtype="uint8",
            crs="EPSG:4326",  # WGS84
            transform=transform,
            compress="lzw",
        ) as dst:
            # Write RGB bands
            for i in range(3):
                dst.write(vis_data_uint8[:, :, i], i + 1)

        print(f"Exported tile to {output_path}")
        print(f"Dimensions: {width}x{height} pixels")
        print(
            f"Geographic bounds: {lon_min:.4f}, {lat_min:.4f}, {lon_max:.4f}, {lat_max:.4f}"
        )

        return output_path

    def _merge_landmasks_for_region(
        self,
        bounds: Tuple[float, float, float, float],
        output_path: str,
        target_crs: str = "EPSG:4326",
    ) -> str:
        """Merge land mask tiles for a geographic region with proper alignment.

        Combines multiple land mask GeoTIFF tiles into a single file, handling
        coordinate system differences between tiles. Each tile may use a different
        optimal projection (e.g., different UTM zones), so this method reprojects
        all tiles to a common coordinate system before merging.

        The land masks provide:
        - Binary classification: 0 = water, 1 = land
        - Coordinate system metadata for accurate georeferencing
        - Projection information to avoid coordinate skew

        Args:
            bounds: Geographic bounds as (min_lon, min_lat, max_lon, max_lat)
                    in WGS84 decimal degrees.
            output_path: Filename for the merged GeoTIFF output.
            target_crs: Target coordinate reference system. Default "EPSG:4326"
                       (WGS84). Can be any CRS supported by rasterio.

        Returns:
            Path to the created merged land mask file.

        Raises:
            ImportError: If rasterio is not installed.
            ValueError: If no land mask tiles are found for the region.

        Note:
            This is an internal method used by merge_embeddings_for_region().
            Binary masks are automatically converted to visible grayscale
            (0 → 0, 1 → 255) for better visualization.
        """
        try:
            import rasterio
            from rasterio.warp import calculate_default_transform, reproject
            from rasterio.enums import Resampling
            from rasterio.merge import merge
            import tempfile
            import shutil
        except ImportError:
            raise ImportError(
                "Please install rasterio for TIFF merging: pip install rasterio"
            )

        min_lon, min_lat, max_lon, max_lat = bounds

        # Find all land mask tiles that intersect with the bounds
        tiles_to_merge = []
        for lat, lon in self._list_available_landmasks():
            # Check if tile intersects with bounds (0.1 degree grid)
            tile_min_lon, tile_min_lat = lon - 0.05, lat - 0.05
            tile_max_lon, tile_max_lat = lon + 0.05, lat + 0.05

            if (
                tile_min_lon < max_lon
                and tile_max_lon > min_lon
                and tile_min_lat < max_lat
                and tile_max_lat > min_lat
            ):
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
                            "w",
                            driver="GTiff",
                            height=height,
                            width=width,
                            count=src.count,
                            dtype=src.dtypes[0],
                            crs=target_crs,
                            transform=transform,
                            compress="lzw",
                        ) as dst:
                            for band_idx in range(1, src.count + 1):
                                reproject(
                                    source=rasterio.band(src, band_idx),
                                    destination=rasterio.band(dst, band_idx),
                                    src_transform=src.transform,
                                    src_crs=src.crs,
                                    dst_transform=transform,
                                    dst_crs=target_crs,
                                    resampling=Resampling.nearest,
                                )

                        reprojected_paths.append(str(reprojected_path))
                    else:
                        reprojected_paths.append(tile_path)

            # Merge all reprojected tiles
            with rasterio.open(reprojected_paths[0]) as src:
                merged_array, merged_transform = merge(
                    [rasterio.open(path) for path in reprojected_paths]
                )

                # Check if this appears to be a land/water mask (binary values)
                is_binary_mask = (
                    merged_array.min() >= 0
                    and merged_array.max() <= 1
                    and merged_array.dtype in ["uint8", "int8"]
                )

                if is_binary_mask:
                    print(
                        "Detected binary land/water mask - converting to visible format"
                    )
                    # Convert binary mask to visible grayscale (0->0, 1->255)
                    display_array = (merged_array * 255).astype("uint8")
                else:
                    display_array = merged_array

                # Write merged result
                with rasterio.open(
                    output_path,
                    "w",
                    driver="GTiff",
                    height=display_array.shape[1],
                    width=display_array.shape[2],
                    count=display_array.shape[0],
                    dtype=display_array.dtype,
                    crs=target_crs,
                    transform=merged_transform,
                    compress="lzw",
                ) as dst:
                    dst.write(display_array)

            print(f"Merged land mask saved to: {output_path}")
            return output_path

        finally:
            # Clean up temporary files
            shutil.rmtree(temp_dir)

    def merge_embeddings_for_region(
        self,
        bounds: Tuple[float, float, float, float],
        output_path: str,
        target_crs: str = "EPSG:4326",
        bands: List[int] = [0, 1, 2],
        normalize: bool = True,
        year: int = 2024,
    ) -> str:
        """Create a seamless mosaic of Tessera embeddings for a geographic region.

        Merges multiple embedding tiles into a single georeferenced GeoTIFF,
        handling coordinate system differences and ensuring perfect alignment.
        This method uses land mask files to obtain optimal projection metadata
        for each tile, preventing coordinate skew when tiles span different
        UTM zones.

        The process:
        1. Identifies all tiles intersecting the bounding box
        2. Downloads embeddings and corresponding land masks
        3. Creates georeferenced temporary files using land mask CRS metadata
        4. Reprojects tiles to common coordinate system if needed
        5. Merges all tiles into seamless mosaic
        6. Applies normalization across entire mosaic if requested

        Args:
            bounds: Region bounds as (min_lon, min_lat, max_lon, max_lat) in
                    decimal degrees. Example: (-0.2, 51.4, 0.1, 51.6) for London.
            output_path: Filename for the output GeoTIFF mosaic.
            target_crs: Coordinate system for output. Default "EPSG:4326" (WGS84).
                       Use local projections (e.g., UTM) for accurate area measurements.
            bands: Three channel indices to visualize as RGB. Must be in range
                   0-127. Different combinations highlight different features.
            normalize: If True, applies global normalization across all merged
                      tiles for consistent visualization. If False, preserves
                      original embedding values.
            year: Year of embeddings to merge (2017-2024).

        Returns:
            Path to the created mosaic GeoTIFF file.

        Raises:
            ImportError: If rasterio is not installed.
            ValueError: If no tiles found for region or invalid parameters.
            RuntimeError: If land masks are not available for alignment.

        Example:
            >>> gt = GeoTessera()
            >>> # Create mosaic of San Francisco Bay Area
            >>> bounds = (-122.6, 37.2, -121.7, 38.0)
            >>> gt.merge_embeddings_for_region(
            ...     bounds=bounds,
            ...     output_path="sf_bay_tessera.tiff",
            ...     bands=[30, 60, 90],  # False color visualization
            ...     normalize=True
            ... )

        Note:
            Large regions require significant memory and processing time.
            The output file includes full georeferencing metadata and can
            be used in any GIS software. Normalization is applied globally
            across all tiles to ensure consistent coloring.
        """
        try:
            import rasterio
            from rasterio.warp import calculate_default_transform, reproject
            from rasterio.enums import Resampling
            from rasterio.merge import merge
            import tempfile
            import shutil
        except ImportError:
            raise ImportError(
                "Please install rasterio for embedding merging: pip install rasterio"
            )

        min_lon, min_lat, max_lon, max_lat = bounds

        # Load only the registry blocks needed for this region (much more efficient)
        self._load_blocks_for_region(bounds, year)

        # Find all embedding tiles that intersect with the bounds
        tiles_to_merge = []
        for emb_year, lat, lon in self._available_embeddings:
            if emb_year != year:
                continue

            # Check if tile intersects with bounds (0.1 degree grid)
            tile_min_lon, tile_min_lat = lon - 0.05, lat - 0.05
            tile_max_lon, tile_max_lat = lon + 0.05, lat + 0.05

            if (
                tile_min_lon < max_lon
                and tile_max_lon > min_lon
                and tile_min_lat < max_lat
                and tile_max_lat > min_lat
            ):
                tiles_to_merge.append((lat, lon))

        if not tiles_to_merge:
            raise ValueError(
                f"No embedding tiles found for the specified region in year {year}"
            )

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

                    # Get the corresponding landmask GeoTIFF for coordinate information
                    # The landmask TIFF provides the optimal projection metadata for this tile
                    landmask_path = self._fetch_landmask(lat, lon, progressbar=False)

                    # Read coordinate information from the landmask GeoTIFF metadata
                    with rasterio.open(landmask_path) as landmask_src:
                        src_transform = landmask_src.transform
                        src_crs = landmask_src.crs
                        src_bounds = landmask_src.bounds
                        src_height, src_width = landmask_src.height, landmask_src.width

                    # Extract the specified bands
                    if len(bands) == 3:
                        vis_data = embedding[:, :, bands].copy()
                    else:
                        raise ValueError(
                            "Exactly 3 bands must be specified for RGB visualization"
                        )

                    # Keep data as float32 for now - normalization happens after merging
                    vis_data = vis_data.astype(np.float32)

                    # Create temporary georeferenced TIFF file
                    temp_tiff_path = Path(temp_dir) / f"embed_{lat:.2f}_{lon:.2f}.tiff"

                    # Handle potential coordinate system differences and reprojection
                    if str(src_crs) != str(target_crs):
                        # Calculate transform for reprojection
                        dst_transform, dst_width, dst_height = (
                            calculate_default_transform(
                                src_crs,
                                target_crs,
                                src_width,
                                src_height,
                                left=src_bounds.left,
                                bottom=src_bounds.bottom,
                                right=src_bounds.right,
                                top=src_bounds.top,
                            )
                        )

                        # Create reprojected array
                        dst_data = np.zeros(
                            (dst_height, dst_width, 3), dtype=np.float32
                        )

                        # Reproject each band
                        for i in range(3):
                            reproject(
                                source=vis_data[:, :, i],
                                destination=dst_data[:, :, i],
                                src_transform=src_transform,
                                src_crs=src_crs,
                                dst_transform=dst_transform,
                                dst_crs=target_crs,
                                resampling=Resampling.bilinear,  # Use bilinear for smoother results
                            )

                        # Use reprojected data
                        final_data = dst_data
                        final_transform = dst_transform
                        final_crs = target_crs
                        final_height, final_width = dst_height, dst_width
                    else:
                        # Use original coordinate system
                        final_data = vis_data
                        final_transform = src_transform
                        final_crs = src_crs
                        final_height, final_width = vis_data.shape[:2]

                    # Write georeferenced TIFF file (as float32 for now)
                    with rasterio.open(
                        temp_tiff_path,
                        "w",
                        driver="GTiff",
                        height=final_height,
                        width=final_width,
                        count=3,
                        dtype="float32",
                        crs=final_crs,
                        transform=final_transform,
                        compress="lzw",
                        tiled=True,
                        blockxsize=256,
                        blockysize=256,
                    ) as dst:
                        for i in range(3):
                            dst.write(final_data[:, :, i], i + 1)

                    temp_tiff_paths.append(str(temp_tiff_path))

                except Exception as e:
                    # All errors during tile processing should be fatal
                    raise RuntimeError(
                        f"Failed to process embedding tile ({lat}, {lon}): {e}"
                    ) from e

            if not temp_tiff_paths:
                raise ValueError("No embedding tiles could be processed")

            print(f"Created {len(temp_tiff_paths)} temporary georeferenced TIFF files")

            # Step 2: Use rasterio.merge to properly merge the georeferenced TIFF files
            print("Merging georeferenced TIFF files...")

            # Open all TIFF files for merging
            src_files = [rasterio.open(path) for path in temp_tiff_paths]

            try:
                # Merge the files
                merged_array, merged_transform = merge(src_files, method="first")

                # Apply global normalization after merging if requested
                if normalize:
                    print("Applying global normalization across all merged tiles...")
                    for band_idx in range(merged_array.shape[0]):  # For each band
                        band_data = merged_array[band_idx]

                        # Only normalize non-zero pixels to preserve background
                        mask = band_data != 0
                        if np.any(mask):
                            # Get global min/max for this band across all tiles
                            min_val = np.min(band_data[mask])
                            max_val = np.max(band_data[mask])

                            if max_val > min_val:
                                # Apply normalization only to non-zero pixels
                                normalized = (band_data[mask] - min_val) / (
                                    max_val - min_val
                                )
                                band_data[mask] = normalized

                # Convert to uint8 for final output
                # Clip to [0,1] range first, then scale to [0,255]
                merged_array = np.clip(merged_array, 0, 1)
                merged_array_uint8 = (merged_array * 255).astype(np.uint8)

                # Write the merged result
                with rasterio.open(
                    output_path,
                    "w",
                    driver="GTiff",
                    height=merged_array_uint8.shape[1],
                    width=merged_array_uint8.shape[2],
                    count=merged_array_uint8.shape[0],
                    dtype="uint8",
                    crs=target_crs,
                    transform=merged_transform,
                    compress="lzw",
                ) as dst:
                    dst.write(merged_array_uint8)

                print(f"Merged embedding visualization saved to: {output_path}")
                print(
                    f"Dimensions: {merged_array_uint8.shape[2]}x{merged_array_uint8.shape[1]} pixels"
                )

                return output_path

            finally:
                # Close all source files
                for src in src_files:
                    src.close()

        finally:
            # Clean up temporary files
            shutil.rmtree(temp_dir)

    def find_tiles_for_geometry(
        self,
        geometry: Union[gpd.GeoDataFrame, "shapely.geometry.BaseGeometry"],
        year: int = 2024,
    ) -> List[Tuple[float, float]]:
        """Find all available tiles intersecting with a given geometry.

        Args:
            geometry: A GeoDataFrame or Shapely geometry (must be in EPSG:4326)
            year: Year of embeddings to search

        Returns:
            List of (lat, lon) tuples for tiles that intersect the geometry
        """
        from shapely.geometry import box

        # Convert to GeoDataFrame if needed
        if hasattr(geometry, "bounds"):  # Shapely geometry
            gdf = gpd.GeoDataFrame([1], geometry=[geometry], crs="EPSG:4326")
        elif isinstance(geometry, gpd.GeoDataFrame):
            if geometry.crs != "EPSG:4326":
                gdf = geometry.to_crs("EPSG:4326")
            else:
                gdf = geometry
        else:
            raise TypeError("geometry must be a GeoDataFrame or Shapely geometry")

        # Get unified geometry
        unified_geom = gdf.unary_union

        # Find intersecting tiles
        tiles = []
        for tile_year, lat, lon in self.list_available_embeddings():
            if tile_year != year:
                continue

            # Create tile bounding box (tile coordinates represent center, so extend ±0.05°)
            tile_box = box(lon - 0.05, lat - 0.05, lon + 0.05, lat + 0.05)

            # Check intersection
            if unified_geom.intersects(tile_box):
                tiles.append((lat, lon))

        return tiles

    def extract_points(
        self,
        points: Union[List[Dict], pd.DataFrame, gpd.GeoDataFrame],
        year: int = 2024,
        include_coords: bool = False,
        progressbar: bool = True,
    ) -> pd.DataFrame:
        """Extract embedding values at specific point locations.

        This method efficiently extracts embeddings for multiple points by:
        1. Grouping points by their containing tiles
        2. Loading each tile only once
        3. Extracting values for all points within that tile

        Args:
            points: Points with 'lat'/'lon' columns/keys. Can include 'label' or other metadata.
                   Accepts list of dicts, pandas DataFrame, or GeoDataFrame.
            year: Year of embeddings to use
            include_coords: If True, includes lat/lon in output DataFrame
            progressbar: Show progress bar during extraction

        Returns:
            DataFrame with embeddings (128 columns named 'emb_0' to 'emb_127') plus any metadata from input.
            Points that fall outside available tiles will be excluded from results.
        """
        import rasterio
        from pyproj import Transformer
        from collections import defaultdict

        # Convert input to consistent format
        if isinstance(points, list):
            points_df = pd.DataFrame(points)
        elif isinstance(points, gpd.GeoDataFrame):
            # Extract coordinates from geometry if needed
            if "lat" not in points.columns or "lon" not in points.columns:
                points = points.copy()
                points["lon"] = points.geometry.x
                points["lat"] = points.geometry.y
            points_df = pd.DataFrame(points.drop(columns="geometry"))
        else:
            points_df = points.copy()

        # Validate required columns
        if "lat" not in points_df.columns or "lon" not in points_df.columns:
            raise ValueError("Input must have 'lat' and 'lon' columns")

        # Group points by potential tiles
        points_by_tile = defaultdict(list)
        for idx, point in points_df.iterrows():
            # Find potential tiles (considering edge cases)
            base_lat = np.floor(point["lat"] * 10) / 10
            base_lon = np.floor(point["lon"] * 10) / 10

            # Check up to 4 adjacent tiles for edge cases
            for lat_offset in [0, -0.1]:
                for lon_offset in [0, -0.1]:
                    tile_lat = base_lat + lat_offset
                    tile_lon = base_lon + lon_offset
                    points_by_tile[(tile_lat, tile_lon)].append((idx, point))

        # Process points tile by tile
        results = []
        processed_indices = set()

        if progressbar:
            from tqdm import tqdm

            tile_iterator = tqdm(points_by_tile.items(), desc="Processing tiles")
        else:
            tile_iterator = points_by_tile.items()

        for (tile_lat, tile_lon), tile_points in tile_iterator:
            try:
                # Skip if tile doesn't exist
                if not any(
                    t_year == year and t_lat == tile_lat and t_lon == tile_lon
                    for t_year, t_lat, t_lon in self.list_available_embeddings()
                ):
                    continue

                # Fetch embedding and landmask for georeferencing
                embedding = self.fetch_embedding(
                    tile_lat, tile_lon, year, progressbar=False
                )
                landmask_path = self._fetch_landmask(
                    tile_lat, tile_lon, progressbar=False
                )

                # Get georeferencing info
                with rasterio.open(landmask_path) as src:
                    transformer = Transformer.from_crs(
                        "EPSG:4326", src.crs, always_xy=True
                    )
                    h, w = src.height, src.width

                    # Process each point
                    for idx, point in tile_points:
                        if (
                            idx in processed_indices
                        ):  # Skip if already processed by another tile
                            continue

                        # Transform coordinates
                        px, py = transformer.transform(point["lon"], point["lat"])
                        row, col = src.index(px, py)

                        # Check if point is within tile bounds
                        if 0 <= row < h and 0 <= col < w:
                            # Extract embedding vector
                            emb_vector = embedding[row, col]

                            # Build result row
                            result = {f"emb_{i}": emb_vector[i] for i in range(128)}

                            # Add metadata
                            for col_name in points_df.columns:
                                if col_name not in ["lat", "lon"] or include_coords:
                                    result[col_name] = point[col_name]

                            results.append(result)
                            processed_indices.add(idx)

            except Exception as e:
                if progressbar:
                    print(
                        f"\nWarning: Failed to process tile ({tile_lat:.2f}, {tile_lon:.2f}): {e}"
                    )
                continue

        if not results:
            print(
                "Warning: No embeddings were extracted. Check that points fall within available tiles."
            )
            return pd.DataFrame()

        # Create DataFrame with consistent column order
        results_df = pd.DataFrame(results)

        # Reorder columns: metadata first, then embeddings
        emb_cols = [f"emb_{i}" for i in range(128)]
        metadata_cols = [col for col in results_df.columns if col not in emb_cols]
        results_df = results_df[metadata_cols + emb_cols]

        if progressbar:
            print(
                f"Successfully extracted embeddings for {len(results_df)}/{len(points_df)} points"
            )

        return results_df

    def get_tile_bounds(
        self, lat: float, lon: float
    ) -> Tuple[float, float, float, float]:
        """Get the geographic bounds (west, south, east, north) of a tile in EPSG:4326.

        Args:
            lat: Tile latitude (lower-left corner)
            lon: Tile longitude (lower-left corner)

        Returns:
            Tuple of (west, south, east, north) bounds
        """
        return (lon - 0.05, lat - 0.05, lon + 0.05, lat + 0.05)

    def get_tile_crs(self, lat: float, lon: float) -> str:
        """Get the CRS of a specific tile.

        Args:
            lat: Tile latitude
            lon: Tile longitude

        Returns:
            CRS string (e.g., 'EPSG:32630')
        """
        import rasterio

        landmask_path = self._fetch_landmask(lat, lon, progressbar=False)
        with rasterio.open(landmask_path) as src:
            return str(src.crs)

    def get_tile_transform(self, lat: float, lon: float):
        """Get the rasterio transform for georeferencing a tile.

        Args:
            lat: Tile latitude
            lon: Tile longitude

        Returns:
            rasterio Affine transform
        """
        import rasterio

        landmask_path = self._fetch_landmask(lat, lon, progressbar=False)
        with rasterio.open(landmask_path) as src:
            return src.transform
