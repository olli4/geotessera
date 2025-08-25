"""Core GeoTessera functionality.

The library focusses on:
1. Downloading Tessera tiles for lat/lon bounding boxes to numpy arrays
2. Exporting tiles to individual GeoTIFF files with accurate metadata
"""

from pathlib import Path
from typing import Union, List, Tuple, Optional
import numpy as np

from .registry import Registry

try:
    import importlib.metadata

    __version__ = importlib.metadata.version("geotessera")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"


class GeoTessera:
    """Library for downloading Tessera tiles and exporting GeoTIFFs.

    Core functionality:
    - Download tiles within a bounding box to numpy arrays
    - Export individual tiles as GeoTIFF files with correct metadata
    - Manage registry and data access
    """

    def __init__(
        self,
        dataset_version: str = "v1",
        cache_dir: Optional[Union[str, Path]] = None,
        registry_dir: Optional[Union[str, Path]] = None,
        auto_update: bool = True,
        manifests_repo_url: str = "https://github.com/ucam-eo/tessera-manifests.git",
    ):
        """Initialize GeoTessera with registry management.

        Args:
            dataset_version: Tessera dataset version (e.g., 'v1', 'v2')
            cache_dir: Directory for caching downloaded files
            registry_dir: Directory containing registry files
            auto_update: Whether to auto-update registry
            manifests_repo_url: Git repository URL for registry manifests
        """
        self.dataset_version = dataset_version
        self.registry = Registry(
            version=dataset_version,
            cache_dir=cache_dir,
            registry_dir=registry_dir,
            auto_update=auto_update,
            manifests_repo_url=manifests_repo_url,
        )

    @property
    def version(self) -> str:
        """Get the GeoTessera library version."""
        return __version__

    def fetch_embeddings(
        self,
        bbox: Tuple[float, float, float, float],
        year: int = 2024,
        progress_callback: Optional[callable] = None,
    ) -> List[Tuple[float, float, np.ndarray, object, object]]:
        """Fetch all embedding tiles within a bounding box with CRS information.

        Args:
            bbox: Bounding box as (min_lon, min_lat, max_lon, max_lat)
            year: Year of embeddings to download
            progress_callback: Optional callback function(current, total) for progress tracking

        Returns:
            List of (tile_lon, tile_lat, embedding_array, crs, transform) tuples where:
            - tile_lon: Tile center longitude
            - tile_lat: Tile center latitude
            - embedding_array: shape (H, W, 128) with dequantized values
            - crs: CRS object from rasterio (coordinate reference system)
            - transform: Affine transform from rasterio
        """
        # Load registry blocks for this region and get available tiles directly
        tiles_to_download = self.registry.load_blocks_for_region(bbox, year)

        # Download each tile with progress tracking
        results = []
        total_tiles = len(tiles_to_download)

        for i, (tile_lon, tile_lat) in enumerate(tiles_to_download):
            try:
                # Create a sub-progress callback for this tile's downloads
                def tile_progress_callback(
                    current: int, total: int, status: str = None
                ):
                    if progress_callback:
                        # Map individual file progress to overall tile progress
                        tile_progress = (
                            i * 100 + (current / max(total, 1)) * 100
                        ) / total_tiles
                        tile_status = (
                            f"Tile {i + 1}/{total_tiles}: {status}"
                            if status
                            else f"Fetching tile {i + 1}/{total_tiles}"
                        )
                        progress_callback(int(tile_progress), 100, tile_status)

                embedding, crs, transform = self.fetch_embedding(
                    tile_lon, tile_lat, year, tile_progress_callback
                )
                results.append((tile_lon, tile_lat, embedding, crs, transform))

                # Update progress for completed tile
                if progress_callback:
                    progress_callback(
                        (i + 1) * 100 // total_tiles,
                        100,
                        f"Completed tile {i + 1}/{total_tiles}",
                    )

            except Exception as e:
                print(
                    f"Warning: Failed to download tile ({tile_lat:.2f}, {tile_lon:.2f}): {e}"
                )
                if progress_callback:
                    progress_callback(
                        (i + 1) * 100 // total_tiles,
                        100,
                        f"Failed tile {i + 1}/{total_tiles}",
                    )
                continue

        return results

    def fetch_embedding(
        self,
        lon: float,
        lat: float,
        year: int,
        progress_callback: Optional[callable] = None,
    ) -> Tuple[np.ndarray, object, object]:
        """Fetch and dequantize a single embedding tile with CRS information.

        Args:
            lon: Tile center longitude
            lat: Tile center latitude
            year: Year of embeddings
            progress_callback: Optional callback for download progress

        Returns:
            Tuple of (dequantized_embedding, crs, transform) where:
            - dequantized_embedding: array of shape (H, W, 128)
            - crs: CRS object from rasterio (coordinate reference system)
            - transform: Affine transform from rasterio
        """
        from .registry import tile_to_embedding_paths

        # Ensure the block is loaded
        self.registry.ensure_block_loaded(year, lon, lat)

        # Get file paths
        embedding_path, scales_path = tile_to_embedding_paths(lon, lat, year)

        # Fetch the files
        embedding_file = self.registry.fetch(
            embedding_path, progressbar=False, progress_callback=progress_callback
        )
        scales_file = self.registry.fetch(
            scales_path, progressbar=False, progress_callback=progress_callback
        )

        # Load and dequantize
        quantized_embedding = np.load(embedding_file)
        scales = np.load(scales_file)

        # Dequantize using scales
        # Handle both 2D scales (H, W) and 3D scales (H, W, 128)
        if scales.ndim == 2 and quantized_embedding.ndim == 3:
            # Broadcast 2D scales to match 3D embedding shape
            scales = scales[..., np.newaxis]  # Add channel dimension

        dequantized = quantized_embedding.astype(np.float32) * scales

        # Get CRS and transform from landmask
        crs, transform = self._get_utm_projection_from_landmask(lon, lat)

        return dequantized, crs, transform

    def _get_utm_projection_from_landmask(self, lon: float, lat: float):
        """Get UTM projection info from corresponding landmask tile.

        Args:
            lon: Tile center longitude
            lat: Tile center latitude

        Returns:
            Tuple of (crs, transform) from landmask tile

        Raises:
            ImportError: If rasterio is not available
            RuntimeError: If landmask tile cannot be fetched or read
        """
        try:
            import rasterio
        except ImportError:
            raise ImportError(
                "rasterio required for UTM projection retrieval: pip install rasterio"
            )

        try:
            from .registry import tile_to_landmask_filename

            # Get landmask filename
            landmask_filename = tile_to_landmask_filename(lon, lat)

            # Ensure registry block is loaded
            self.registry.ensure_tile_block_loaded(lon, lat)

            # Fetch landmask file
            landmask_path = self.registry.fetch_landmask(
                landmask_filename, progressbar=False
            )

            # Extract CRS and transform
            with rasterio.open(landmask_path) as src:
                if src.crs is None:
                    raise RuntimeError(
                        f"Landmask tile {landmask_filename} has no CRS information"
                    )
                if src.transform is None:
                    raise RuntimeError(
                        f"Landmask tile {landmask_filename} has no transform information"
                    )
                return src.crs, src.transform

        except Exception as e:
            if isinstance(e, (ImportError, RuntimeError)):
                raise
            raise RuntimeError(
                f"Failed to get UTM projection from landmask for ({lon:.2f}, {lat:.2f}): {e}"
            ) from e

    def export_embedding_geotiff(
        self,
        lon: float,
        lat: float,
        output_path: Union[str, Path],
        year: int = 2024,
        bands: Optional[List[int]] = None,
        compress: str = "lzw",
    ) -> str:
        """Export a single embedding tile as a GeoTIFF file with native UTM projection.

        Args:
            lon: Tile center longitude
            lat: Tile center latitude
            output_path: Output path for GeoTIFF file
            year: Year of embeddings to export
            bands: List of band indices to export (None = all 128 bands)
            compress: Compression method for GeoTIFF

        Returns:
            Path to created GeoTIFF file

        Raises:
            ImportError: If rasterio is not available
            RuntimeError: If landmask tile or embedding data cannot be fetched
            FileNotFoundError: If registry files are missing
        """
        try:
            import rasterio
        except ImportError:
            raise ImportError(
                "rasterio required for GeoTIFF export: pip install rasterio"
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Fetch single tile with CRS info
        embedding, crs, transform = self.fetch_embedding(lon, lat, year)

        # Select bands
        if bands is not None:
            data = embedding[:, :, bands].copy()
            band_count = len(bands)
        else:
            data = embedding.copy()
            band_count = 128

        # Get dimensions for GeoTIFF
        height, width = data.shape[:2]

        # Write GeoTIFF
        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=band_count,
            dtype="float32",
            crs=crs,
            transform=transform,
            compress=compress,
            tiled=True,
            blockxsize=256,
            blockysize=256,
        ) as dst:
            # Write bands
            for i in range(band_count):
                dst.write(data[:, :, i], i + 1)

            # Add band descriptions
            if bands is not None:
                for i, band_idx in enumerate(bands):
                    dst.set_band_description(i + 1, f"Tessera_Band_{band_idx}")
            else:
                for i in range(128):
                    dst.set_band_description(i + 1, f"Tessera_Band_{i}")

            # Add metadata
            dst.update_tags(
                TESSERA_DATASET_VERSION=self.dataset_version,
                TESSERA_YEAR=str(year),
                TESSERA_TILE_LAT=f"{lat:.2f}",
                TESSERA_TILE_LON=f"{lon:.2f}",
                TESSERA_DESCRIPTION="GeoTessera satellite embedding tile",
                GEOTESSERA_VERSION=__version__,
            )

        return str(output_path)

    def export_embedding_geotiffs(
        self,
        bbox: Tuple[float, float, float, float],
        output_dir: Union[str, Path],
        year: int = 2024,
        bands: Optional[List[int]] = None,
        compress: str = "lzw",
        progress_callback: Optional[callable] = None,
    ) -> List[str]:
        """Export all embedding tiles in bounding box as individual GeoTIFF files with native UTM projections.

        Args:
            bbox: Bounding box as (min_lon, min_lat, max_lon, max_lat)
            output_dir: Directory to save GeoTIFF files
            year: Year of embeddings to export
            bands: List of band indices to export (None = all 128 bands)
            compress: Compression method for GeoTIFF
            progress_callback: Optional callback function(current, total) for progress tracking

        Returns:
            List of paths to created GeoTIFF files

        Raises:
            ImportError: If rasterio is not available
            RuntimeError: If landmask tiles or embedding data cannot be fetched
            FileNotFoundError: If registry files are missing
        """
        try:
            import rasterio
        except ImportError:
            raise ImportError(
                "rasterio required for GeoTIFF export: pip install rasterio"
            )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create a wrapper callback to handle two-phase progress
        def fetch_progress_callback(current: int, total: int, status: str = None):
            if progress_callback:
                # Phase 1: Fetching tiles (0-50% of total progress)
                overall_progress = int((current / total) * 50)
                display_status = status or f"Fetching tile {current}/{total}"
                progress_callback(overall_progress, 100, display_status)

        # Fetch tiles with progress tracking
        if progress_callback:
            progress_callback(0, 100, "Loading registry blocks...")

        tiles = self.fetch_embeddings(bbox, year, fetch_progress_callback)

        if not tiles:
            print("No tiles found in bounding box")
            return []

        if progress_callback:
            progress_callback(
                50, 100, f"Fetched {len(tiles)} tiles, starting GeoTIFF export..."
            )

        created_files = []
        total_tiles = len(tiles)

        for i, (tile_lon, tile_lat, embedding, crs, transform) in enumerate(tiles):
            # Create filename first for progress reporting
            filename = f"tessera_{year}_lat{tile_lat:.2f}_lon{tile_lon:.2f}.tif"
            output_path = output_dir / filename

            # Update progress to show we're starting this file
            if progress_callback:
                export_progress = int(50 + (i / total_tiles) * 50)
                progress_callback(export_progress, 100, f"Creating {filename}...")

            # Select bands
            if bands is not None:
                data = embedding[:, :, bands].copy()
                band_count = len(bands)
            else:
                data = embedding.copy()
                band_count = 128

            # Get dimensions for GeoTIFF
            height, width = data.shape[:2]

            # Write GeoTIFF
            with rasterio.open(
                output_path,
                "w",
                driver="GTiff",
                height=height,
                width=width,
                count=band_count,
                dtype="float32",
                crs=crs,
                transform=transform,
                compress=compress,
                tiled=True,
                blockxsize=256,
                blockysize=256,
            ) as dst:
                # Write bands
                for i in range(band_count):
                    dst.write(data[:, :, i], i + 1)

                # Add band descriptions
                if bands is not None:
                    for i, band_idx in enumerate(bands):
                        dst.set_band_description(i + 1, f"Tessera_Band_{band_idx}")
                else:
                    for i in range(128):
                        dst.set_band_description(i + 1, f"Tessera_Band_{i}")

                # Add metadata
                dst.update_tags(
                    TESSERA_DATASET_VERSION=self.dataset_version,
                    TESSERA_YEAR=str(year),
                    TESSERA_TILE_LAT=f"{tile_lat:.2f}",
                    TESSERA_TILE_LON=f"{tile_lon:.2f}",
                    TESSERA_DESCRIPTION="GeoTessera satellite embedding tile",
                    GEOTESSERA_VERSION=__version__,
                )

            created_files.append(str(output_path))

            # Update progress for GeoTIFF export phase
            if progress_callback:
                # Phase 2: Exporting GeoTIFFs (50-100% of total progress)
                export_progress = int(50 + ((i + 1) / total_tiles) * 50)
                filename = f"tessera_{year}_lat{tile_lat:.2f}_lon{tile_lon:.2f}.tif"
                progress_callback(
                    export_progress, 100, f"Exported {filename} ({i + 1}/{total_tiles})"
                )

        if progress_callback:
            progress_callback(
                100, 100, f"Completed! Exported {len(created_files)} GeoTIFF files"
            )

        print(f"Exported {len(created_files)} GeoTIFF files to {output_dir}")
        return created_files

    def merge_geotiffs_to_mosaic(
        self,
        geotiff_paths: List[str],
        output_path: Union[str, Path],
        target_crs: str = "EPSG:3857",
        compress: str = "lzw",
    ) -> str:
        """Merge a list of GeoTIFF files into a single mosaic in the target CRS.

        Args:
            geotiff_paths: List of paths to GeoTIFF files to merge
            output_path: Path for output mosaic GeoTIFF
            target_crs: Target CRS for the mosaic (default: Web Mercator EPSG:3857)
            compress: Compression method for output GeoTIFF

        Returns:
            Path to created mosaic file

        Raises:
            ImportError: If rasterio is not available
            RuntimeError: If merge fails
        """
        try:
            import rasterio
            from rasterio.merge import merge
            from rasterio.warp import calculate_default_transform
            import tempfile
            import os
        except ImportError:
            raise ImportError(
                "rasterio required for mosaic creation: pip install rasterio"
            )

        if not geotiff_paths:
            raise RuntimeError("No GeoTIFF files provided")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Determine source resolution from first file
        with rasterio.open(geotiff_paths[0]) as first_src:
            source_resolution = min(abs(first_src.transform.a), abs(first_src.transform.e))

        # Create temporary directory for reprojected files
        with tempfile.TemporaryDirectory(prefix="geotessera_reproject_") as temp_dir:
            # Reproject all files to target CRS
            reprojected_files = []
            for i, geotiff_file in enumerate(geotiff_paths):
                reprojected_file = os.path.join(temp_dir, f"reprojected_{i}.tif")
                
                with rasterio.open(geotiff_file) as src:
                    # Calculate transform and dimensions for target CRS
                    transform, width, height = calculate_default_transform(
                        src.crs, target_crs, src.width, src.height, *src.bounds,
                        resolution=source_resolution
                    )
                    
                    # Create reprojected file
                    with rasterio.open(
                        reprojected_file,
                        'w',
                        driver='GTiff',
                        height=height,
                        width=width,
                        count=src.count,
                        dtype=src.dtypes[0],  # Get dtype from first band
                        crs=target_crs,
                        transform=transform,
                        compress=compress,
                        tiled=True,
                        blockxsize=256,
                        blockysize=256,
                    ) as dst:
                        from rasterio.warp import reproject, Resampling
                        
                        for band_idx in range(1, src.count + 1):
                            reproject(
                                source=rasterio.band(src, band_idx),
                                destination=rasterio.band(dst, band_idx),
                                src_transform=src.transform,
                                src_crs=src.crs,
                                dst_transform=transform,
                                dst_crs=target_crs,
                                resampling=Resampling.bilinear
                            )
                        
                        # Copy metadata and band descriptions
                        dst.update_tags(**src.tags())
                        for band_idx in range(1, src.count + 1):
                            band_desc = src.descriptions[band_idx-1] if src.descriptions and band_idx <= len(src.descriptions) else None
                            if band_desc:
                                dst.set_band_description(band_idx, band_desc)

                reprojected_files.append(reprojected_file)

            # Open all reprojected files for merging
            src_files = [rasterio.open(f) for f in reprojected_files]
            
            try:
                # Merge tiles
                mosaic_array, mosaic_transform = merge(src_files, method='first')
                
                # Get metadata from first file
                first_src = src_files[0]
                profile = first_src.profile.copy()
                profile.update({
                    'height': mosaic_array.shape[1],
                    'width': mosaic_array.shape[2],
                    'transform': mosaic_transform,
                    'dtype': mosaic_array.dtype,  # Use mosaic array dtype
                    'compress': compress,
                    'tiled': True,
                    'blockxsize': 512,
                    'blockysize': 512,
                })
                
                # Write mosaic
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(mosaic_array)
                    
                    # Copy band descriptions from first file
                    for band_idx in range(1, mosaic_array.shape[0] + 1):
                        band_desc = first_src.descriptions[band_idx-1] if first_src.descriptions and band_idx <= len(first_src.descriptions) else None
                        if band_desc:
                            dst.set_band_description(band_idx, band_desc)
                    
                    # Update metadata
                    dst.update_tags(
                        TESSERA_TARGET_CRS=target_crs,
                        TESSERA_RESOLUTION=str(source_resolution),
                        TESSERA_TILE_COUNT=str(len(geotiff_paths)),
                        TESSERA_DESCRIPTION="GeoTessera satellite embedding mosaic",
                        GEOTESSERA_VERSION=__version__,
                    )

            finally:
                # Close all source files
                for src in src_files:
                    src.close()

        return str(output_path)
