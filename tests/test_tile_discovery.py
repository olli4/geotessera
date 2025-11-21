"""Tests for tile discovery with pattern-based filtering."""

from pathlib import Path
import tempfile
import numpy as np

from geotessera.tiles import (
    discover_npy_tiles,
    discover_geotiff_tiles,
    discover_zarr_tiles,
    discover_tiles,
)


def test_discover_npy_tiles_skips_invalid_patterns():
    """Test that discover_npy_tiles silently skips files that don't match the expected pattern."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        embeddings_dir = base_dir / "global_0.1_degree_representation" / "2024"
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a regular tile file  
        regular_file = embeddings_dir / "grid_0.15_52.05.npy"
        test_array = np.random.rand(10, 10, 16).astype(np.float32)
        np.save(regular_file, test_array)
        
        # Create files with invalid patterns (should be silently skipped)
        # These include temporary files and other non-tile files
        invalid_files = [
            embeddings_dir / ".grid_0.15_52.05_tmp_abc123",  # temp file
            embeddings_dir / "invalid_name",  # doesn't match pattern
            embeddings_dir / "grid_invalid",  # doesn't match pattern
        ]
        for invalid_file in invalid_files:
            np.save(invalid_file, test_array)
        
        # Count total .npy files created
        all_npy_files = list(embeddings_dir.glob("*.npy"))
        assert len(all_npy_files) == 4, f"Expected 4 .npy files, got {len(all_npy_files)}"
        
        # Run discovery (will fail to load tiles without proper metadata, but should not error on invalid names)
        tiles = discover_npy_tiles(base_dir)
        
        # Should return empty list (no valid tiles with all required files)
        # but importantly, should not raise exceptions for invalid patterns
        assert isinstance(tiles, list)


def test_discover_geotiff_tiles_skips_invalid_patterns():
    """Test that discover_geotiff_tiles silently skips files that don't match the expected pattern."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        
        # Create files with various patterns
        regular_file = base_dir / "grid_0.15_52.05_2024.tif"
        regular_file.touch()
        
        invalid_files = [
            base_dir / ".grid_0.15_52.05_2024.tif_tmp_xyz789",  # temp file
            base_dir / "invalid_name.tif",  # doesn't match pattern
            base_dir / "grid_invalid.tiff",  # doesn't match pattern
        ]
        for invalid_file in invalid_files:
            invalid_file.touch()
        
        # Count total tif/tiff files created (note: glob doesn't match files starting with .)
        all_tif_files = list(base_dir.rglob("*.tif")) + list(base_dir.rglob("*.tiff"))
        # Should find 3 visible files (the .tif_tmp file is hidden)
        assert len(all_tif_files) >= 3, f"Expected at least 3 tif/tiff files, got {len(all_tif_files)}"
        
        # Run discovery (will fail to load as GeoTIFF, but should not error on invalid patterns)
        tiles = discover_geotiff_tiles(base_dir)
        
        # Should return empty list (no valid GeoTIFFs)
        # but importantly, should not raise exceptions for invalid patterns
        assert isinstance(tiles, list)


def test_discover_zarr_tiles_skips_invalid_patterns():
    """Test that discover_zarr_tiles silently skips files that don't match the expected pattern."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        
        # Create directories with various patterns
        regular_file = base_dir / "grid_0.15_52.05_2024.zarr"
        regular_file.mkdir()
        
        invalid_files = [
            base_dir / ".grid_0.15_52.05_2024.zarr_tmp_def456",  # temp file
            base_dir / "invalid_name.zarr",  # doesn't match pattern
            base_dir / "grid_invalid.zarr",  # doesn't match pattern
        ]
        for invalid_file in invalid_files:
            invalid_file.mkdir()
        
        # Count total zarr directories created (note: glob doesn't match files starting with .)
        all_zarr_dirs = list(base_dir.rglob("*.zarr"))
        # Should find 3 visible directories (the .zarr_tmp directory is hidden)
        assert len(all_zarr_dirs) >= 3, f"Expected at least 3 zarr dirs, got {len(all_zarr_dirs)}"
        
        # Run discovery (will fail to load as zarr, but should not error on invalid patterns)
        tiles = discover_zarr_tiles(base_dir)
        
        # Should return empty list (no valid zarr stores)
        # but importantly, should not raise exceptions for invalid patterns
        assert isinstance(tiles, list)


def test_discover_tiles_filters_invalid_patterns():
    """Test that discover_tiles filters out files with invalid patterns in initial check."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        embeddings_dir = base_dir / "global_0.1_degree_representation" / "2024"
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        # Create only files with invalid patterns
        invalid_files = [
            embeddings_dir / ".grid_0.15_52.05_tmp_abc123",
            embeddings_dir / "invalid_name",
        ]
        test_array = np.random.rand(10, 10, 16).astype(np.float32)
        for invalid_file in invalid_files:
            np.save(invalid_file, test_array)
        
        # discover_tiles should not find any valid NPY files (pattern-filtered in initial check)
        # and fall back to other formats. Since no valid tiles exist, returns empty list
        tiles = discover_tiles(base_dir)
        
        # Should find no tiles
        assert len(tiles) == 0
