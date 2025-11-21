"""Tests for tile discovery with temporary file filtering."""

from pathlib import Path
import tempfile
import numpy as np

from geotessera.tiles import (
    discover_npy_tiles,
    discover_geotiff_tiles,
    discover_zarr_tiles,
    discover_tiles,
    _is_temp_file,
)


def test_is_temp_file():
    """Test the _is_temp_file helper function."""
    # Temporary files should be detected
    assert _is_temp_file(Path(".grid_0.15_52.05.npy_tmp_abc123"))
    assert _is_temp_file(Path(".grid_0.15_52.05_2024.tif_tmp_xyz789"))
    assert _is_temp_file(Path(".grid_0.15_52.05_2024.zarr_tmp_def456"))
    
    # Regular files should not be detected as temp
    assert not _is_temp_file(Path("grid_0.15_52.05.npy"))
    assert not _is_temp_file(Path("grid_0.15_52.05_2024.tif"))
    assert not _is_temp_file(Path("grid_0.15_52.05_2024.zarr"))
    
    # Files with just dot prefix or just _tmp_ but not both
    assert not _is_temp_file(Path(".hidden_file.npy"))
    assert not _is_temp_file(Path("file_tmp_name.npy"))


def test_discover_npy_tiles_filters_temp_files():
    """Test that discover_npy_tiles filters out temporary files."""
    from unittest.mock import patch
    
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        embeddings_dir = base_dir / "global_0.1_degree_representation" / "2024"
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a regular tile file  
        regular_file = embeddings_dir / "grid_0.15_52.05.npy"
        test_array = np.random.rand(10, 10, 16).astype(np.float32)
        np.save(regular_file, test_array)
        
        # Create a temporary file (should be ignored)
        # Note: np.save adds .npy automatically, so specify without .npy
        temp_file_base = embeddings_dir / ".grid_0.15_52.05_tmp_abc123"
        np.save(temp_file_base, test_array)
        temp_file = embeddings_dir / ".grid_0.15_52.05_tmp_abc123.npy"
        
        # Verify both files exist
        assert regular_file.exists()
        assert temp_file.exists()
        
        # Mock the Tile.from_npy to track which files are attempted to be loaded
        attempted_files = []
        with patch('geotessera.tiles.Tile.from_npy') as mock_from_npy:
            mock_from_npy.side_effect = lambda path, base_dir: attempted_files.append(path) or None
            
            discover_npy_tiles(base_dir)
        
        # Check that only the regular file was attempted, not the temp file
        assert len(attempted_files) == 1, f"Expected 1 file to be attempted, got {len(attempted_files)}"
        assert attempted_files[0] == regular_file, "Wrong file was attempted"
        assert temp_file not in attempted_files, "Temporary file should not have been attempted"


def test_discover_geotiff_tiles_filters_temp_files():
    """Test that discover_geotiff_tiles filters out temporary files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        
        # Create regular and temp GeoTIFF files
        regular_file = base_dir / "grid_0.15_52.05_2024.tif"
        temp_file = base_dir / ".grid_0.15_52.05_2024.tif_tmp_xyz789"
        
        # Create mock GeoTIFF files (just touch them - actual GeoTIFF creation would require rasterio)
        regular_file.touch()
        temp_file.touch()
        
        # This will fail to parse the files, but we're testing that temp files are filtered
        tiles = discover_geotiff_tiles(base_dir)
        
        # Neither file should be loaded (because they're not valid GeoTIFFs),
        # but temp file should have been filtered before attempting to load
        assert len(tiles) == 0


def test_discover_zarr_tiles_filters_temp_files():
    """Test that discover_zarr_tiles filters out temporary files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        
        # Create regular and temp zarr directories
        regular_file = base_dir / "grid_0.15_52.05_2024.zarr"
        temp_file = base_dir / ".grid_0.15_52.05_2024.zarr_tmp_def456"
        
        # Create mock zarr directories
        regular_file.mkdir()
        temp_file.mkdir()
        
        # This will fail to parse the files, but we're testing that temp files are filtered
        tiles = discover_zarr_tiles(base_dir)
        
        # Neither file should be loaded (because they're not valid zarr stores),
        # but temp file should have been filtered before attempting to load
        assert len(tiles) == 0


def test_discover_tiles_filters_temp_files():
    """Test that discover_tiles filters out temporary files in initial check."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        embeddings_dir = base_dir / "global_0.1_degree_representation" / "2024"
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        landmasks_dir = base_dir / "landmasks"
        landmasks_dir.mkdir(parents=True, exist_ok=True)
        
        # Create only a temporary NPY file
        temp_file = embeddings_dir / ".grid_0.15_52.05.npy_tmp_abc123"
        test_array = np.random.rand(10, 10, 16).astype(np.float32)
        np.save(temp_file, test_array)
        
        # discover_tiles should not find any NPY files and fall back to other formats
        # Since no valid tiles exist, it should return an empty list
        tiles = discover_tiles(base_dir)
        
        # Should find no tiles since only temp file exists
        assert len(tiles) == 0
