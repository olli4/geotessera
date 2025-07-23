## v0.2.0

### Breaking Changes

- **API**: `get_embedding()` method renamed to `fetch_embedding()` for clarity
- **Registry**: Switched from year-based to block-based (5x5 degree) registry system
- **Package**: Individual year registry files (`registry_2017.txt` through `registry_2024.txt`)
  removed as they are now tracked in https://github.com/ucam-eo/tessera-manifests

### New Features

- **Tessera utilities**:
  - `find_tiles_for_geometry()` - Find tiles intersecting with regions of interest
  - `extract_points()` - multi-point embedding extraction
  - Georeferencing utilities: `get_tile_bounds()`, `get_tile_crs()`, `get_tile_transform()`

- **New modules**:
  - `io.py` - Flexible I/O supporting JSON, CSV, GeoJSON, Shapefile, and Parquet formats
  - `spatial.py` - Spatial utilities for bounding boxes, grids, and raster stitching
  - `parallel.py` - Parallel processing for efficient tile operations
  - `export.py` - Export utilities for georeferenced GeoTIFFs

- **Registry improvements**:
  - Block-based registry system (5x5 degree blocks) for faster startup
  - Support for local registry via `TESSERA_REGISTRY_DIR` environment variable
  - Auto-cloning of tessera-manifests repository when no local registry specified
  - SHA256 checksum verification
  - New `geotessera-registry` CLI tool for registry management

### API Additions

- **GeoTessera constructor now autoclones manifests**:
  - `registry_dir` - Optional local registry directory path
  - `auto_update` - Auto-update tessera-manifests repository
  - `manifests_repo_url` - Custom manifests repository URL

- **New methods**:
  - `get_available_years()` - List available years in the dataset
  - Multiple georeferencing helper methods

### CLI Enhancements

The `geotessera` tool has also been improved.

- **New arguments**:
  - `--registry-dir` - Specify local registry directory
  - `--auto-update` - Auto-update tessera-manifests repository
  - `--manifests-repo-url` - Custom manifests repository URL

- **Command improvements**:
  - `info` command shows detailed registry and year information
  - `map` command displays year distribution
  - Better progress reporting and error messages

### Infrastructure

- Added `TESSERA_DATA_DIR` environment variable to override cache location
- Lazy loading of registry blocks for improved performance

### Dependencies

- Added `rich` for enhanced CLI output and progress bars
- Updated package metadata with license information and PyPI classifiers

### Bug Fixes

- Fixed tile alignment issues
- Improved landmask and TIFF file handling
- Better error handling and user feedback via exceptions
- Fixed coverage map generation
- Resolved coordinate formatting issues

## v0.1.0

Initial release to GitHub
