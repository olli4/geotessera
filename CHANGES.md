## v0.5.0 (unreleased)

### Breaking API Changes

- **Modified return values**: `fetch_embedding()` and `fetch_embeddings()` now return CRS and transform information
  - `fetch_embedding()` returns `(embedding, crs, transform)` instead of just `embedding`
  - `fetch_embeddings()` returns list of `(lat, lon, embedding, crs, transform)` tuples instead of `(lat, lon, embedding)`
  - This provides direct access to the coordinate reference system from landmask tiles
  - Useful for applications that need projection information without exporting to GeoTIFF

## v0.4.0

### Enhanced Full-Band GeoTIFF Support

- **Simplified GeoTIFF export**: Always uses float32 precision without normalization
  - **Removed normalization logic**: All outputs preserve dequantized embedding values exactly
  - **Consistent data type**: Always float32 to maintain precision regardless of band count
  - **Band selection**: Still supports selecting specific bands (e.g., `--bands 0 1 2`) while preserving raw values
  - **Backward compatible**: Existing scripts continue to work unchanged
- **Enhanced CLI**: `geotessera visualize` now defaults to full 128-band export when `--bands` is not specified
  - Default: `geotessera visualize --region area.json --output full.tif` (128 bands, float32)
  - Selected bands: `geotessera visualize --region area.json --bands 0 1 2 --output subset.tif` (3 bands, float32)

### CLI Improvements and Bug Fixes

- **Fixed `visualize` command**: Resolved "Unknown geometry type: 'featurecollection'" error
  - Fixed condition order bug in `find_tiles_for_geometry()` that incorrectly handled GeoDataFrames
  - Command now works reliably with GeoJSON, Shapefile, GeoPackage, and other region file formats
- **Improved performance**: Made `find_tiles_for_geometry()` efficient by loading only needed registry blocks
  - Previously loaded entire 400+ block registry, now loads only 1-4 blocks for typical regions
  - Faster startup and reduced memory usage for both `visualize` and `serve` commands
- **Enhanced tile generation**: Fixed `serve` command's gdal2tiles compatibility
  - Automatically converts float32 TIFF to 8-bit using `gdal_translate -scale` before tile generation
- **Better logging**: Improved registry loading messages
  - Clear distinction between newly loaded vs. already cached registry blocks
  - More informative progress reporting during region processing
- **Code rationalization**: Created shared logic between `visualize` and `serve` commands
  - Added `merge_embeddings_for_region_file()` method to core library for region file handling
  - Eliminated code duplication while maintaining full functionality

### Infrastructure Improvements

- **Natural Earth integration**: Set proper user agent when downloading world map data
- **Cleanup**: Removed accidentally committed world map files to reduce repository size

## v0.3.0

- Moved the map updating CI to https://github.com/ucam-eo/tessera-coverage.
  This results in a reset main branch with a cleaner git history.
- Modified `export_single_tile_to_tiff` so it can take not just 3 bands,
  allowing exporting of all 128 bands to a TIFF (#3 @epingchris)
- Fix degrees for georeferencing (#3 @nkarasiak and @avsm)
- Improve GDAL compatibility with different versions (#3 @nkarasiak)
- Fix map coverage generation with geopandas>1.0 (#4 @avsm, reported by @epingchris)
- Remove unnecessary registry directory existence check that prevented custom TESSERA_REGISTRY_DIR usage (#5 @avsm, reported by @epingchris)

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
