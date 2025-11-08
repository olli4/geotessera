GeoTessera CLI Tests
=====================

These are tests for the `geotessera` command-line interface.

Setup
-----

Set environment variable to disable fancy terminal output (ANSI codes, boxes, colors):

  $ export TERM=dumb

Create a temporary directory for test outputs and cache:

  $ export TESTDIR="$CRAMTMP/test_outputs"
  $ mkdir -p "$TESTDIR"

Override XDG cache directory to use temporary location (for test isolation):

  $ export XDG_CACHE_HOME="$CRAMTMP/cache"
  $ mkdir -p "$XDG_CACHE_HOME"

Test: Version Command
---------------------

The version command should print the version number.

  $ uv run -m geotessera.cli version
  0.7.0

Test: Info Command (Library Info)
----------------------------------

Test the info command without arguments to see library information.
We just verify key information is present, ignoring formatting:

  $ uv run -m geotessera.cli info --dataset-version v1
  Downloading registry from https://dl2.geotessera.org/v1/registry.parquet
  Registry downloaded successfully
  Loaded GeoParquet with 1,158,150 tiles
  Downloading landmasks registry from https://dl2.geotessera.org/v1/landmasks.parquet
  Landmasks registry downloaded successfully
   Version:         0.7.0                                          
   Available years: 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024 
     2017 tiles:    14,229                                         
     2018 tiles:    15,958                                         
     2019 tiles:    15,999                                         
     2020 tiles:    17,420                                         
     2021 tiles:    15,684                                         
     2022 tiles:    15,404                                         
     2023 tiles:    15,997                                         
     2024 tiles:    1,047,459                                      
   Total landmasks: 1,593,479                                      

Test: Download Dry Run for UK Tile
-----------------------------------

Test downloading a single tile covering London, UK using --dry-run to avoid actual downloads.
Verify key information is present:

  $ uv run -m geotessera.cli download \
  >   --bbox "-0.1,51.4,0.1,51.6" \
  >   --year 2024 \
  >   --format tiff \
  >   --dry-run \
  >   --dataset-version v1 2>&1 | grep -E '(Format|Year|Compression|Dataset version|Found|Files to download|Total download|Tiles in region)' | sed 's/ *$//'
   Format:          TIFF
   Year:            2024
   Compression:     lzw
   Dataset version: v1
  Found 4 tiles for region in year 2024
   Files to download:   4
   Total download size: 1.6 GB
   Tiles in region:     4
   Year:                2024
   Format:              TIFF

Test: Download Single UK Tile (TIFF format)
--------------------------------------------

Download a single tile in TIFF format to a temporary directory:

  $ uv run -m geotessera.cli download \
  >   --bbox "-0.1,51.4,0.1,51.6" \
  >   --year 2024 \
  >   --format tiff \
  >   --output "$TESTDIR/uk_tiles_tiff" \
  >   --dataset-version v1 2>&1 | grep -E 'SUCCESS' | sed 's/ *$//'
  SUCCESS: Exported 4 GeoTIFF files

Verify TIFF files were created in the registry structure:

  $ [ -n "$(find "$TESTDIR/uk_tiles_tiff/global_0.1_degree_representation/2024" -name "*.tif*" 2>/dev/null)" ] && echo "TIFF files created"
  TIFF files created

  $ find "$TESTDIR/uk_tiles_tiff/global_0.1_degree_representation/2024" -name "*.tif*" | wc -l | tr -d ' '
  4

Test: Download Single UK Tile (NPY format)
-------------------------------------------

Download the same tile in NPY format (quantized arrays with scales):

  $ uv run -m geotessera.cli download \
  >   --bbox "-0.1,51.4,0.1,51.6" \
  >   --year 2024 \
  >   --format npy \
  >   --output "$TESTDIR/uk_tiles_npy" \
  >   --dataset-version v1 2>&1 | grep -E 'SUCCESS' | sed 's/ *$//'
  SUCCESS: Downloaded 4 tiles (12 files, 424.0 MB)

Verify NPY directory structure was created:

  $ test -d "$TESTDIR/uk_tiles_npy/global_0.1_degree_representation/2024" && echo "Embeddings directory created"
  Embeddings directory created

  $ test -d "$TESTDIR/uk_tiles_npy/global_0.1_degree_tiff_all" && echo "Landmasks directory created"
  Landmasks directory created

Verify NPY files exist in grid subdirectories:

  $ [ -n "$(find "$TESTDIR/uk_tiles_npy/global_0.1_degree_representation/2024" -name "grid_*.npy" ! -name "*_scales.npy" 2>/dev/null)" ] && echo "Embedding NPY files created"
  Embedding NPY files created

  $ find "$TESTDIR/uk_tiles_npy/global_0.1_degree_representation/2024" -name "*.npy" | wc -l | tr -d ' '
  8

  $ [ -n "$(find "$TESTDIR/uk_tiles_npy/global_0.1_degree_representation/2024" -name "*_scales.npy" 2>/dev/null)" ] && echo "Scales NPY files created"
  Scales NPY files created

  $ [ -n "$(find "$TESTDIR/uk_tiles_npy/global_0.1_degree_tiff_all" -name "*.tif*" 2>/dev/null)" ] && echo "Landmask TIFF files created"
  Landmask TIFF files created

Test: Info Command on Downloaded TIFF Tiles
--------------------------------------------

Test the info command on the downloaded TIFF tiles:

  $ uv run -m geotessera.cli info --tiles "$TESTDIR/uk_tiles_tiff"
   Total tiles: 4                      
   Format:      GEOTIFF                
   Years:       2024                   
   CRS:         EPSG:32630, EPSG:32631 
   Longitude: -0.100000 to 0.100000  
   Latitude:  51.400000 to 51.600000 
   Band Count Files 
   128 bands      4 

  $ uv run -m geotessera.cli info --tiles "$TESTDIR/uk_tiles_tiff"
   Total tiles: 4                      
   Format:      GEOTIFF                
   Years:       2024                   
   CRS:         EPSG:32630, EPSG:32631 
   Longitude: -0.100000 to 0.100000  
   Latitude:  51.400000 to 51.600000 
   Band Count Files 
   128 bands      4 

Test: Info Command on Downloaded NPY Tiles
-------------------------------------------

Test the info command on the downloaded NPY tiles:

  $ uv run -m geotessera.cli info --tiles "$TESTDIR/uk_tiles_npy"
   Total tiles: 4                      
   Format:      NPY                    
   Years:       2024                   
   CRS:         EPSG:32630, EPSG:32631 
   Longitude: -0.100000 to 0.100000  
   Latitude:  51.400000 to 51.600000 
   Band Count Files 
   128 bands      4 

  $ uv run -m geotessera.cli info --tiles "$TESTDIR/uk_tiles_npy"
   Total tiles: 4                      
   Format:      NPY                    
   Years:       2024                   
   CRS:         EPSG:32630, EPSG:32631 
   Longitude: -0.100000 to 0.100000  
   Latitude:  51.400000 to 51.600000 
   Band Count Files 
   128 bands      4 

Test: Resume Capability for NPY Downloads
------------------------------------------

Test that re-running the NPY download skips existing files:

  $ uv run -m geotessera.cli download \
  >   --bbox "-0.1,51.4,0.1,51.6" \
  >   --year 2024 \
  >   --format npy \
  >   --output "$TESTDIR/uk_tiles_npy" \
  >   --dataset-version v1 2>&1 | grep -E '(Skipped|existing files)'
     Skipped 12 existing files (resume capability)

