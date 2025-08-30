"""Simplified GeoTessera command-line interface.

Focused on downloading tiles and creating visualizations from the generated GeoTIFFs.
"""

# Will configure pooch logging after imports

import webbrowser
import threading
import time
import http.server
import socketserver
import json
import tempfile
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Optional, Callable
from typing_extensions import Annotated

import numpy as np
import typer
from rich.console import Console
from geotessera import __version__
from rich.progress import Progress, TaskID, BarColumn, TextColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

from .core import GeoTessera
from .country import get_country_bbox
from .visualization import (
    calculate_bbox_from_file,
    create_pca_mosaic,
    analyze_geotiff_coverage,
)
from .web import (
    geotiff_to_web_tiles,
    create_simple_web_viewer,
    create_coverage_summary_map,
    prepare_mosaic_for_web,
)


def is_url(string: str) -> bool:
    """Check if a string is a valid URL."""
    try:
        result = urllib.parse.urlparse(string)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def download_region_file(url: str) -> str:
    """Download a region file from a URL to a temporary location.
    
    Args:
        url: The URL to download from
        
    Returns:
        Path to the temporary downloaded file
        
    Raises:
        Exception: If download fails
    """
    try:
        # Create a temporary file with appropriate extension
        parsed_url = urllib.parse.urlparse(url)
        path = parsed_url.path
        if path.endswith('.geojson'):
            suffix = '.geojson'
        elif path.endswith('.json'):
            suffix = '.json'
        elif path.endswith('.shp'):
            suffix = '.shp'
        elif path.endswith('.gpkg'):
            suffix = '.gpkg'
        else:
            # Default to geojson for unknown extensions
            suffix = '.geojson'
            
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_path = temp_file.name
        temp_file.close()
        
        # Download the file
        urllib.request.urlretrieve(url, temp_path)
        
        return temp_path
        
    except Exception as e:
        raise Exception(f"Failed to download region file from {url}: {e}")


def format_bbox(bbox_coords) -> str:
    """Format bounding box coordinates for pretty display.

    Args:
        bbox_coords: Tuple of (min_lon, min_lat, max_lon, max_lat)

    Returns:
        Compact human-readable string representation of bbox with degree symbols
    """
    min_lon, min_lat, max_lon, max_lat = bbox_coords

    # Format longitude with E/W direction
    min_lon_str = f"{abs(min_lon):.6f}°{'W' if min_lon < 0 else 'E'}"
    max_lon_str = f"{abs(max_lon):.6f}°{'W' if max_lon < 0 else 'E'}"

    # Format latitude with N/S direction
    min_lat_str = f"{abs(min_lat):.6f}°{'S' if min_lat < 0 else 'N'}"
    max_lat_str = f"{abs(max_lat):.6f}°{'S' if max_lat < 0 else 'N'}"

    return f"[{min_lon_str}, {min_lat_str}] - [{max_lon_str}, {max_lat_str}]"


app = typer.Typer(
    name="geotessera",
    help=f"GeoTessera v{__version__}: Download satellite embedding tiles as GeoTIFFs",
    add_completion=False,
    rich_markup_mode="rich",
)

console = Console()





def create_progress_callback(progress: Progress, task_id: TaskID) -> Callable:
    """Create a progress callback for core library operations."""

    def progress_callback(current: int, total: int, status: str = None):
        if status:
            progress.update(task_id, completed=current, total=total, status=status)
        else:
            progress.update(task_id, completed=current, total=total)

    return progress_callback


def create_download_progress_callback(progress: Progress, task_id: TaskID) -> Callable:
    """Create a progress callback specifically for download operations.

    This callback handles both high-level tile progress and individual file downloads.
    """

    def progress_callback(current: int, total: int, status: str = None):
        if status:
            # Update with status message
            progress.update(task_id, completed=current, total=total, status=status)
        else:
            # Simple numeric progress update
            progress.update(task_id, completed=current, total=total)

    return progress_callback


@app.command()
def info(
    geotiffs: Annotated[
        Optional[Path],
        typer.Option("--geotiffs", help="Analyze GeoTIFF files/directory"),
    ] = None,
    dataset_version: Annotated[
        str,
        typer.Option(
            "--dataset-version", help="Tessera dataset version (e.g., v1, v2)"
        ),
    ] = "v1",
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Verbose output")
    ] = False,
):
    """Show information about GeoTIFF files or library."""

    if geotiffs:
        # Analyze GeoTIFF files
        if geotiffs.is_file():
            geotiff_paths = [str(geotiffs)]
        else:
            geotiff_paths = list(map(str, geotiffs.glob("*.tif")))
            geotiff_paths.extend(map(str, geotiffs.glob("*.tiff")))

        if not geotiff_paths:
            rprint(f"[red]No GeoTIFF files found in {geotiffs}[/red]")
            raise typer.Exit(1)

        coverage = analyze_geotiff_coverage(geotiff_paths)

        # Create analysis table
        analysis_table = Table(show_header=False, box=None)
        analysis_table.add_row("Total files:", str(coverage["total_files"]))
        analysis_table.add_row("Years:", ", ".join(coverage["years"]))
        analysis_table.add_row("CRS:", ", ".join(coverage["crs"]))

        rprint(
            Panel(
                analysis_table,
                title="[bold]📊 GeoTIFF Analysis[/bold]",
                border_style="blue",
            )
        )

        bounds = coverage["bounds"]

        bounds_table = Table(show_header=False, box=None)
        bounds_table.add_row(
            "Longitude:", f"{bounds['min_lon']:.6f} to {bounds['max_lon']:.6f}"
        )
        bounds_table.add_row(
            "Latitude:", f"{bounds['min_lat']:.6f} to {bounds['max_lat']:.6f}"
        )

        rprint(
            Panel(
                bounds_table, title="[bold]🗺️ Bounding Box[/bold]", border_style="green"
            )
        )

        bands_table = Table(show_header=True, header_style="bold blue")
        bands_table.add_column("Band Count")
        bands_table.add_column("Files", justify="right")

        for bands_count, count in coverage["band_counts"].items():
            bands_table.add_row(f"{bands_count} bands", str(count))

        rprint(
            Panel(
                bands_table,
                title="[bold]🎵 Band Information[/bold]",
                border_style="cyan",
            )
        )

        if verbose:
            tiles_table = Table(show_header=True, header_style="bold blue")
            tiles_table.add_column("Filename")
            tiles_table.add_column("Coordinates")
            tiles_table.add_column("Bands", justify="right")

            for tile in coverage["tiles"][:10]:
                tiles_table.add_row(
                    Path(tile["path"]).name,
                    f"({tile['tile_lat']}, {tile['tile_lon']})",
                    str(tile["bands"]),
                )

            rprint(
                Panel(
                    tiles_table,
                    title="[bold]📁 First 10 Tiles[/bold]",
                    border_style="yellow",
                )
            )

    else:
        # Show library info
        gt = GeoTessera(dataset_version=dataset_version)
        years = gt.registry.get_available_years()

        info_table = Table(show_header=False, box=None)
        info_table.add_row("Version:", gt.version)
        info_table.add_row("Available years:", ", ".join(map(str, years)))
        info_table.add_row(
            "Registry loaded blocks:", str(len(gt.registry.loaded_blocks))
        )

        rprint(
            Panel(
                info_table,
                title=f"[bold]🌍 GeoTessera v{__version__} Library Info[/bold]",
                border_style="blue",
            )
        )


@app.command()
def coverage(
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Output PNG file path")
    ] = Path("tessera_coverage.png"),
    year: Annotated[
        Optional[int],
        typer.Option(
            "--year",
            help="Specific year to visualize (e.g., 2024). If not specified, shows multi-year analysis.",
        ),
    ] = None,
    region_file: Annotated[
        Optional[str],
        typer.Option(
            "--region-file", help="GeoJSON/Shapefile to focus coverage map on specific region (file path or URL)"
        ),
    ] = None,
    country: Annotated[
        Optional[str],
        typer.Option(
            "--country", help="Country name to focus coverage map on (e.g., 'United Kingdom', 'UK', 'GB')"
        ),
    ] = None,
    tile_color: Annotated[
        str, typer.Option("--tile-color", help="Color for tile rectangles (when not using multi-year colors)")
    ] = "red",
    tile_alpha: Annotated[
        float, typer.Option("--tile-alpha", help="Transparency of tiles (0.0-1.0)")
    ] = 0.6,
    tile_size: Annotated[
        float,
        typer.Option(
            "--tile-size", help="Size multiplier for tiles (1.0 = actual size)"
        ),
    ] = 1.0,
    width_pixels: Annotated[
        int, typer.Option("--width", help="Output image width in pixels")
    ] = 2000,
    no_countries: Annotated[
        bool, typer.Option("--no-countries", help="Don't show country boundaries")
    ] = False,
    no_multi_year_colors: Annotated[
        bool, typer.Option("--no-multi-year-colors", help="Disable multi-year color coding")
    ] = False,
    dataset_version: Annotated[
        str, typer.Option("--dataset-version", help="Tessera dataset version")
    ] = "v1",
    cache_dir: Annotated[
        Optional[Path], typer.Option("--cache-dir", help="Cache directory")
    ] = None,
    registry_dir: Annotated[
        Optional[Path], typer.Option("--registry-dir", help="Registry directory")
    ] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Verbose output")
    ] = False,
):
    """Generate a world map showing Tessera embedding coverage.

    1. Create or obtain a GeoJSON/Shapefile of your region of interest
    2. Run this command to check data coverage: geotessera coverage --region-file your_region.geojson
    3. Review the coverage map to understand data availability
    4. Proceed to download data: geotessera download --region-file your_region.geojson

    Creates a PNG visualization with available tiles overlaid on a world map,
    helping users understand data availability for their regions of interest.
    Can focus on a specific region for detailed coverage analysis.
    
    By default, when no specific year is requested, the map uses three colors to show:
    - Green: All available years present for this tile
    - Blue: Only the latest year available for this tile  
    - Orange: Partial years coverage (some combination of years)

    Examples:
        # STEP 1: Check coverage for your region (recommended first step)
        geotessera coverage --region-file study_area.geojson
        geotessera coverage --region-file colombia_aoi.gpkg
        geotessera coverage --country "Colombia"

        # Check coverage for specific year only
        geotessera coverage --region-file study_area.shp --year 2024
        geotessera coverage --country "UK" --year 2024

        # Global coverage overview (all regions)
        geotessera coverage

        # Customize visualization
        geotessera coverage --region-file area.geojson --tile-alpha 0.3 --width 3000
        geotessera coverage --country "Germany" --tile-alpha 0.3 --width 3000
    """
    from .visualization import visualize_global_coverage
    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

    # Process region file or country if provided
    region_bbox = None
    country_geojson_file = None
    region_file_temp = None  # Track if we created a temporary file
    
    if region_file and country:
        rprint("[red]Error: Cannot specify both --region-file and --country. Choose one.[/red]")
        raise typer.Exit(1)
    
    if region_file:
        try:
            from .visualization import calculate_bbox_from_file
            
            # Check if region_file is a URL
            if is_url(region_file):
                rprint(f"[blue]Downloading region file from URL: {region_file}[/blue]")
                region_file_temp = download_region_file(region_file)
                region_file_path = region_file_temp
            else:
                # Check if local file exists
                region_path = Path(region_file)
                if not region_path.exists():
                    rprint(f"[red]Error: Region file {region_file} does not exist[/red]")
                    raise typer.Exit(1)
                region_file_path = str(region_path)
            
            region_bbox = calculate_bbox_from_file(region_file_path)
            rprint(f"[green]Region bounding box: {format_bbox(region_bbox)}[/green]")
        except Exception as e:
            rprint(f"[red]Error reading region file: {e}[/red]")
            raise typer.Exit(1)
    elif country:
        # Create progress bar for country data download
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TextColumn("[dim]{task.fields[status]}", justify="left"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            country_task = progress.add_task(
                "🌍 Loading country data...", total=100, status="Checking cache..."
            )

            def country_progress_callback(current: int, total: int, status: str = None):
                progress.update(
                    country_task,
                    completed=current,
                    total=total,
                    status=status or "Processing...",
                )

            try:
                # Get country lookup instance 
                from .country import get_country_lookup
                country_lookup = get_country_lookup(progress_callback=country_progress_callback)
                
                # Get both bbox and geometry
                region_bbox = country_lookup.get_bbox(country)
                country_gdf = country_lookup.get_geometry(country)
                
                # Create temporary GeoJSON file for the country boundary
                import tempfile
                import os
                with tempfile.NamedTemporaryFile(mode='w', suffix='.geojson', delete=False) as tmp:
                    country_gdf.to_file(tmp.name, driver='GeoJSON')
                    country_geojson_file = tmp.name
                
                progress.update(country_task, completed=100, status="Complete")
            except ValueError as e:
                rprint(f"[red]Error: {e}[/red]")
                rprint(
                    "[blue]Use 'geotessera countries list' to see available countries[/blue]"
                )
                raise typer.Exit(1)
            except Exception as e:
                rprint(f"[red]Error fetching country data: {e}[/red]")
                raise typer.Exit(1)
        
        rprint(f"[green]Using country '{country}': {format_bbox(region_bbox)}[/green]")

    # Initialize GeoTessera
    if verbose:
        rprint("[blue]Initializing GeoTessera...[/blue]")

    gt = GeoTessera(
        dataset_version=dataset_version,
        cache_dir=str(cache_dir) if cache_dir else None,
        registry_dir=str(registry_dir) if registry_dir else None,
    )

    # Generate coverage map
    try:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TextColumn("[dim]{task.fields[status]}", justify="left"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                "🔄 Generating coverage map...", total=100, status="Starting..."
            )

            if verbose:
                rprint(
                    f"[blue]Generating coverage map for year: {year if year else 'All years'}[/blue]"
                )

            # When using region files or countries, default to no countries for cleaner view
            show_countries_final = not no_countries and not region_file and not country
            
            # Determine which region file to use (original region file or country boundary)
            region_file_to_use = None
            if region_file:
                region_file_to_use = region_file_path if 'region_file_path' in locals() else str(region_file)
            elif country and 'country_geojson_file' in locals() and country_geojson_file:
                region_file_to_use = country_geojson_file
            
            output_path = visualize_global_coverage(
                tessera_client=gt,
                output_path=str(output),
                year=year,
                width_pixels=width_pixels,
                show_countries=show_countries_final,
                tile_color=tile_color,
                tile_alpha=tile_alpha,
                tile_size=tile_size,
                multi_year_colors=not no_multi_year_colors,
                progress_callback=create_progress_callback(progress, task),
                region_bbox=region_bbox,
                region_file=region_file_to_use,
            )

        rprint(f"[green]✅ Coverage map saved to: {output_path}[/green]")
        
        # Show next steps hint
        if region_file:
            rprint("[blue]Next step: Download data for your region:[/blue]")
            rprint(f"[cyan]  geotessera download --region-file {region_file} --output tiles/[/cyan]")
        elif country:
            rprint("[blue]Next step: Download data for your country:[/blue]")
            rprint(f"[cyan]  geotessera download --country \"{country}\" --output tiles/[/cyan]")
        else:
            rprint("[blue]Next step: Download data for a specific region:[/blue]")
            rprint("[cyan]  geotessera download --bbox 'lon1,lat1,lon2,lat2' --output tiles/[/cyan]")

        # Show summary statistics
        available_embeddings = gt.registry.get_available_embeddings()
        if available_embeddings:
            if year:
                tile_count = len(
                    [(y, lon, lat) for y, lon, lat in available_embeddings if y == year]
                )
                rprint(f"[cyan]📊 Tiles shown: {tile_count:,} (year {year})[/cyan]")
            else:
                unique_tiles = len(
                    set((lon, lat) for _, lon, lat in available_embeddings)
                )
                years = sorted(set(y for y, _, _ in available_embeddings))
                rprint(f"[cyan]📊 Unique tile locations: {unique_tiles:,}[/cyan]")
                if years:
                    rprint(f"[cyan]📅 Years covered: {min(years)}-{max(years)}[/cyan]")

    except ImportError:
        rprint("[red]Error: Missing required dependencies[/red]")
        rprint("[yellow]Please install: pip install matplotlib geodatasets[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        rprint(f"[red]Error generating coverage map: {e}[/red]")
        if verbose:
            import traceback

            traceback.print_exc()
        raise typer.Exit(1)
    finally:
        # Clean up temporary country GeoJSON file if created
        if country_geojson_file and (not region_file or country_geojson_file != str(region_file)):
            try:
                import os
                os.unlink(country_geojson_file)
            except Exception:
                pass  # Ignore cleanup errors
        
        # Clean up temporary region file if downloaded from URL
        if region_file_temp:
            try:
                import os
                os.unlink(region_file_temp)
            except Exception:
                pass  # Ignore cleanup errors


@app.command()
def download(
    output: Annotated[Path, typer.Option("--output", "-o", help="Output directory")],
    bbox: Annotated[
        Optional[str],
        typer.Option("--bbox", help="Bounding box: 'min_lon,min_lat,max_lon,max_lat'"),
    ] = None,
    region_file: Annotated[
        Optional[str],
        typer.Option(
            "--region-file", help="GeoJSON/Shapefile to define region (file path or URL)"
        ),
    ] = None,
    country: Annotated[
        Optional[str],
        typer.Option(
            "--country", help="Country name (e.g., 'United Kingdom', 'UK', 'GB')"
        ),
    ] = None,
    format: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Output format: 'tiff' (georeferenced) or 'npy' (raw arrays)",
        ),
    ] = "tiff",
    year: Annotated[int, typer.Option("--year", help="Year of embeddings")] = 2024,
    bands: Annotated[
        Optional[str],
        typer.Option("--bands", help="Comma-separated band indices (default: all 128)"),
    ] = None,
    compress: Annotated[
        str, typer.Option("--compress", help="Compression method (tiff format only)")
    ] = "lzw",
    list_files: Annotated[
        bool, typer.Option("--list-files", help="List all created files with details")
    ] = False,
    dataset_version: Annotated[
        str,
        typer.Option(
            "--dataset-version", help="Tessera dataset version (e.g., v1, v2)"
        ),
    ] = "v1",
    cache_dir: Annotated[
        Optional[Path], typer.Option("--cache-dir", help="Cache directory")
    ] = None,
    registry_dir: Annotated[
        Optional[Path], typer.Option("--registry-dir", help="Registry directory")
    ] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Verbose output")
    ] = False,
):
    """Download embeddings as numpy arrays or GeoTIFF files.

    Supports two output formats:
    - tiff: Georeferenced GeoTIFF files with proper CRS metadata (default)
    - npy: Raw numpy arrays with accompanying landmask TIFFs and metadata JSON files

    For GeoTIFF format, each tile preserves the original UTM coordinate system.
    For numpy format, arrays are saved with landmask TIFF files and comprehensive JSON metadata files alongside each .npy file.

    JSON Metadata File Format (for npy format):
    Each embedding_<lat>_<lon>.npy file has a corresponding embedding_<lat>_<lon>.json file containing:
    {
        "lat": <tile_latitude>,
        "lon": <tile_longitude>, 
        "filename": "<npy_filename>",
        "shape": [height, width, channels],
        "bands": [list of band indices included],
        "crs": "<coordinate_reference_system>",
        "transform": [rasterio transform coefficients],
        "year": <data_year>,
        "dataset_version": "<version_string>",
        "created_at": "<ISO_timestamp>",
        "download_bbox": [min_lon, min_lat, max_lon, max_lat]
    }
    """
    
    # Initialize GeoTessera
    gt = GeoTessera(
        dataset_version=dataset_version,
        cache_dir=str(cache_dir) if cache_dir else None,
        registry_dir=str(registry_dir) if registry_dir else None,
    )

    # Parse bounding box
    if bbox:
        try:
            bbox_coords = tuple(map(float, bbox.split(",")))
            if len(bbox_coords) != 4:
                rprint(
                    "[red]Error: bbox must be 'min_lon,min_lat,max_lon,max_lat'[/red]"
                )
                raise typer.Exit(1)
            rprint(f"[green]Using bounding box:[/green] {format_bbox(bbox_coords)}")
        except ValueError:
            rprint(
                "[red]Error: Invalid bbox format. Use: 'min_lon,min_lat,max_lon,max_lat'[/red]"
            )
            raise typer.Exit(1)
    elif region_file:
        try:
            # Check if region_file is a URL
            if is_url(region_file):
                rprint(f"[blue]Downloading region file from URL: {region_file}[/blue]")
                region_file_path = download_region_file(region_file)
                region_file_temp = region_file_path  # Track for cleanup
            else:
                # Check if local file exists
                region_path = Path(region_file)
                if not region_path.exists():
                    rprint(f"[red]Error: Region file {region_file} does not exist[/red]")
                    raise typer.Exit(1)
                region_file_path = str(region_path)
                region_file_temp = None
            
            bbox_coords = calculate_bbox_from_file(region_file_path)
            rprint(
                f"[green]Calculated bbox from {region_file}:[/green] {format_bbox(bbox_coords)}"
            )
        except Exception as e:
            rprint(f"[red]Error reading region file: {e}[/red]")
            rprint("Supported formats: GeoJSON, Shapefile, etc.")
            # Clean up temp file if we created one
            if 'region_file_temp' in locals() and region_file_temp:
                try:
                    import os
                    os.unlink(region_file_temp)
                except Exception:
                    pass
            raise typer.Exit(1)
    elif country:
        # Create progress bar for country data download
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TextColumn("[dim]{task.fields[status]}", justify="left"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            country_task = progress.add_task(
                "🌍 Loading country data...", total=100, status="Checking cache..."
            )

            def country_progress_callback(current: int, total: int, status: str = None):
                progress.update(
                    country_task,
                    completed=current,
                    total=total,
                    status=status or "Processing...",
                )

            try:
                bbox_coords = get_country_bbox(
                    country, progress_callback=country_progress_callback
                )
                progress.update(country_task, completed=100, status="Complete")
                rprint(
                    f"[green]Using country '{country}':[/green] {format_bbox(bbox_coords)}"
                )
            except ValueError as e:
                rprint(f"[red]Error: {e}[/red]")
                rprint(
                    "[blue]Use 'geotessera countries list' to see available countries[/blue]"
                )
                raise typer.Exit(1)
            except Exception as e:
                rprint(f"[red]Error fetching country data: {e}[/red]")
                raise typer.Exit(1)
    else:
        rprint(
            "[red]Error: Must specify either --bbox, --region-file, or --country[/red]"
        )
        rprint("Examples:")
        rprint("  --bbox '-0.2,51.4,0.1,51.6'  # London area")
        rprint("  --region-file london.geojson  # From GeoJSON file")
        rprint("  --country 'United Kingdom'    # Country by name")
        raise typer.Exit(1)

    # Parse bands
    bands_list = None
    if bands:
        try:
            bands_list = list(map(int, bands.split(",")))
            rprint(
                f"[blue]Exporting {len(bands_list)} selected bands:[/blue] {bands_list}"
            )
        except ValueError:
            rprint("[red]Error: bands must be comma-separated integers (0-127)[/red]")
            rprint("Example: --bands '0,1,2' for first 3 bands")
            raise typer.Exit(1)
    else:
        rprint("[blue]Exporting all 128 bands[/blue]")

    # Validate format
    if format not in ["tiff", "npy"]:
        rprint(f"[red]Error: Invalid format '{format}'. Must be 'tiff' or 'npy'[/red]")
        raise typer.Exit(1)

    # Display export info
    info_table = Table(show_header=False, box=None)
    info_table.add_row("Format:", format.upper())
    info_table.add_row("Year:", str(year))
    info_table.add_row("Output directory:", str(output))
    if format == "tiff":
        info_table.add_row("Compression:", compress)
    info_table.add_row("Dataset version:", dataset_version)

    rprint(
        Panel(
            info_table,
            title=f"[bold]GeoTessera v{__version__} - Region of Interest Download[/bold]",
            border_style="blue",
        )
    )

    try:
        
        # Export tiles with progress tracking
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TextColumn("[dim]{task.fields[status]}", justify="left"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                "🔄 Processing tiles...", total=100, status="Starting..."
            )

            if format == "tiff":
                # Export as GeoTIFF files
                files = gt.export_embedding_geotiffs(
                    bbox=bbox_coords,
                    output_dir=output,
                    year=year,
                    bands=bands_list,
                    compress=compress,
                    progress_callback=create_download_progress_callback(progress, task),
                )

                if not files:
                    rprint(
                        "[yellow]⚠️  No tiles found in the specified region.[/yellow]"
                    )
                    rprint(
                        "Try expanding your bounding box or checking data availability."
                    )
                    return

                rprint(
                    f"\n[green]✅ SUCCESS: Exported {len(files)} GeoTIFF files[/green]"
                )
                rprint(
                    "   Each file preserves its native UTM projection from landmask tiles"
                )
                rprint("   Files can be individually inspected and processed")

            else:  # format == 'npy'
                # Export as numpy arrays
                # Fetch embeddings as numpy arrays
                embeddings = gt.fetch_embeddings(
                    bbox=bbox_coords,
                    year=year,
                    progress_callback=create_download_progress_callback(progress, task),
                )

                if not embeddings:
                    rprint(
                        "[yellow]⚠️  No tiles found in the specified region.[/yellow]"
                    )
                    rprint(
                        "Try expanding your bounding box or checking data availability."
                    )
                    return

                # Create output directory
                output.mkdir(parents=True, exist_ok=True)

                import datetime
                
                files = []

                for tile_lon, tile_lat, embedding_array, crs, transform in embeddings:
                    # Apply band selection if specified
                    if bands_list:
                        embedding_array = embedding_array[:, :, bands_list]

                    # Save numpy array
                    filename = f"embedding_{tile_lat:.2f}_{tile_lon:.2f}.npy"
                    filepath = output / filename
                    np.save(filepath, embedding_array)
                    files.append(str(filepath))

                    # Download and save landmask TIFF file
                    try:
                        from .registry import tile_to_landmask_filename
                        
                        landmask_filename = tile_to_landmask_filename(tile_lon, tile_lat)
                        
                        # Ensure tile registry block is loaded
                        gt.registry.ensure_tile_block_loaded(tile_lon, tile_lat)
                        
                        # Fetch landmask file
                        landmask_cache_path = gt.registry.fetch_landmask(
                            landmask_filename, progressbar=False
                        )
                        
                        # Copy landmask to output directory
                        landmask_output_filename = f"landmask_{tile_lat:.2f}_{tile_lon:.2f}.tif"
                        landmask_output_path = output / landmask_output_filename
                        
                        import shutil
                        shutil.copy2(landmask_cache_path, landmask_output_path)
                        files.append(str(landmask_output_path))
                        
                        landmask_info = {
                            "landmask_filename": landmask_output_filename,
                            "landmask_cache_path": landmask_cache_path,
                            "landmask_size_bytes": landmask_output_path.stat().st_size,
                        }
                        
                    except Exception as e:
                        print(f"Warning: Could not download landmask for tile ({tile_lat:.2f}, {tile_lon:.2f}): {e}")
                        landmask_info = {
                            "landmask_filename": None,
                            "landmask_error": str(e)
                        }

                    # Create comprehensive metadata for this tile
                    tile_metadata = {
                        "lat": tile_lat,
                        "lon": tile_lon,
                        "filename": filename,
                        "shape": list(embedding_array.shape),
                        "bands": bands_list if bands_list else list(range(128)),
                        "crs": str(crs) if crs else None,
                        "transform": list(transform) if transform else None,
                        "year": year,
                        "dataset_version": dataset_version,
                        "created_at": datetime.datetime.now().isoformat(),
                        "download_bbox": list(bbox_coords),
                        "tile_bounds": [tile_lon - 0.05, tile_lat - 0.05, tile_lon + 0.05, tile_lat + 0.05],
                        "landmask": landmask_info,
                        "file_size_bytes": filepath.stat().st_size,
                        "compression": "none",
                        "data_type": str(embedding_array.dtype)
                    }

                    # Save comprehensive JSON metadata file
                    json_filename = f"embedding_{tile_lat:.2f}_{tile_lon:.2f}.json"
                    json_filepath = output / json_filename
                    with open(json_filepath, "w") as f:
                        json.dump(tile_metadata, f, indent=2)
                    files.append(str(json_filepath))

                rprint(
                    f"\n[green]✅ SUCCESS: Exported {len(embeddings)} numpy arrays[/green]"
                )
                rprint("   Landmask TIFF files downloaded for each tile")
                rprint("   Comprehensive JSON metadata files created for each tile")
                rprint(
                    f"   Arrays contain {'selected bands' if bands_list else 'all 128 bands'}"
                )

        if verbose or list_files:
            rprint("\n[blue]📁 Created files:[/blue]")
            file_table = Table(show_header=True, header_style="bold blue")
            file_table.add_column("#", style="dim", width=3)
            file_table.add_column("Filename")
            file_table.add_column("Size", justify="right")

            for i, f in enumerate(files, 1):
                file_path = Path(f)
                file_size = file_path.stat().st_size if file_path.exists() else 0
                file_table.add_row(str(i), file_path.name, f"{file_size:,} bytes")

            console.print(file_table)
        elif len(files) > 0:
            rprint(
                "\n[blue]📁 Sample files (use --verbose or --list-files to see all):[/blue]"
            )
            for f in files[:3]:
                file_path = Path(f)
                file_size = file_path.stat().st_size if file_path.exists() else 0
                rprint(f"     {file_path.name} ({file_size:,} bytes)")
            if len(files) > 3:
                rprint(f"     ... and {len(files) - 3} more files")

        # Show spatial information
        rprint("\n[blue]🗺️  Spatial Information:[/blue]")
        if verbose:
            try:
                import rasterio

                with rasterio.open(files[0]) as src:
                    rprint(f"   CRS: {src.crs}")
                    rprint(f"   Transform: {src.transform}")
                    rprint(f"   Dimensions: {src.width} x {src.height} pixels")
                    rprint(f"   Data type: {src.dtypes[0]}")
            except Exception:
                pass

        rprint(f"   Output directory: {Path(output).resolve()}")

        tips_table = Table(show_header=False, box=None)
        tips_table.add_row("• Inspect individual tiles with QGIS, GDAL, or rasterio")
        tips_table.add_row("• Use 'gdalinfo <filename>' to see projection details")
        tips_table.add_row("• Process tiles individually or in groups as needed")
        if format == "tiff":
            tips_table.add_row("• Create PCA visualization:")
            tips_table.add_row(f"  [cyan]geotessera visualize {output} pca_mosaic.tif[/cyan]")

        rprint(
            Panel(tips_table, title="[bold]💡 Next steps[/bold]", border_style="green")
        )

    except Exception as e:
        rprint(f"\n[red]❌ Error: {e}[/red]")
        if verbose:
            rprint("\n[dim]Full traceback:[/dim]")
            console.print_exception()
        raise typer.Exit(1)
    finally:
        # Clean up temporary region file if downloaded from URL
        if 'region_file_temp' in locals() and region_file_temp:
            try:
                import os
                os.unlink(region_file_temp)
            except Exception:
                pass  # Ignore cleanup errors


@app.command()
def visualize(
    input_path: Annotated[Path, typer.Argument(help="Input GeoTIFF file or directory")],
    output_file: Annotated[Path, typer.Argument(help="Output PCA mosaic file (.tif)")],
    target_crs: Annotated[
        str, typer.Option("--crs", help="Target CRS for reprojection")
    ] = "EPSG:3857",
    n_components: Annotated[
        int, typer.Option("--n-components", help="Number of PCA components. Only first 3 used for RGB visualization - increase for analysis/research.")
    ] = 3,
    balance_method: Annotated[
        str, typer.Option("--balance", help="RGB balance method: histogram (default), percentile, or adaptive")
    ] = "histogram",
    percentile_low: Annotated[
        float, typer.Option("--percentile-low", help="Lower percentile for percentile balance method")
    ] = 2.0,
    percentile_high: Annotated[
        float, typer.Option("--percentile-high", help="Upper percentile for percentile balance method")
    ] = 98.0,
):
    """Create PCA visualization from multiband GeoTIFF files.
    
    This command combines all embedding data across tiles, applies a single PCA
    transformation to the combined dataset, then creates a unified RGB mosaic.
    This ensures consistent principal components across the entire region,
    eliminating tiling artifacts.
    
    The first 3 principal components are mapped to RGB channels for visualization.
    Additional components can be computed for research/analysis purposes.
    
    Examples:
        # Create PCA visualization (3 components optimal for RGB)
        geotessera visualize tiles/ pca_mosaic.tif
        
        # Use histogram equalization for maximum contrast
        geotessera visualize tiles/ pca_balanced.tif --balance histogram
        
        # Use adaptive scaling based on variance
        geotessera visualize tiles/ pca_adaptive.tif --balance adaptive
        
        # Custom percentile range for outlier-robust scaling
        geotessera visualize tiles/ pca_custom.tif --percentile-low 5 --percentile-high 95
        
        # Use custom projection
        geotessera visualize tiles/ pca_mosaic.tif --crs EPSG:4326
        
        # PCA for research - compute more components for analysis
        # (still only uses first 3 for RGB, but saves variance info)
        geotessera visualize tiles/ pca_research.tif --n-components 10
        
        # Then create web visualization
        geotessera webmap pca_mosaic.tif --serve
    """
    
    # Validate output file extension
    if output_file.suffix.lower() not in [".tif", ".tiff"]:
        rprint("[red]Error: Output file must have .tif or .tiff extension[/red]")
        raise typer.Exit(1)
        
    # Validate n_components
    if n_components < 1:
        rprint("[red]Error: Number of components must be at least 1[/red]")
        raise typer.Exit(1)
    if n_components < 3:
        rprint(f"[yellow]Warning: Using {n_components} component(s). RGB visualization works best with 3+ components[/yellow]")
    
    # Validate balance_method
    if balance_method not in ["percentile", "histogram", "adaptive"]:
        rprint(f"[red]Error: Invalid balance method '{balance_method}'. Must be 'percentile', 'histogram', or 'adaptive'[/red]")
        raise typer.Exit(1)
    
    # Validate percentile ranges
    if balance_method == "percentile":
        if not (0 <= percentile_low < percentile_high <= 100):
            rprint(f"[red]Error: Invalid percentile range [{percentile_low}, {percentile_high}]. Must be 0 <= low < high <= 100[/red]")
            raise typer.Exit(1)

    # Find GeoTIFF files
    if input_path.is_file():
        geotiff_paths = [str(input_path)]
    else:
        geotiff_paths = list(map(str, input_path.glob("*.tif")))
        geotiff_paths.extend(map(str, input_path.glob("*.tiff")))

    if not geotiff_paths:
        rprint(f"[red]No GeoTIFF files found in {input_path}[/red]")
        raise typer.Exit(1)

    rprint(f"[blue]Found {len(geotiff_paths)} GeoTIFF files[/blue]")

    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TextColumn("[dim]{task.fields[status]}", justify="left"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Creating PCA mosaic ({n_components} components)...", total=5, status="Starting..."
        )

        try:
            # Create a progress callback that maps to our 5-step progress
            def visualization_progress_callback(current: float, total: float, status: str = None):
                progress.update(task, completed=current, total=total, status=status or "Processing...")
            
            # PCA MODE: Use clean visualization function
            create_pca_mosaic(
                geotiff_paths=geotiff_paths,
                output_path=output_file,
                n_components=n_components,
                target_crs=target_crs,
                progress_callback=visualization_progress_callback,
                balance_method=balance_method,
                percentile_range=(percentile_low, percentile_high),
            )
            
            progress.update(task, completed=5, total=5, status="Complete")

        except Exception as e:
            rprint(f"[red]Error creating PCA visualization: {e}[/red]")
            raise typer.Exit(1)
    
    # Success output after progress bar completes
    rprint(f"[green]Created PCA mosaic: {output_file}[/green]")
    rprint(f"[blue]Components: {n_components} | CRS: {target_crs}[/blue]")
    rprint("[blue]Next step: Create web visualization with:[/blue]")
    rprint(f"[cyan]  geotessera webmap {output_file} --serve[/cyan]")


@app.command()
def webmap(
    rgb_mosaic: Annotated[Path, typer.Argument(help="3-band RGB mosaic GeoTIFF file")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Output directory")] = None,
    min_zoom: Annotated[int, typer.Option("--min-zoom", help="Min zoom for web tiles")] = 8,
    max_zoom: Annotated[int, typer.Option("--max-zoom", help="Max zoom for web tiles")] = 15,
    initial_zoom: Annotated[int, typer.Option("--initial-zoom", help="Initial zoom level")] = 10,
    force_regenerate: Annotated[
        bool,
        typer.Option(
            "--force/--no-force", help="Force regeneration of tiles even if they exist"
        ),
    ] = False,
    serve_immediately: Annotated[
        bool, typer.Option("--serve/--no-serve", help="Start web server immediately")
    ] = False,
    port: Annotated[int, typer.Option("--port", "-p", help="Port for web server")] = 8000,
    region_file: Annotated[
        Optional[str],
        typer.Option(
            "--region-file",
            help="GeoJSON/Shapefile boundary to overlay (file path or URL)",
        ),
    ] = None,
    use_gdal_raster: Annotated[
        bool,
        typer.Option(
            "--use-gdal-raster/--use-gdal2tiles",
            help="Use newer gdal raster tile (faster but less stable) vs gdal2tiles (default, stable)",
        ),
    ] = False,
):
    """Create web tiles and viewer from a 3-band RGB mosaic.
    
    This command takes an RGB GeoTIFF mosaic, reprojects it if needed for web viewing,
    generates web tiles, creates an HTML viewer, and optionally starts a web server.
    
    Example workflow:
        1. geotessera download --bbox lon1,lat1,lon2,lat2 tiles/
        2. geotessera visualize tiles/ --type rgb --output mosaics/
        3. geotessera webmap mosaics/rgb_mosaic.tif --output webmap/ --serve
    """
    if not rgb_mosaic.exists():
        rprint(f"[red]Error: Mosaic file {rgb_mosaic} does not exist[/red]")
        raise typer.Exit(1)
    
    if rgb_mosaic.suffix.lower() not in ['.tif', '.tiff']:
        rprint("[red]Error: Input must be a GeoTIFF file (.tif/.tiff)[/red]")
        raise typer.Exit(1)
    
    # Handle region file URL download
    region_file_temp = None
    region_file_path = None
    if region_file:
        try:
            if is_url(region_file):
                rprint(f"[blue]Downloading region file from URL: {region_file}[/blue]")
                region_file_temp = download_region_file(region_file)
                region_file_path = region_file_temp
            else:
                # Check if local file exists
                region_path = Path(region_file)
                if not region_path.exists():
                    rprint(f"[red]Error: Region file {region_file} does not exist[/red]")
                    raise typer.Exit(1)
                region_file_path = str(region_path)
        except Exception as e:
            rprint(f"[red]Error processing region file: {e}[/red]")
            # Clean up temp file if we created one
            if region_file_temp:
                try:
                    import os
                    os.unlink(region_file_temp)
                except Exception:
                    pass
            raise typer.Exit(1)
    
    # Default output directory
    if output is None:
        output = Path(f"{rgb_mosaic.stem}_webmap")
    
    output.mkdir(parents=True, exist_ok=True)
    
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TextColumn("[dim]{task.fields[status]}", justify="left"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        
        # Step 1: Prepare mosaic for web (reproject if needed)
        web_mosaic_path = output / "web_ready_mosaic.tif"
        
        task1 = progress.add_task(
            "Preparing mosaic for web...", total=100, status="Starting..."
        )
        
        try:
            actual_mosaic_path = prepare_mosaic_for_web(
                input_mosaic=str(rgb_mosaic),
                output_path=str(web_mosaic_path),
                target_crs="EPSG:3857",
                progress_callback=create_progress_callback(progress, task1),
            )
            
            # If no reprojection was needed, use original file
            if actual_mosaic_path == str(rgb_mosaic):
                actual_mosaic_path = str(rgb_mosaic)
                mosaic_status = "Using original mosaic (already in correct CRS)"
            else:
                mosaic_status = f"Created web-ready mosaic: {web_mosaic_path}"
                
        except Exception as e:
            rprint(f"[red]Error preparing mosaic: {e}[/red]")
            raise typer.Exit(1)
        
        # Step 2: Generate web tiles
        tiles_dir = output / "tiles"
        
        # Check if we should regenerate tiles
        if force_regenerate and tiles_dir.exists():
            import shutil
            shutil.rmtree(tiles_dir)
            tiles_regenerated = True
        else:
            tiles_regenerated = False
        
        tiles_force_hint = None
        
        if not tiles_dir.exists() or not any(tiles_dir.iterdir()):
            task2 = progress.add_task(
                "Generating web tiles...", total=100, status="Starting..."
            )
            
            try:
                result_dir = geotiff_to_web_tiles(
                    geotiff_path=actual_mosaic_path,
                    output_dir=str(tiles_dir),
                    zoom_levels=(min_zoom, max_zoom),
                    use_gdal_raster=use_gdal_raster,
                )
                progress.update(task2, completed=100)
                tiles_status = f"Created web tiles in: {result_dir}"
                
            except Exception as e:
                rprint(f"[red]Error generating web tiles: {e}[/red]")
                raise typer.Exit(1)
        else:
            tiles_status = f"Using existing tiles in: {tiles_dir}"
            tiles_force_hint = "Use --force to regenerate tiles"
        
        # Step 3: Create HTML viewer
        html_path = output / "viewer.html"
        
        task3 = progress.add_task(
            "Creating web viewer...", total=100, status="Starting..."
        )
        
        try:
            # Get mosaic bounds for centering
            import rasterio
            with rasterio.open(actual_mosaic_path) as src:
                bounds = src.bounds
                # Transform bounds to lat/lon if needed
                if src.crs != 'EPSG:4326':
                    from rasterio.warp import transform_bounds
                    lon_min, lat_min, lon_max, lat_max = transform_bounds(
                        src.crs, 'EPSG:4326', bounds.left, bounds.bottom, bounds.right, bounds.top
                    )
                else:
                    lon_min, lat_min, lon_max, lat_max = bounds.left, bounds.bottom, bounds.right, bounds.top
                    
                center_lat = (lat_min + lat_max) / 2
                center_lon = (lon_min + lon_max) / 2
            
            create_simple_web_viewer(
                tiles_dir=str(tiles_dir),
                output_html=str(html_path),
                center_lon=center_lon,
                center_lat=center_lat,
                zoom=initial_zoom,
                title=f"GeoTessera v{__version__} - {rgb_mosaic.name}",
                region_file=region_file_path if region_file_path else None,
            )
            
            progress.update(task3, completed=100)
            viewer_status = f"Created web viewer: {html_path}"
            
        except Exception as e:
            rprint(f"[red]Error creating web viewer: {e}[/red]")
            raise typer.Exit(1)
    
    # Summary
    rprint(f"\n[green]✅ Web visualization ready in: {output}[/green]")
    
    # Print status messages from the progress context
    rprint(f"[green]{mosaic_status}[/green]")
    
    if tiles_regenerated:
        rprint("[yellow]Removed existing tiles directory for regeneration[/yellow]")
    
    rprint(f"[green]{tiles_status}[/green]")
    if tiles_force_hint:
        rprint(f"[blue]{tiles_force_hint}[/blue]")
    
    rprint(f"[green]{viewer_status}[/green]")
    
    if serve_immediately:
        rprint("[blue]Starting web server...[/blue]")
        # Call the serve function directly
        try:
            serve(directory=output, port=port, open_browser=True, html_file="viewer.html")
        except KeyboardInterrupt:
            rprint("\n[green]Web server stopped.[/green]")
        except Exception as e:
            rprint(f"[yellow]Could not start server automatically: {e}[/yellow]")
            rprint("[blue]To view the map, start a web server manually:[/blue]")
            rprint(f"[cyan]  geotessera serve {output} --port {port}[/cyan]")
    else:
        rprint("[blue]To view the map, start a web server:[/blue]")
        rprint(f"[cyan]  geotessera serve {output} --port {port}[/cyan]")
    
    # Clean up temporary region file if downloaded from URL
    if region_file_temp:
        try:
            import os
            os.unlink(region_file_temp)
        except Exception:
            pass  # Ignore cleanup errors


@app.command()
def serve(
    directory: Annotated[
        Path, typer.Argument(help="Directory containing web visualization files")
    ],
    port: Annotated[
        int, typer.Option("--port", "-p", help="Port number for web server")
    ] = 8000,
    open_browser: Annotated[
        bool, typer.Option("--open/--no-open", help="Automatically open browser")
    ] = True,
    html_file: Annotated[
        Optional[str],
        typer.Option(
            "--html", help="Specific HTML file to serve (relative to directory)"
        ),
    ] = None,
):
    """Start a web server to serve visualization files.

    This is needed for leaflet-based web visualizations to work properly
    since they require HTTP access to load tiles and other resources.
    """
    if not directory.exists():
        rprint(f"[red]Error: Directory {directory} does not exist[/red]")
        raise typer.Exit(1)

    if not directory.is_dir():
        rprint(f"[red]Error: {directory} is not a directory[/red]")
        raise typer.Exit(1)

    # Change to the directory to serve
    original_dir = Path.cwd()

    class QuietHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            # Only log errors, not every request
            if args[1] != "200":
                super().log_message(format, *args)

    try:
        # Find available port
        while True:
            try:
                with socketserver.TCPServer(("", port), QuietHTTPRequestHandler):
                    break
            except OSError:
                port += 1
                if port > 9000:
                    rprint("[red]Error: Could not find available port[/red]")
                    raise typer.Exit(1)

        rprint(f"[green]Starting web server on port {port}[/green]")
        rprint(f"[blue]Serving directory: {directory.absolute()}[/blue]")

        # Debug: Show directory contents
        try:
            contents = list(directory.iterdir())
            rprint(
                f"[yellow]Directory contains: {[p.name for p in contents[:10]]}{'...' if len(contents) > 10 else ''}[/yellow]"
            )
        except Exception as e:
            rprint(f"[yellow]Could not list directory contents: {e}[/yellow]")

        # Determine what to open in browser
        if html_file:
            html_path = directory / html_file
            if not html_path.exists():
                rprint(f"[yellow]Warning: HTML file {html_file} not found[/yellow]")
                browser_url = f"http://localhost:{port}/"
            else:
                browser_url = f"http://localhost:{port}/{html_file}"
        else:
            # Look for common HTML files
            common_names = ["index.html", "viewer.html", "map.html", "coverage.html"]
            found_html = None
            for name in common_names:
                if (directory / name).exists():
                    found_html = name
                    break

            if found_html:
                browser_url = f"http://localhost:{port}/{found_html}"
                rprint(f"[blue]Found HTML file: {found_html}[/blue]")
            else:
                browser_url = f"http://localhost:{port}/"

        # Start server in background thread
        def start_server():
            import os

            os.chdir(directory)
            try:
                with socketserver.TCPServer(
                    ("", port), QuietHTTPRequestHandler
                ) as httpd:
                    httpd.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                os.chdir(original_dir)

        server_thread = threading.Thread(target=start_server, daemon=True)
        server_thread.start()

        # Give server a moment to start
        time.sleep(0.5)

        rprint(f"[green]✅ Web server running at: http://localhost:{port}/[/green]")

        if open_browser:
            rprint(f"[blue]Opening browser: {browser_url}[/blue]")
            webbrowser.open(browser_url)
        else:
            rprint(f"[blue]Open in browser: {browser_url}[/blue]")

        rprint("\n[yellow]Press Ctrl+C to stop the server[/yellow]")

        try:
            # Keep main thread alive
            while server_thread.is_alive():
                time.sleep(1)
        except KeyboardInterrupt:
            rprint("\n[green]Stopping web server...[/green]")
            raise typer.Exit(0)

    except Exception as e:
        rprint(f"[red]Error starting web server: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def version():
    """Print the geotessera library version."""
    from geotessera import __version__
    print(__version__)


def main():
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()
