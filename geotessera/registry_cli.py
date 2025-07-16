#!/usr/bin/env python3
"""
Command-line interface for managing GeoTessera registry files.

This module provides tools for generating and maintaining Pooch registry files
used by the GeoTessera package. It supports parallel processing, incremental
updates, and generation of a master registry index.
"""

import os
import hashlib
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
import multiprocessing

from .registry_utils import (
    get_block_coordinates, 
    get_block_registry_filename,
    parse_grid_coordinates,
    BLOCK_SIZE
)


def calculate_sha256(file_path):
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def process_file(args):
    """Process a single file and return its relative path and hash."""
    file_path, base_dir, skip_checksum = args
    try:
        rel_path = os.path.relpath(file_path, base_dir)
        if skip_checksum:
            file_hash = ""
        else:
            file_hash = calculate_sha256(file_path)
        return rel_path, file_hash
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None


def load_existing_registry(registry_path):
    """Load existing registry file into a dictionary."""
    registry = {}
    if os.path.exists(registry_path):
        with open(registry_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(' ', 1)
                    if len(parts) == 2:
                        registry[parts[0]] = parts[1]
    return registry


def find_npy_files_by_blocks(base_dir):
    """Find all .npy files and organize them by year and block."""
    files_by_year_and_block = defaultdict(lambda: defaultdict(list))

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.npy'):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, base_dir)

                # Extract year from path (assuming format ./YYYY/...)
                path_parts = rel_path.split(os.sep)
                if len(path_parts) > 0 and path_parts[0].isdigit() and len(path_parts[0]) == 4:
                    year = path_parts[0]
                    
                    # Extract coordinates from the grid directory name
                    grid_dir = os.path.basename(os.path.dirname(file_path))
                    lon, lat = parse_grid_coordinates(grid_dir)
                    
                    if lon is not None and lat is not None:
                        block_lon, block_lat = get_block_coordinates(lon, lat)
                        block_key = (block_lon, block_lat)
                        files_by_year_and_block[year][block_key].append(file_path)

    return files_by_year_and_block




def find_tiff_files_by_blocks(base_dir):
    """Find all .tiff files and organize them by block."""
    files_by_block = defaultdict(list)

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.tiff'):
                file_path = os.path.join(root, file)
                
                # Extract coordinates from the tiff filename (e.g., grid_-120.55_53.45.tiff)
                filename = os.path.basename(file_path)
                tiff_name = filename.replace('.tiff', '')
                lon, lat = parse_grid_coordinates(tiff_name)
                
                if lon is not None and lat is not None:
                    block_lon, block_lat = get_block_coordinates(lon, lat)
                    block_key = (block_lon, block_lat)
                    files_by_block[block_key].append(file_path)

    return files_by_block




def update_representations_command(args):
    """Update block-based registry files for .npy representation files."""
    base_dir = os.path.abspath(args.base_dir)
    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} does not exist")
        return

    # Determine registry output directory - go up to parent of base_dir
    if hasattr(args, 'registry_dir') and args.registry_dir:
        registry_dir = os.path.join(os.path.abspath(args.registry_dir), "registry")
    else:
        # Place registry at same level as global_0.1_degree_representation
        parent_dir = os.path.dirname(base_dir)
        registry_dir = os.path.join(parent_dir, "registry")
    
    # Ensure registry directory exists
    os.makedirs(registry_dir, exist_ok=True)
    print(f"Registry files will be written to: {registry_dir}")

    # Set number of workers
    num_workers = args.workers or multiprocessing.cpu_count()
    print(f"Using {num_workers} parallel workers")
    
    if getattr(args, 'skip_checksum', False):
        print("WARNING: Skipping checksum calculation - registry will not verify file integrity")

    # Find all .npy files organized by year and block
    print("Scanning for .npy representation files...")
    files_by_year_and_block = find_npy_files_by_blocks(base_dir)

    if not files_by_year_and_block:
        print("No .npy files found")
        return

    total_blocks = 0
    all_registry_files = []
    
    # Process each year
    for year in sorted(files_by_year_and_block.keys()):
        blocks_for_year = files_by_year_and_block[year]
        print(f"\nProcessing year {year}: {len(blocks_for_year)} blocks")
        total_blocks += len(blocks_for_year)
        
        year_registry_files = []

        # Process each block within the year
        for (block_lon, block_lat), block_files in sorted(blocks_for_year.items()):
            registry_filename = get_block_registry_filename(year, block_lon, block_lat)
            registry_file = os.path.join(registry_dir, registry_filename)
            year_registry_files.append(registry_file)
            all_registry_files.append(registry_file)
            
            print(f"  Block ({block_lon}, {block_lat}): {len(block_files)} files -> {registry_filename}")

            # Load existing registry
            existing_registry = load_existing_registry(registry_file)
            
            # Determine which files need processing
            files_to_process = []
            for file_path in block_files:
                rel_path = os.path.relpath(file_path, base_dir)
                
                if hasattr(args, 'force') and args.force:
                    if rel_path in existing_registry:
                        try:
                            current_hash = calculate_sha256(file_path)
                            if current_hash != existing_registry[rel_path]:
                                files_to_process.append(file_path)
                        except Exception as e:
                            print(f"    Error checking {rel_path}: {e}, will update")
                            files_to_process.append(file_path)
                    else:
                        files_to_process.append(file_path)
                else:
                    if rel_path not in existing_registry:
                        files_to_process.append(file_path)

            if not files_to_process:
                continue

            print(f"    Processing {len(files_to_process)} files...")

            # Process files in parallel
            new_entries = {}
            skip_checksum = getattr(args, 'skip_checksum', False)
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                future_to_file = {
                    executor.submit(process_file, (file_path, base_dir, skip_checksum)): file_path
                    for file_path in files_to_process
                }

                for future in as_completed(future_to_file):
                    rel_path, file_hash = future.result()
                    if rel_path and file_hash:
                        new_entries[rel_path] = file_hash

            # Merge with existing registry
            final_registry = existing_registry.copy()
            final_registry.update(new_entries)

            # Write registry file
            with open(registry_file, 'w') as f:
                for rel_path in sorted(final_registry.keys()):
                    checksum = final_registry[rel_path]
                    if checksum:
                        f.write(f"{rel_path} {checksum}\n")
                    else:
                        f.write(f"{rel_path}\n")

            print(f"    Written: {len(final_registry)} total entries, {len(new_entries)} new")

    # Generate master index of all registry files
    if all_registry_files:
        master_index_path = os.path.join(registry_dir, "registry_index.txt")
        print(f"\nGenerating master registry index: {master_index_path}")
        
        with open(master_index_path, 'w') as f:
            for registry_file in sorted(all_registry_files):
                filename = os.path.basename(registry_file)
                f.write(f"{filename}\n")
        
        print(f"Master index written with {len(all_registry_files)} registry files")
        print(f"Total blocks processed: {total_blocks}")
        
        # Generate master registry.txt that includes all individual registries
        master_registry_path = os.path.join(registry_dir, "registry.txt")
        print(f"Generating master registry.txt: {master_registry_path}")
        
        with open(master_registry_path, 'w') as f:
            total_entries = 0
            for registry_file in sorted(all_registry_files):
                if os.path.exists(registry_file):
                    with open(registry_file, 'r') as reg_f:
                        for line in reg_f:
                            line = line.strip()
                            if line and not line.startswith('#'):
                                f.write(f"{line}\n")
                                total_entries += 1
        
        print(f"Master registry.txt written with {total_entries} total entries")



def update_tiles_command(args):
    """Update block-based registry files for .tiff tile files."""
    base_dir = os.path.abspath(args.base_dir)
    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} does not exist")
        return

    # Determine registry output directory - go up to parent of base_dir
    if hasattr(args, 'registry_dir') and args.registry_dir:
        registry_dir = os.path.join(os.path.abspath(args.registry_dir), "registry")
    else:
        # Place registry at same level as global_0.1_degree_tiff_all
        parent_dir = os.path.dirname(base_dir)
        registry_dir = os.path.join(parent_dir, "registry")
    
    # Ensure registry directory exists
    os.makedirs(registry_dir, exist_ok=True)
    print(f"Registry files will be written to: {registry_dir}")

    # Set number of workers
    num_workers = args.workers or multiprocessing.cpu_count()
    print(f"Using {num_workers} parallel workers")
    
    if getattr(args, 'skip_checksum', False):
        print("WARNING: Skipping checksum calculation - registry will not verify file integrity")

    # Find all .tiff files organized by block
    print("Scanning for .tiff tile files...")
    files_by_block = find_tiff_files_by_blocks(base_dir)

    if not files_by_block:
        print("No .tiff files found")
        return

    total_blocks = len(files_by_block)
    all_registry_files = []
    
    print(f"Found {total_blocks} blocks with TIFF files")

    # Process each block
    for (block_lon, block_lat), block_files in sorted(files_by_block.items()):
        # For tiles, we don't have years, so use a generic naming scheme
        registry_filename = f"registry_tiles_lon{block_lon}_lat{block_lat}.txt"
        registry_file = os.path.join(registry_dir, registry_filename)
        all_registry_files.append(registry_file)
        
        print(f"  Block ({block_lon}, {block_lat}): {len(block_files)} files -> {registry_filename}")

        # Load existing registry
        existing_registry = load_existing_registry(registry_file)
        
        # Determine which files need processing
        files_to_process = []
        for file_path in block_files:
            rel_path = os.path.relpath(file_path, base_dir)
            
            if hasattr(args, 'force') and args.force:
                if rel_path in existing_registry:
                    try:
                        current_hash = calculate_sha256(file_path)
                        if current_hash != existing_registry[rel_path]:
                            files_to_process.append(file_path)
                    except Exception as e:
                        print(f"    Error checking {rel_path}: {e}, will update")
                        files_to_process.append(file_path)
                else:
                    files_to_process.append(file_path)
            else:
                if rel_path not in existing_registry:
                    files_to_process.append(file_path)

        if not files_to_process:
            continue

        print(f"    Processing {len(files_to_process)} files...")

        # Process files in parallel
        new_entries = {}
        skip_checksum = getattr(args, 'skip_checksum', False)
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_file = {
                executor.submit(process_file, (file_path, base_dir, skip_checksum)): file_path
                for file_path in files_to_process
            }

            for future in as_completed(future_to_file):
                rel_path, file_hash = future.result()
                if rel_path and file_hash:
                    new_entries[rel_path] = file_hash

        # Merge with existing registry
        final_registry = existing_registry.copy()
        final_registry.update(new_entries)

        # Write registry file
        with open(registry_file, 'w') as f:
            for rel_path in sorted(final_registry.keys()):
                checksum = final_registry[rel_path]
                if checksum:
                    f.write(f"{rel_path} {checksum}\n")
                else:
                    f.write(f"{rel_path}\n")

        print(f"    Written: {len(final_registry)} total entries, {len(new_entries)} new")

    # Generate master index of all tile registry files
    if all_registry_files:
        master_index_path = os.path.join(registry_dir, "tiles_registry_index.txt")
        print(f"\nGenerating master tiles registry index: {master_index_path}")
        
        with open(master_index_path, 'w') as f:
            for registry_file in sorted(all_registry_files):
                filename = os.path.basename(registry_file)
                f.write(f"{filename}\n")
        
        print(f"Master tiles index written with {len(all_registry_files)} registry files")
        print(f"Total blocks processed: {total_blocks}")


def generate_master_registry(registry_dir):
    """Generate a master registry.txt file containing hashes of all registry files."""
    registry_files = []
    
    # Find all registry files (both representation and tile registry files)
    for file in os.listdir(registry_dir):
        if file.startswith("registry_") and file.endswith(".txt"):
            registry_files.append(file)
    
    if not registry_files:
        print("No registry files found to create master registry")
        return
    
    # Create the master registry.txt
    master_registry_path = os.path.join(registry_dir, "registry.txt")
    print(f"Generating master registry.txt: {master_registry_path}")
    
    with open(master_registry_path, 'w') as f:
        for filename in sorted(registry_files):
            file_path = os.path.join(registry_dir, filename)
            if os.path.exists(file_path):
                file_hash = calculate_sha256(file_path)
                f.write(f"{filename} {file_hash}\n")
    
    # Remove the old index files as they're now replaced by registry.txt
    old_index_files = [
        os.path.join(registry_dir, "registry_index.txt"),
        os.path.join(registry_dir, "tiles_registry_index.txt")
    ]
    for old_file in old_index_files:
        if os.path.exists(old_file):
            os.remove(old_file)
            print(f"Removed old index file: {os.path.basename(old_file)}")
    
    print(f"Master registry.txt created with {len(registry_files)} registry file entries")


def update_command(args):
    """Update registry files for both representation and tile data."""
    base_dir = os.path.abspath(args.base_dir)
    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} does not exist")
        return

    print(f"Scanning base directory: {base_dir}")
    
    # Look for both expected directories
    repr_dir = os.path.join(base_dir, "global_0.1_degree_representation")
    tiles_dir = os.path.join(base_dir, "global_0.1_degree_tiff_all")
    
    processed_any = False
    registry_dir = None
    
    # Process representations if directory exists
    if os.path.exists(repr_dir):
        print(f"\n{'='*60}")
        print("PROCESSING REPRESENTATIONS")
        print(f"{'='*60}")
        
        # Create a mock args object for the representations command
        repr_args = argparse.Namespace(
            base_dir=repr_dir,
            workers=args.workers,
            force=getattr(args, 'force', False),
            skip_checksum=getattr(args, 'skip_checksum', False),
            registry_dir=getattr(args, 'registry_dir', None)
        )
        update_representations_command(repr_args)
        processed_any = True
        
        # Get the registry directory path
        if hasattr(args, 'registry_dir') and args.registry_dir:
            registry_dir = os.path.join(os.path.abspath(args.registry_dir), "registry")
        else:
            parent_dir = os.path.dirname(repr_dir)
            registry_dir = os.path.join(parent_dir, "registry")
    else:
        print(f"Representations directory not found: {repr_dir}")
    
    # Process tiles if directory exists
    if os.path.exists(tiles_dir):
        print(f"\n{'='*60}")
        print("PROCESSING TILES")
        print(f"{'='*60}")
        
        # Create a mock args object for the tiles command
        tiles_args = argparse.Namespace(
            base_dir=tiles_dir,
            workers=args.workers,
            force=getattr(args, 'force', False),
            skip_checksum=getattr(args, 'skip_checksum', False),
            registry_dir=getattr(args, 'registry_dir', None)
        )
        update_tiles_command(tiles_args)
        processed_any = True
        
        # Get the registry directory path if not set from representations
        if not registry_dir:
            if hasattr(args, 'registry_dir') and args.registry_dir:
                registry_dir = os.path.join(os.path.abspath(args.registry_dir), "registry")
            else:
                parent_dir = os.path.dirname(tiles_dir)
                registry_dir = os.path.join(parent_dir, "registry")
    else:
        print(f"Tiles directory not found: {tiles_dir}")
    
    if not processed_any:
        print("No data directories found. Expected:")
        print(f"  - {repr_dir}")
        print(f"  - {tiles_dir}")
        return
    
    # Generate master registry.txt containing hashes of all registry files
    if registry_dir and os.path.exists(registry_dir):
        print(f"\n{'='*60}")
        print("GENERATING MASTER REGISTRY")
        print(f"{'='*60}")
        generate_master_registry(registry_dir)
    
    print(f"\n{'='*60}")
    print("REGISTRY UPDATE COMPLETE")
    print(f"{'='*60}")
    if os.path.exists(repr_dir):
        print(f"Representations: {repr_dir}")
    if os.path.exists(tiles_dir):
        print(f"Tiles: {tiles_dir}")
    if registry_dir:
        print(f"Registry: {registry_dir}")


def list_command(args):
    """List existing registry files in the specified directory."""
    base_dir = os.path.abspath(args.base_dir)
    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} does not exist")
        return

    print(f"Scanning for registry files in: {base_dir}")
    
    # Find all registry_*.txt files
    registry_files = []
    for file in os.listdir(base_dir):
        if file.startswith("registry_") and file.endswith(".txt"):
            registry_path = os.path.join(base_dir, file)
            # Count entries in the registry
            try:
                with open(registry_path, 'r') as f:
                    entry_count = sum(1 for line in f if line.strip() and not line.startswith('#'))
                registry_files.append((file, entry_count))
            except Exception as e:
                registry_files.append((file, -1))
    
    if not registry_files:
        print("No registry files found")
        return
    
    # Sort by filename
    registry_files.sort()
    
    print(f"\nFound {len(registry_files)} registry files:")
    for filename, count in registry_files:
        if count >= 0:
            print(f"  - {filename}: {count:,} entries")
        else:
            print(f"  - {filename}: (error reading file)")
    
    # Check for master registry
    master_registry = os.path.join(base_dir, "registry.txt")
    if os.path.exists(master_registry):
        print(f"\nMaster registry found: registry.txt")


def main():
    """Main entry point for the geotessera-registry CLI tool."""
    parser = argparse.ArgumentParser(
        description='GeoTessera Registry Management Tool - Generate and maintain Pooch registry files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update registry files for both representation and tile data (recommended)
  geotessera-registry update /path/to/data
  
  # Update with separate registry output directory
  geotessera-registry update /path/to/data --registry-dir /path/to/registry
  
  # Update with custom worker count
  geotessera-registry update /path/to/data --workers 8
  
  # Force checksum verification of all files (not just missing ones)
  geotessera-registry update /path/to/data --force
  
  # Skip checksum calculation for quick initialization
  geotessera-registry update /path/to/data --skip-checksum
  
  # Update representation data using block-based registries
  geotessera-registry update-representations /path/to/global_0.1_degree_representation
  
  # Update representation data with separate output directory
  geotessera-registry update-representations /path/to/data --registry-dir /path/to/registry
  
  # Update only tile data (flat structure) with force checking
  geotessera-registry update-tiles /path/to/global_0.1_degree_tiff_all --force
  
  # Quick scan without checksums for tiles
  geotessera-registry update-tiles /path/to/global_0.1_degree_tiff_all --skip-checksum
  
  # List existing registry files
  geotessera-registry list /path/to/data

This tool is intended for GeoTessera data maintainers to generate the registry
files that are distributed with the package. End users typically don't need
to use this tool.

Note: This tool creates block-based registries for efficient lazy loading:
  - Embeddings: Organized into 5x5 degree blocks (registry_YYYY_lonX_latY.txt)
  - Tiles: Organized into 5x5 degree blocks (registry_tiles_lonX_latY.txt)
  - Each block contains ~2,500 tiles instead of one massive registry
  - Registry files are created in subdirectories: registry/embeddings/ and registry/tiles/

Directory Structure:
  The 'update' command expects to find these subdirectories:
  - global_0.1_degree_representation/  (contains .npy files organized by year)
  - global_0.1_degree_tiff_all/        (contains .tiff files in flat structure)
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Update command (processes both directories)
    update_parser = subparsers.add_parser('update', 
                                         help='Update registry files for both representation and tile data')
    update_parser.add_argument('base_dir', help='Base directory containing global_0.1_degree_representation and global_0.1_degree_tiff_all subdirectories')
    update_parser.add_argument('--registry-dir', type=str, default=None,
                              help='Output directory for registry files (default: same as base_dir)')
    update_parser.add_argument('--workers', type=int, default=None,
                              help='Number of parallel workers (default: number of CPU cores)')
    update_parser.add_argument('--force', action='store_true',
                              help='Force checksum verification of all files, not just missing ones')
    update_parser.add_argument('--skip-checksum', action='store_true',
                              help='Skip checksum calculation for quick initialization')
    update_parser.set_defaults(func=update_command)
    
    # Update representations command
    update_repr_parser = subparsers.add_parser('update-representations', 
                                               help='Generate block-based registry files for representation data (5x5 degree blocks)')
    update_repr_parser.add_argument('base_dir', help='Base directory containing year subdirectories with .npy files')
    update_repr_parser.add_argument('--registry-dir', type=str, default=None,
                                    help='Output directory for registry files (default: same as base_dir)')
    update_repr_parser.add_argument('--workers', type=int, default=None,
                                    help='Number of parallel workers (default: number of CPU cores)')
    update_repr_parser.add_argument('--force', action='store_true',
                                    help='Force checksum verification of all files, not just missing ones')
    update_repr_parser.add_argument('--skip-checksum', action='store_true',
                                    help='Skip checksum calculation for quick initialization')
    update_repr_parser.set_defaults(func=update_representations_command)
    
    # Update tiles command
    update_tiles_parser = subparsers.add_parser('update-tiles',
                                                help='Generate or update registry file for tile data (.tiff files in flat structure)')
    update_tiles_parser.add_argument('base_dir', help='Base directory containing .tiff files')
    update_tiles_parser.add_argument('--registry-dir', type=str, default=None,
                                     help='Output directory for registry files (default: same as base_dir)')
    update_tiles_parser.add_argument('--workers', type=int, default=None,
                                     help='Number of parallel workers (default: number of CPU cores)')
    update_tiles_parser.add_argument('--force', action='store_true',
                                     help='Force checksum verification of all files, not just missing ones')
    update_tiles_parser.add_argument('--skip-checksum', action='store_true',
                                     help='Skip checksum calculation for quick initialization')
    update_tiles_parser.set_defaults(func=update_tiles_command)
    
    # List command
    list_parser = subparsers.add_parser('list', help='List existing registry files')
    list_parser.add_argument('base_dir', help='Base directory to scan for registry files')
    list_parser.set_defaults(func=list_command)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute the command
    args.func(args)


if __name__ == "__main__":
    main()