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


def calculate_sha256(file_path):
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def process_file(args):
    """Process a single file and return its relative path and hash."""
    file_path, base_dir = args
    try:
        rel_path = os.path.relpath(file_path, base_dir)
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


def find_npy_files(base_dir):
    """Find all .npy files and organize them by year."""
    files_by_year = defaultdict(list)

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.npy'):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, base_dir)

                # Extract year from path (assuming format ./YYYY/...)
                path_parts = rel_path.split(os.sep)
                if len(path_parts) > 0 and path_parts[0].isdigit() and len(path_parts[0]) == 4:
                    year = path_parts[0]
                    files_by_year[year].append(file_path)

    return files_by_year


def generate_master_registry(base_dir, registry_files):
    """Generate a master registry.txt file listing all registry files."""
    master_registry_path = os.path.join(base_dir, "registry.txt")
    
    print("\nGenerating master registry file...")
    
    # Sort registry files by year
    sorted_files = sorted(registry_files)
    
    with open(master_registry_path, 'w') as f:
        f.write("# GeoTessera Master Registry Index\n")
        f.write("# Lists all available year-specific registry files\n")
        f.write("# Format: registry_YYYY.txt\n\n")
        
        for registry_file in sorted_files:
            # Extract just the filename without path
            filename = os.path.basename(registry_file)
            f.write(f"{filename}\n")
    
    print(f"Master registry written to: {master_registry_path}")
    print(f"Listed {len(sorted_files)} registry files")


def update_command(args):
    """Update registry files for .npy files organized by year."""
    base_dir = os.path.abspath(args.base_dir)
    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} does not exist")
        return

    # Set number of workers
    num_workers = args.workers or multiprocessing.cpu_count()
    print(f"Using {num_workers} parallel workers")

    # Find all .npy files organized by year
    print("Scanning for .npy files...")
    files_by_year = find_npy_files(base_dir)

    if not files_by_year:
        print("No .npy files found")
        return

    print(f"Found files for years: {', '.join(sorted(files_by_year.keys()))}")

    registry_files = []

    # Process each year
    for year, year_files in sorted(files_by_year.items()):
        registry_file = os.path.join(base_dir, f"registry_{year}.txt")
        registry_files.append(registry_file)
        
        print(f"\nProcessing year {year}: {len(year_files)} files")

        # Load existing registry if incremental mode
        existing_registry = {}
        if args.incremental:
            existing_registry = load_existing_registry(registry_file)
            print(f"  Loaded {len(existing_registry)} existing entries")

        # Determine which files need processing
        files_to_process = []
        for file_path in year_files:
            rel_path = os.path.relpath(file_path, base_dir)
            if not args.incremental or rel_path not in existing_registry:
                files_to_process.append(file_path)

        if not files_to_process:
            print(f"  No new files to process for year {year}")
            continue

        print(f"  Processing {len(files_to_process)} files...")

        # Process files in parallel
        new_entries = {}
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(process_file, (file_path, base_dir)): file_path
                for file_path in files_to_process
            }

            # Process completed tasks
            completed = 0
            for future in as_completed(future_to_file):
                rel_path, file_hash = future.result()
                if rel_path and file_hash:
                    new_entries[rel_path] = file_hash

                completed += 1
                if completed % 1000 == 0:
                    print(f"    Processed {completed}/{len(files_to_process)} files...")

        # Merge with existing registry
        final_registry = existing_registry.copy()
        final_registry.update(new_entries)

        # Write registry file
        print(f"  Writing registry file: {registry_file}")
        with open(registry_file, 'w') as f:
            for rel_path in sorted(final_registry.keys()):
                f.write(f"{rel_path} {final_registry[rel_path]}\n")

        print(f"  Total entries in registry: {len(final_registry)}")
        if args.incremental:
            print(f"  New entries added: {len(new_entries)}")

    # Generate master registry if requested
    if args.generate_master and registry_files:
        generate_master_registry(base_dir, registry_files)


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
  # Generate/update registry files for all years
  geotessera-registry update /path/to/data
  
  # Update incrementally (only process new files)
  geotessera-registry update /path/to/data --incremental
  
  # Generate with custom worker count and create master registry
  geotessera-registry update /path/to/data --workers 8 --generate-master
  
  # List existing registry files
  geotessera-registry list /path/to/data

This tool is intended for GeoTessera data maintainers to generate the registry
files that are distributed with the package. End users typically don't need
to use this tool.
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Update command
    update_parser = subparsers.add_parser('update', help='Generate or update registry files')
    update_parser.add_argument('base_dir', help='Base directory containing year subdirectories')
    update_parser.add_argument('--workers', type=int, default=None,
                              help='Number of parallel workers (default: number of CPU cores)')
    update_parser.add_argument('--incremental', action='store_true',
                              help='Update existing registry files instead of regenerating')
    update_parser.add_argument('--generate-master', action='store_true',
                              help='Generate a master registry.txt file listing all registry files')
    update_parser.set_defaults(func=update_command)
    
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