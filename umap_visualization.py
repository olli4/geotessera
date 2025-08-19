# /// script
# dependencies = [
#     "rasterio",
#     "numpy",
#     "umap-learn",
#     "scikit-learn",
#     "tqdm",
#     "tensorflow-macos; sys_platform == 'darwin'",
#     "tensorflow-metal; sys_platform == 'darwin'",
#     "tensorflow; sys_platform != 'darwin'",
#     "joblib",
# ]
# ///
"""
Standalone script for creating RGB visualizations from GeoTIFF files using UMAP.

This script:
1. Loads multiple GeoTIFF files and samples a percentage of pixels
2. Uses UMAP to project high-dimensional data to 3D RGB space
3. Normalizes values to 0-255 range
4. Creates a mosaic and outputs the final RGB visualization
5. Supports checkpointing for resuming interrupted runs

Usage:
    uv run umap_visualization.py input_dir/ output.tif --checkpoint-dir cache/ [--sample-rate 0.05]
"""

import argparse
import sys
import json
import hashlib
from pathlib import Path
import numpy as np
import rasterio
from rasterio.merge import merge
import umap
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings
import joblib

warnings.filterwarnings("ignore")


def get_file_hash(file_path: Path) -> str:
    """Generate a hash for a file to detect changes."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_checkpoint_metadata(checkpoint_dir: Path) -> dict:
    """Load checkpoint metadata if it exists."""
    metadata_path = checkpoint_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            return json.load(f)
    return {}


def save_checkpoint_metadata(checkpoint_dir: Path, metadata: dict):
    """Save checkpoint metadata."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = checkpoint_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def load_geotiffs(input_path: Path, sample_rate: float, checkpoint_dir: Path = None):
    """Load GeoTIFF files and sample pixels for UMAP processing."""
    print(f"Loading GeoTIFF files from {input_path}")

    if input_path.is_file():
        tiff_files = [input_path]
    else:
        tiff_files = sorted(
            list(input_path.glob("*.tif")) + list(input_path.glob("*.tiff"))
        )

    if not tiff_files:
        raise ValueError(f"No GeoTIFF files found in {input_path}")

    print(f"Found {len(tiff_files)} GeoTIFF files")

    # Check if we have cached sampled data
    if checkpoint_dir:
        metadata = load_checkpoint_metadata(checkpoint_dir)
        sampled_data_path = checkpoint_dir / "sampled_data.npy"
        file_list_path = checkpoint_dir / "file_list.json"

        # Create a hash of input parameters
        input_hash = hashlib.md5(
            f"{str(input_path)}:{sample_rate}:{len(tiff_files)}".encode()
        ).hexdigest()

        if (
            sampled_data_path.exists()
            and file_list_path.exists()
            and metadata.get("sampling_complete")
            and metadata.get("input_hash") == input_hash
            and metadata.get("sample_rate") == sample_rate
        ):
            print("Loading cached sampled data...")
            sampled_data = np.load(sampled_data_path)
            with open(file_list_path, "r") as f:
                cached_files = [Path(p) for p in json.load(f)]
            print(f"Loaded cached data: {sampled_data.shape}")
            return sampled_data, None, cached_files

    all_data = []
    datasets = []

    for tiff_file in tqdm(tiff_files, desc="Loading files"):
        with rasterio.open(tiff_file) as src:
            datasets.append(src)
            data = src.read()  # Shape: (bands, height, width)

            # Reshape to (height*width, bands)
            data_reshaped = data.transpose(1, 2, 0).reshape(-1, data.shape[0])

            # Remove NaN/invalid values
            valid_mask = ~np.isnan(data_reshaped).any(axis=1)
            valid_data = data_reshaped[valid_mask]

            if len(valid_data) == 0:
                continue

            # Sample pixels
            n_samples = int(len(valid_data) * sample_rate)
            if n_samples > 0:
                indices = np.random.choice(
                    len(valid_data), size=n_samples, replace=False
                )
                sampled_data = valid_data[indices]
                all_data.append(sampled_data)

    if not all_data:
        raise ValueError("No valid data found in GeoTIFF files")

    # Combine all sampled data
    combined_data = np.vstack(all_data)
    print(f"Sampled {len(combined_data)} pixels from {len(all_data)} files")
    print(f"Data shape: {combined_data.shape}")

    # Save checkpoint
    if checkpoint_dir:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        np.save(checkpoint_dir / "sampled_data.npy", combined_data)
        with open(checkpoint_dir / "file_list.json", "w") as f:
            json.dump([str(f) for f in tiff_files], f, indent=2)

        metadata = load_checkpoint_metadata(checkpoint_dir)
        metadata["sampling_complete"] = True
        metadata["input_hash"] = hashlib.md5(
            f"{str(input_path)}:{sample_rate}:{len(tiff_files)}".encode()
        ).hexdigest()
        metadata["sample_rate"] = sample_rate
        metadata["num_files"] = len(tiff_files)
        metadata["sampled_pixels"] = len(combined_data)
        save_checkpoint_metadata(checkpoint_dir, metadata)
        print(f"Saved sampled data checkpoint to {checkpoint_dir}")

    return combined_data, datasets, tiff_files


def apply_umap_projection(
    data: np.ndarray,
    checkpoint_dir: Path = None,
    n_components: int = 3,
    random_state: int = 42,
):
    """Apply parametric UMAP dimensionality reduction to project data to RGB space."""

    # Check for cached UMAP model
    if checkpoint_dir:
        reducer_path = checkpoint_dir / "umap_reducer.pkl"
        scaler_path = checkpoint_dir / "scaler.pkl"
        embedding_path = checkpoint_dir / "embedding.npy"
        metadata = load_checkpoint_metadata(checkpoint_dir)

        if (
            reducer_path.exists()
            and scaler_path.exists()
            and embedding_path.exists()
            and metadata.get("umap_complete")
        ):
            print("Loading cached UMAP model and embedding...")
            reducer = joblib.load(reducer_path)
            scaler = joblib.load(scaler_path)
            embedding = np.load(embedding_path)
            print(f"Loaded UMAP embedding shape: {embedding.shape}")
            return embedding, reducer, scaler

    print(f"Applying parametric UMAP projection to {n_components} dimensions")

    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Apply parametric UMAP with correct parameters
    reducer = umap.ParametricUMAP(
        n_components=n_components,
        random_state=random_state,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        verbose=True,
        n_epochs=500,
    )

    embedding = reducer.fit_transform(data_scaled)

    print(f"Parametric UMAP embedding shape: {embedding.shape}")
    print(f"Embedding range: [{embedding.min():.3f}, {embedding.max():.3f}]")

    # Save checkpoint
    if checkpoint_dir:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(reducer, checkpoint_dir / "umap_reducer.pkl")
        joblib.dump(scaler, checkpoint_dir / "scaler.pkl")
        np.save(checkpoint_dir / "embedding.npy", embedding)

        metadata = load_checkpoint_metadata(checkpoint_dir)
        metadata["umap_complete"] = True
        metadata["n_components"] = n_components
        metadata["random_state"] = random_state
        save_checkpoint_metadata(checkpoint_dir, metadata)
        print(f"Saved UMAP model checkpoint to {checkpoint_dir}")

    return embedding, reducer, scaler


def normalize_to_rgb(embedding: np.ndarray):
    """Normalize UMAP embedding to 0-255 RGB range."""
    # Use percentile-based normalization for better color distribution
    # This prevents extreme values from dominating the color mapping
    percentile_low = 2
    percentile_high = 98

    rgb_normalized = np.zeros_like(embedding)

    for i in range(embedding.shape[1]):
        component = embedding[:, i]
        # Use percentiles to clip extreme values
        p_low = np.percentile(component, percentile_low)
        p_high = np.percentile(component, percentile_high)

        # Clip and normalize
        component_clipped = np.clip(component, p_low, p_high)

        if p_high > p_low:
            rgb_normalized[:, i] = (component_clipped - p_low) / (p_high - p_low)
        else:
            rgb_normalized[:, i] = 0.5

    # Apply a slight contrast enhancement to make colors more vivid
    # Using a sigmoid-like transformation
    rgb_enhanced = rgb_normalized
    rgb_enhanced = np.clip(rgb_enhanced * 1.2 - 0.1, 0, 1)  # Boost contrast slightly

    # Scale to 0-255 and convert to uint8
    rgb_255 = (rgb_enhanced * 255).astype(np.uint8)

    # Print statistics for debugging
    print("RGB channel statistics after normalization:")
    print(
        f"  R: min={rgb_255[:, 0].min()}, max={rgb_255[:, 0].max()}, mean={rgb_255[:, 0].mean():.1f}"
    )
    print(
        f"  G: min={rgb_255[:, 1].min()}, max={rgb_255[:, 1].max()}, mean={rgb_255[:, 1].mean():.1f}"
    )
    print(
        f"  B: min={rgb_255[:, 2].min()}, max={rgb_255[:, 2].max()}, mean={rgb_255[:, 2].mean():.1f}"
    )

    return rgb_255


def normalize_to_rgb_global(embedding: np.ndarray, global_norm_params: list):
    """Normalize UMAP embedding to 0-255 RGB range using global parameters."""
    rgb_normalized = np.zeros_like(embedding)

    for i in range(embedding.shape[1]):
        component = embedding[:, i]
        p_low, p_high = global_norm_params[i]

        # Clip and normalize using global parameters
        component_clipped = np.clip(component, p_low, p_high)

        if p_high > p_low:
            rgb_normalized[:, i] = (component_clipped - p_low) / (p_high - p_low)
        else:
            rgb_normalized[:, i] = 0.5

    # Apply a slight contrast enhancement to make colors more vivid
    rgb_enhanced = rgb_normalized
    rgb_enhanced = np.clip(rgb_enhanced * 1.2 - 0.1, 0, 1)  # Boost contrast slightly

    # Scale to 0-255 and convert to uint8
    rgb_255 = (rgb_enhanced * 255).astype(np.uint8)

    return rgb_255


def create_rgb_mosaic(
    tiff_files: list, reducer, scaler, output_path: Path, checkpoint_dir: Path = None
):
    """Create RGB mosaic by applying UMAP projection to all pixels."""
    print("Creating RGB mosaic from all files")

    rgb_datasets = []
    # Use Web Mercator (EPSG:3857) as the common CRS for merging
    # This ensures all tiles can be properly aligned and matches web mapping standards
    target_crs = rasterio.crs.CRS.from_epsg(3857)

    # Create RGB cache directory
    rgb_cache_dir = None
    if checkpoint_dir:
        rgb_cache_dir = checkpoint_dir / "rgb_tiles"
        rgb_cache_dir.mkdir(parents=True, exist_ok=True)

    # First pass: collect all embeddings to compute global statistics
    print("First pass: collecting all embeddings for global normalization")
    all_embeddings = []
    tile_embeddings = []  # Store embeddings per tile for second pass
    tile_metadata = []    # Store metadata per tile

    for tiff_file in tqdm(tiff_files, desc="Collecting embeddings"):
        # Check if we have a cached RGB version
        rgb_cache_path = None
        if rgb_cache_dir:
            file_hash = get_file_hash(tiff_file)
            rgb_cache_path = rgb_cache_dir / f"rgb_{tiff_file.stem}_{file_hash[:8]}.tif"

            if rgb_cache_path.exists():
                rgb_datasets.append(rgb_cache_path)
                tile_embeddings.append(None)  # Skip embedding collection for cached tiles
                tile_metadata.append(None)
                continue

        with rasterio.open(tiff_file) as src:
            data = src.read()  # Shape: (bands, height, width)
            height, width = data.shape[1], data.shape[2]

            # Reshape for processing
            data_reshaped = data.transpose(1, 2, 0).reshape(-1, data.shape[0])

            # Handle NaN values
            valid_mask = ~np.isnan(data_reshaped).any(axis=1)

            tile_embedding = None
            if np.any(valid_mask):
                valid_data = data_reshaped[valid_mask]

                # Apply same preprocessing as training
                valid_scaled = scaler.transform(valid_data)

                # Apply UMAP transformation
                valid_embedding = reducer.transform(valid_scaled)
                
                # Store embedding for this tile
                tile_embedding = valid_embedding
                all_embeddings.append(valid_embedding)

            # Store tile data for second pass
            tile_embeddings.append(tile_embedding)
            tile_metadata.append({
                'data_shape': (height, width),
                'valid_mask': valid_mask,
                'file': tiff_file,
                'cache_path': rgb_cache_path
            })

    # Compute global normalization parameters from all embeddings
    if all_embeddings:
        print("Computing global normalization parameters")
        combined_embeddings = np.vstack(all_embeddings)
        
        # Use percentile-based normalization for better color distribution
        percentile_low = 2
        percentile_high = 98
        
        global_norm_params = []
        for i in range(combined_embeddings.shape[1]):
            component = combined_embeddings[:, i]
            p_low = np.percentile(component, percentile_low)
            p_high = np.percentile(component, percentile_high)
            global_norm_params.append((p_low, p_high))
        
        print(f"Global normalization parameters computed from {len(combined_embeddings)} pixels across {len(all_embeddings)} tiles")
    else:
        global_norm_params = [(0, 1)] * 3  # Default if no embeddings

    # Second pass: apply global normalization and create RGB tiles
    print("Second pass: applying global normalization and creating tiles")
    for tile_embedding, metadata in zip(tile_embeddings, tile_metadata):
        if tile_embedding is None or metadata is None:
            continue  # Skip cached tiles
            
        tiff_file = metadata['file']
        height, width = metadata['data_shape']
        valid_mask = metadata['valid_mask']
        rgb_cache_path = metadata['cache_path']

        with rasterio.open(tiff_file) as src:
            from rasterio.warp import calculate_default_transform, reproject, Resampling

            # Apply global normalization to RGB
            rgb_data = np.zeros((len(valid_mask), 3), dtype=np.uint8)

            if tile_embedding is not None:
                # Apply global normalization
                valid_rgb = normalize_to_rgb_global(tile_embedding, global_norm_params)
                rgb_data[valid_mask] = valid_rgb

            # Reshape back to image format
            rgb_image = rgb_data.reshape(height, width, 3).transpose(2, 0, 1)

            # Calculate target transform for reprojection to Web Mercator
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds
            )

            # Create reprojected RGB dataset
            rgb_profile = {
                "driver": "GTiff",
                "height": dst_height,
                "width": dst_width,
                "count": 3,
                "dtype": "uint8",
                "crs": target_crs,
                "transform": dst_transform,
                "compress": "lzw",
                "nodata": 0,
            }

            # Write to cache or temp file
            if rgb_cache_path:
                output_rgb_path = rgb_cache_path
            else:
                output_rgb_path = output_path.parent / f"temp_rgb_{tiff_file.stem}.tif"

            # Write and reproject RGB image to Web Mercator
            with rasterio.open(output_rgb_path, "w", **rgb_profile) as dst:
                # Reproject each RGB band
                for i in range(3):
                    reproject(
                        source=rgb_image[i],
                        destination=rasterio.band(dst, i + 1),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=dst_transform,
                        dst_crs=target_crs,
                        resampling=Resampling.bilinear,
                    )

            rgb_datasets.append(output_rgb_path)

    # Merge all RGB datasets
    print("Merging RGB datasets into final mosaic")

    src_files_to_mosaic = []
    for rgb_file in rgb_datasets:
        src = rasterio.open(rgb_file)
        src_files_to_mosaic.append(src)

    mosaic, out_trans = merge(src_files_to_mosaic)

    # Update profile for output
    out_profile = src_files_to_mosaic[0].profile.copy()
    out_profile.update(
        {
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
            "count": 3,
            "dtype": "uint8",
        }
    )

    # Write final mosaic
    with rasterio.open(output_path, "w", **out_profile) as dest:
        dest.write(mosaic)

    # Clean up
    for src in src_files_to_mosaic:
        src.close()

    # Clean up temporary files (but not cached ones)
    if not rgb_cache_dir:
        for temp_file in rgb_datasets:
            if temp_file.name.startswith("temp_rgb_"):
                temp_file.unlink()

    # Update metadata
    if checkpoint_dir:
        metadata = load_checkpoint_metadata(checkpoint_dir)
        metadata["mosaic_complete"] = True
        metadata["output_path"] = str(output_path)
        save_checkpoint_metadata(checkpoint_dir, metadata)

    print(f"RGB mosaic saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create RGB visualizations from GeoTIFF files using UMAP"
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Input directory containing GeoTIFF files or single GeoTIFF file",
    )
    parser.add_argument(
        "output_path",
        type=Path,
        help="Output path for RGB visualization (e.g., output.tif)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory for storing checkpoints (delete to reset)",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=0.05,
        help="Percentage of pixels to sample for UMAP training (default: 0.05)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear checkpoint cache before running",
    )

    args = parser.parse_args()

    if not args.input_path.exists():
        print(f"Error: Input path {args.input_path} does not exist")
        sys.exit(1)

    # Handle checkpoint directory
    if args.checkpoint_dir:
        if args.clear_cache and args.checkpoint_dir.exists():
            print(f"Clearing checkpoint directory: {args.checkpoint_dir}")
            import shutil

            shutil.rmtree(args.checkpoint_dir)

        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"Using checkpoint directory: {args.checkpoint_dir}")

        # Show current checkpoint status
        metadata = load_checkpoint_metadata(args.checkpoint_dir)
        if metadata:
            print("Checkpoint status:")
            if metadata.get("sampling_complete"):
                print(
                    f"  ✓ Sampling complete ({metadata.get('sampled_pixels', 0)} pixels)"
                )
            if metadata.get("umap_complete"):
                print("  ✓ UMAP training complete")
            if metadata.get("mosaic_complete"):
                print("  ✓ Previous mosaic complete")

    # Create output directory if needed
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Set random seed
        np.random.seed(args.random_seed)

        # Load and sample data
        sampled_data, datasets, tiff_files = load_geotiffs(
            args.input_path, args.sample_rate, args.checkpoint_dir
        )

        # Apply UMAP projection
        embedding, reducer, scaler = apply_umap_projection(
            sampled_data, args.checkpoint_dir, random_state=args.random_seed
        )

        # Create RGB mosaic
        create_rgb_mosaic(
            tiff_files, reducer, scaler, args.output_path, args.checkpoint_dir
        )

        print(f"Successfully created RGB visualization: {args.output_path}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
