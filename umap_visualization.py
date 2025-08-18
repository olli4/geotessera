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
from rasterio.transform import from_bounds
import umap
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings
import joblib

warnings.filterwarnings('ignore')


def get_file_hash(file_path: Path) -> str:
    """Generate a hash for a file to detect changes."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_checkpoint_metadata(checkpoint_dir: Path) -> dict:
    """Load checkpoint metadata if it exists."""
    metadata_path = checkpoint_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return {}


def save_checkpoint_metadata(checkpoint_dir: Path, metadata: dict):
    """Save checkpoint metadata."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = checkpoint_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def load_geotiffs(input_path: Path, sample_rate: float, checkpoint_dir: Path = None):
    """Load GeoTIFF files and sample pixels for UMAP processing."""
    print(f"Loading GeoTIFF files from {input_path}")
    
    if input_path.is_file():
        tiff_files = [input_path]
    else:
        tiff_files = sorted(list(input_path.glob("*.tif")) + list(input_path.glob("*.tiff")))
    
    if not tiff_files:
        raise ValueError(f"No GeoTIFF files found in {input_path}")
    
    print(f"Found {len(tiff_files)} GeoTIFF files")
    
    # Check if we have cached sampled data
    if checkpoint_dir:
        metadata = load_checkpoint_metadata(checkpoint_dir)
        sampled_data_path = checkpoint_dir / "sampled_data.npy"
        file_list_path = checkpoint_dir / "file_list.json"
        
        # Create a hash of input parameters
        input_hash = hashlib.md5(f"{str(input_path)}:{sample_rate}:{len(tiff_files)}".encode()).hexdigest()
        
        if (sampled_data_path.exists() and 
            file_list_path.exists() and
            metadata.get('sampling_complete') and
            metadata.get('input_hash') == input_hash and
            metadata.get('sample_rate') == sample_rate):
            
            print("Loading cached sampled data...")
            sampled_data = np.load(sampled_data_path)
            with open(file_list_path, 'r') as f:
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
            height, width = data.shape[1], data.shape[2]
            data_reshaped = data.transpose(1, 2, 0).reshape(-1, data.shape[0])
            
            # Remove NaN/invalid values
            valid_mask = ~np.isnan(data_reshaped).any(axis=1)
            valid_data = data_reshaped[valid_mask]
            
            if len(valid_data) == 0:
                continue
                
            # Sample pixels
            n_samples = int(len(valid_data) * sample_rate)
            if n_samples > 0:
                indices = np.random.choice(len(valid_data), size=n_samples, replace=False)
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
        with open(checkpoint_dir / "file_list.json", 'w') as f:
            json.dump([str(f) for f in tiff_files], f, indent=2)
        
        metadata = load_checkpoint_metadata(checkpoint_dir)
        metadata['sampling_complete'] = True
        metadata['input_hash'] = hashlib.md5(f"{str(input_path)}:{sample_rate}:{len(tiff_files)}".encode()).hexdigest()
        metadata['sample_rate'] = sample_rate
        metadata['num_files'] = len(tiff_files)
        metadata['sampled_pixels'] = len(combined_data)
        save_checkpoint_metadata(checkpoint_dir, metadata)
        print(f"Saved sampled data checkpoint to {checkpoint_dir}")
    
    return combined_data, datasets, tiff_files


def apply_umap_projection(data: np.ndarray, checkpoint_dir: Path = None, n_components: int = 3, random_state: int = 42):
    """Apply parametric UMAP dimensionality reduction to project data to RGB space."""
    
    # Check for cached UMAP model
    if checkpoint_dir:
        reducer_path = checkpoint_dir / "umap_reducer.pkl"
        scaler_path = checkpoint_dir / "scaler.pkl"
        embedding_path = checkpoint_dir / "embedding.npy"
        metadata = load_checkpoint_metadata(checkpoint_dir)
        
        if (reducer_path.exists() and 
            scaler_path.exists() and 
            embedding_path.exists() and
            metadata.get('umap_complete')):
            
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
        metric='euclidean',
        verbose=True,
        n_epochs=500
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
        metadata['umap_complete'] = True
        metadata['n_components'] = n_components
        metadata['random_state'] = random_state
        save_checkpoint_metadata(checkpoint_dir, metadata)
        print(f"Saved UMAP model checkpoint to {checkpoint_dir}")
    
    return embedding, reducer, scaler


def normalize_to_rgb(embedding: np.ndarray):
    """Normalize UMAP embedding to 0-255 RGB range."""
    # Normalize each component to 0-1 range
    rgb_normalized = np.zeros_like(embedding)
    for i in range(embedding.shape[1]):
        component = embedding[:, i]
        min_val, max_val = component.min(), component.max()
        if max_val > min_val:
            rgb_normalized[:, i] = (component - min_val) / (max_val - min_val)
        else:
            rgb_normalized[:, i] = 0.5  # Set to middle value if no variation
    
    # Scale to 0-255 and convert to uint8
    rgb_255 = (rgb_normalized * 255).astype(np.uint8)
    
    return rgb_255


def create_rgb_mosaic(tiff_files: list, reducer, scaler, output_path: Path, checkpoint_dir: Path = None):
    """Create RGB mosaic by applying UMAP projection to all pixels."""
    print("Creating RGB mosaic from all files")
    
    rgb_datasets = []
    target_crs = None
    
    # Create RGB cache directory
    rgb_cache_dir = None
    if checkpoint_dir:
        rgb_cache_dir = checkpoint_dir / "rgb_tiles"
        rgb_cache_dir.mkdir(parents=True, exist_ok=True)
    
    for tiff_file in tqdm(tiff_files, desc="Processing files for mosaic"):
        # Check if we have a cached RGB version
        rgb_cache_path = None
        if rgb_cache_dir:
            file_hash = get_file_hash(tiff_file)
            rgb_cache_path = rgb_cache_dir / f"rgb_{tiff_file.stem}_{file_hash[:8]}.tif"
            
            if rgb_cache_path.exists():
                rgb_datasets.append(rgb_cache_path)
                # Get CRS from cached file if needed
                if target_crs is None:
                    with rasterio.open(rgb_cache_path) as src:
                        target_crs = src.crs
                continue
        
        with rasterio.open(tiff_file) as src:
            # Store the first CRS as the target
            if target_crs is None:
                target_crs = src.crs
            
            data = src.read()  # Shape: (bands, height, width)
            height, width = data.shape[1], data.shape[2]
            
            # Reshape for processing
            data_reshaped = data.transpose(1, 2, 0).reshape(-1, data.shape[0])
            
            # Handle NaN values
            valid_mask = ~np.isnan(data_reshaped).any(axis=1)
            
            # Apply preprocessing and UMAP transformation
            rgb_data = np.zeros((len(data_reshaped), 3), dtype=np.uint8)
            
            if np.any(valid_mask):
                valid_data = data_reshaped[valid_mask]
                
                # Apply same preprocessing as training
                valid_scaled = scaler.transform(valid_data)
                
                # Apply UMAP transformation
                valid_embedding = reducer.transform(valid_scaled)
                
                # Normalize to RGB
                valid_rgb = normalize_to_rgb(valid_embedding)
                
                rgb_data[valid_mask] = valid_rgb
            
            # Reshape back to image format
            rgb_image = rgb_data.reshape(height, width, 3).transpose(2, 0, 1)
            
            # Create RGB dataset profile
            rgb_profile = src.profile.copy()
            rgb_profile.update({
                'count': 3,
                'dtype': 'uint8',
                'nodata': 0,
                'crs': target_crs  # Ensure all files use the same CRS
            })
            
            # Write to cache or temp file
            if rgb_cache_path:
                output_rgb_path = rgb_cache_path
            else:
                output_rgb_path = output_path.parent / f"temp_rgb_{tiff_file.stem}.tif"
            
            with rasterio.open(output_rgb_path, 'w', **rgb_profile) as dst:
                dst.write(rgb_image)
            
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
    out_profile.update({
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "count": 3,
        "dtype": 'uint8'
    })
    
    # Write final mosaic
    with rasterio.open(output_path, 'w', **out_profile) as dest:
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
        metadata['mosaic_complete'] = True
        metadata['output_path'] = str(output_path)
        save_checkpoint_metadata(checkpoint_dir, metadata)
    
    print(f"RGB mosaic saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create RGB visualizations from GeoTIFF files using UMAP"
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Input directory containing GeoTIFF files or single GeoTIFF file"
    )
    parser.add_argument(
        "output_path",
        type=Path,
        help="Output path for RGB visualization (e.g., output.tif)"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory for storing checkpoints (delete to reset)"
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=0.05,
        help="Percentage of pixels to sample for UMAP training (default: 0.05)"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear checkpoint cache before running"
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
            if metadata.get('sampling_complete'):
                print(f"  ✓ Sampling complete ({metadata.get('sampled_pixels', 0)} pixels)")
            if metadata.get('umap_complete'):
                print(f"  ✓ UMAP training complete")
            if metadata.get('mosaic_complete'):
                print(f"  ✓ Previous mosaic complete")
    
    # Create output directory if needed
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Set random seed
        np.random.seed(args.random_seed)
        
        # Load and sample data
        sampled_data, datasets, tiff_files = load_geotiffs(
            args.input_path, 
            args.sample_rate,
            args.checkpoint_dir
        )
        
        # Apply UMAP projection
        embedding, reducer, scaler = apply_umap_projection(
            sampled_data, 
            args.checkpoint_dir,
            random_state=args.random_seed
        )
        
        # Create RGB mosaic
        create_rgb_mosaic(
            tiff_files, 
            reducer, 
            scaler, 
            args.output_path,
            args.checkpoint_dir
        )
        
        print(f"Successfully created RGB visualization: {args.output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
