#!/usr/bin/env python3

"""
Standalone script for creating RGB visualizations from GeoTIFF files using UMAP.

This script:
1. Loads multiple GeoTIFF files and samples a percentage of pixels
2. Uses UMAP to project high-dimensional data to 3D RGB space
3. Normalizes values to 0-255 range
4. Creates a mosaic and outputs the final RGB visualization

Usage:
    uv run umap_visualization.py input_dir/ output.tif [--sample-rate 0.05]
"""

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
# ]
# ///

import argparse
import sys
from pathlib import Path
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.transform import from_bounds
import umap
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


def load_geotiffs(input_path: Path, sample_rate: float = 0.05):
    """Load GeoTIFF files and sample pixels for UMAP processing."""
    print(f"Loading GeoTIFF files from {input_path}")
    
    if input_path.is_file():
        tiff_files = [input_path]
    else:
        tiff_files = list(input_path.glob("*.tif")) + list(input_path.glob("*.tiff"))
    
    if not tiff_files:
        raise ValueError(f"No GeoTIFF files found in {input_path}")
    
    print(f"Found {len(tiff_files)} GeoTIFF files")
    
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
    
    return combined_data, datasets, tiff_files


def apply_umap_projection(data: np.ndarray, n_components: int = 3, random_state: int = 42):
    """Apply parametric UMAP dimensionality reduction to project data to RGB space."""
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
        n_epochs=100
    )
    
    embedding = reducer.fit_transform(data_scaled)
    
    print(f"Parametric UMAP embedding shape: {embedding.shape}")
    print(f"Embedding range: [{embedding.min():.3f}, {embedding.max():.3f}]")
    
    return embedding, reducer, scaler


def normalize_to_rgb(embedding: np.ndarray):
    """Normalize UMAP embedding to 0-255 RGB range."""
    print("Normalizing to RGB range (0-255)")
    
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
    
    print(f"RGB range: [{rgb_255.min()}, {rgb_255.max()}]")
    
    return rgb_255


def create_rgb_mosaic(tiff_files: list, reducer, scaler, output_path: Path):
    """Create RGB mosaic by applying UMAP projection to all pixels."""
    print("Creating RGB mosaic from all files")
    
    rgb_datasets = []
    
    for tiff_file in tqdm(tiff_files, desc="Processing files for mosaic"):
        with rasterio.open(tiff_file) as src:
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
            
            # Create temporary RGB dataset
            rgb_profile = src.profile.copy()
            rgb_profile.update({
                'count': 3,
                'dtype': 'uint8',
                'nodata': 0
            })
            
            # Write to temporary file
            temp_rgb_path = output_path.parent / f"temp_rgb_{tiff_file.stem}.tif"
            
            with rasterio.open(temp_rgb_path, 'w', **rgb_profile) as dst:
                dst.write(rgb_image)
            
            rgb_datasets.append(temp_rgb_path)
    
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
    
    # Clean up temporary files
    for src in src_files_to_mosaic:
        src.close()
    
    for temp_file in rgb_datasets:
        temp_file.unlink()
    
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
    
    args = parser.parse_args()
    
    if not args.input_path.exists():
        print(f"Error: Input path {args.input_path} does not exist")
        sys.exit(1)
    
    # Create output directory if needed
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Set random seed
        np.random.seed(args.random_seed)
        
        # Load and sample data
        sampled_data, datasets, tiff_files = load_geotiffs(args.input_path, args.sample_rate)
        
        # Apply UMAP projection
        embedding, reducer, scaler = apply_umap_projection(sampled_data, random_state=args.random_seed)
        
        # Create RGB mosaic
        create_rgb_mosaic(tiff_files, reducer, scaler, args.output_path)
        
        print(f"Successfully created RGB visualization: {args.output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()