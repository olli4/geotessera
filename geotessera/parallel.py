"""Parallel processing utilities for efficient tile processing.

This module provides utilities to process multiple tiles in parallel,
which is essential for working with large regions. It includes both
generic parallel mapping functions and specialized functions for
common classification workflows.
"""

from pathlib import Path
from typing import List, Tuple, Callable, Any, Optional, Union
import concurrent.futures
import numpy as np
from tqdm import tqdm


def map_over_tiles(
    tiles: List[Tuple[float, float]],
    func: Callable[[float, float], Any],
    max_workers: Optional[int] = None,
    desc: str = "Processing tiles",
    show_progress: bool = True,
) -> List[Any]:
    """Apply a function to multiple tiles in parallel.

    This is a generic parallel mapping function that processes tiles
    concurrently. The function should take (lat, lon) as arguments
    and return the processing result.

    Args:
        tiles: List of (lat, lon) tuples
        func: Function that takes (lat, lon) and returns a result
        max_workers: Maximum number of parallel workers. If None, uses
                    CPU count. Set to 1 for sequential processing.
        desc: Description for progress bar
        show_progress: Whether to show progress bar

    Returns:
        List of results in the same order as input tiles

    Examples:
        >>> # Example: Extract mean values from tiles
        >>> def get_tile_mean(lat, lon):
        ...     embedding = gt.fetch_embedding(lat, lon, year=2024)
        ...     return np.mean(embedding)
        ...
        >>> tiles = [(52.1, 0.1), (52.2, 0.1), (52.3, 0.1)]
        >>> means = map_over_tiles(tiles, get_tile_mean, max_workers=4)
        >>> print(f"Processed {len(means)} tiles")

        >>> # Example: Classify tiles with a trained model
        >>> def classify_tile(lat, lon):
        ...     embedding = gt.fetch_embedding(lat, lon, year=2024)
        ...     h, w, c = embedding.shape
        ...     predictions = model.predict(embedding.reshape(-1, c))
        ...     return predictions.reshape(h, w)
        ...
        >>> classifications = map_over_tiles(tiles, classify_tile)
    """
    if not tiles:
        return []

    # Handle sequential processing
    if max_workers == 1:
        results = []
        iterator = tqdm(tiles, desc=desc) if show_progress else tiles
        for lat, lon in iterator:
            try:
                result = func(lat, lon)
                results.append(result)
            except Exception as e:
                print(f"\\nError processing tile ({lat:.2f}, {lon:.2f}): {e}")
                results.append(None)
        return results

    # Parallel processing
    results = [None] * len(tiles)

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(func, lat, lon): i for i, (lat, lon) in enumerate(tiles)
        }

        # Process completed tasks
        iterator = concurrent.futures.as_completed(future_to_idx)
        if show_progress:
            iterator = tqdm(iterator, total=len(tiles), desc=desc)

        for future in iterator:
            idx = future_to_idx[future]
            lat, lon = tiles[idx]

            try:
                result = future.result()
                results[idx] = result
            except Exception as e:
                print(f"\\nError processing tile ({lat:.2f}, {lon:.2f}): {e}")
                results[idx] = None

    return results


def process_tiles_to_dir(
    gt,
    tiles: List[Tuple[float, float]],
    output_dir: Union[str, Path],
    process_func: Callable[[np.ndarray], np.ndarray],
    year: int = 2024,
    max_workers: Optional[int] = None,
    output_format: str = "GTiff",
    output_dtype: str = "uint8",
    compress: str = "lzw",
    show_progress: bool = True,
) -> List[Path]:
    """Process tiles and save results as georeferenced rasters.

    This function handles the common pattern of:
    1. Loading embeddings for each tile
    2. Processing them with a custom function
    3. Saving results as georeferenced rasters

    Args:
        gt: GeoTessera instance for fetching embeddings
        tiles: List of (lat, lon) tuples
        output_dir: Directory to save processed tiles
        process_func: Function that takes embeddings (H, W, 128) and returns
                     processed array (H, W) or (H, W, C)
        year: Year of embeddings to use
        max_workers: Maximum number of parallel workers
        output_format: Rasterio driver name (default 'GTiff')
        output_dtype: Output data type (default 'uint8')
        compress: Compression method (default 'lzw')
        show_progress: Whether to show progress bar

    Returns:
        List of paths to created files (None for failed tiles)

    Examples:
        >>> # Classification example
        >>> def classify(embedding):
        ...     h, w, c = embedding.shape
        ...     return model.predict(embedding.reshape(-1, c)).reshape(h, w)
        ...
        >>> output_files = process_tiles_to_dir(
        ...     gt, tiles, "classified_tiles/", classify,
        ...     year=2024, max_workers=8
        ... )

        >>> # Band extraction example
        >>> def extract_bands(embedding):
        ...     return embedding[:, :, [10, 20, 30]]  # Extract 3 bands
        ...
        >>> rgb_files = process_tiles_to_dir(
        ...     gt, tiles, "rgb_tiles/", extract_bands,
        ...     output_dtype='float32'
        ... )
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define the worker function
    def process_single_tile(args):
        lat, lon, year, output_dir, process_func = args

        try:
            # Import here to avoid serialization issues
            from geotessera import GeoTessera
            import rasterio
            import numpy as np

            # Create new GT instance in worker process
            gt_worker = GeoTessera()

            # Fetch embedding
            embedding = gt_worker.fetch_embedding(lat, lon, year, progressbar=False)

            # Process the embedding
            processed = process_func(embedding)

            # Ensure proper shape
            if processed.ndim == 2:
                processed = processed[np.newaxis, :, :]  # Add band dimension
            elif processed.ndim == 3:
                # Move band dimension to first position if needed
                if processed.shape[2] < processed.shape[0]:
                    processed = np.transpose(processed, (2, 0, 1))

            # Get georeferencing info
            crs = gt_worker.get_tile_crs(lat, lon)
            transform = gt_worker.get_tile_transform(lat, lon)

            # Save the result
            output_path = output_dir / f"processed_{lat:.2f}_{lon:.2f}.tif"

            with rasterio.open(
                output_path,
                "w",
                driver=output_format,
                height=processed.shape[1],
                width=processed.shape[2],
                count=processed.shape[0],
                dtype=output_dtype,
                crs=crs,
                transform=transform,
                compress=compress,
            ) as dst:
                dst.write(processed)

            return output_path

        except Exception as e:
            print(f"\\nError processing tile ({lat:.2f}, {lon:.2f}): {e}")
            return None

    # Prepare arguments for workers
    tasks = [(lat, lon, year, output_dir, process_func) for lat, lon in tiles]

    # Process in parallel
    if max_workers == 1:
        # Sequential processing
        results = []
        iterator = tqdm(tasks, desc="Processing tiles") if show_progress else tasks
        for task in iterator:
            result = process_single_tile(task)
            results.append(result)
    else:
        # Parallel processing
        results = [None] * len(tasks)

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        ) as executor:
            future_to_idx = {
                executor.submit(process_single_tile, task): i
                for i, task in enumerate(tasks)
            }

            iterator = concurrent.futures.as_completed(future_to_idx)
            if show_progress:
                iterator = tqdm(iterator, total=len(tasks), desc="Processing tiles")

            for future in iterator:
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    results[idx] = result
                except Exception as e:
                    lat, lon = tiles[idx]
                    print(f"\\nError with tile ({lat:.2f}, {lon:.2f}): {e}")
                    results[idx] = None

    # Filter out None values and return paths
    return [r for r in results if r is not None]


def classify_tiles_batch(
    gt,
    tiles: List[Tuple[float, float]],
    model: Any,
    output_dir: Union[str, Path],
    year: int = 2024,
    max_workers: Optional[int] = None,
    batch_size: int = 1000,
    show_progress: bool = True,
) -> List[Path]:
    """Specialized function for classifying tiles with a scikit-learn style model.

    This function is optimized for classification tasks and handles:
    - Efficient batching for model predictions
    - Automatic dtype selection based on number of classes
    - Progress tracking

    Args:
        gt: GeoTessera instance
        tiles: List of (lat, lon) tuples
        model: Trained model with predict() method
        output_dir: Directory to save classified tiles
        year: Year of embeddings
        max_workers: Maximum parallel workers
        batch_size: Batch size for model predictions
        show_progress: Whether to show progress

    Returns:
        List of paths to classified tiles

    Examples:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> model = RandomForestClassifier()
        >>> # ... train model ...
        >>>
        >>> classified = classify_tiles_batch(
        ...     gt, tiles, model, "classified/",
        ...     max_workers=8
        ... )
    """
    import pickle
    import tempfile

    # Serialize model for workers
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
        pickle.dump(model, f)
        model_path = f.name

    try:

        def classify_embedding(embedding):
            # Load model in worker
            with open(model_path, "rb") as f:
                model = pickle.load(f)

            h, w, c = embedding.shape
            pixels = embedding.reshape(-1, c)

            # Predict in batches
            predictions = []
            for i in range(0, len(pixels), batch_size):
                batch = pixels[i : i + batch_size]
                pred = model.predict(batch)
                predictions.append(pred)

            predictions = np.concatenate(predictions)
            return predictions.reshape(h, w)

        # Determine output dtype based on number of classes
        if hasattr(model, "n_classes_"):
            n_classes = model.n_classes_
            if n_classes <= 255:
                dtype = "uint8"
            else:
                dtype = "uint16"
        else:
            dtype = "uint8"

        return process_tiles_to_dir(
            gt,
            tiles,
            output_dir,
            classify_embedding,
            year=year,
            max_workers=max_workers,
            output_dtype=dtype,
            show_progress=show_progress,
        )

    finally:
        # Clean up model file
        Path(model_path).unlink()
