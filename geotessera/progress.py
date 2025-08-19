"""Custom HTTP downloader for Pooch with Rich progress integration.

This module integrates Pooch downloads with Rich progress displays using a
custom HTTP downloader.
"""

import os
import re
from typing import Optional, Callable
from pathlib import Path


def extract_tile_coordinates_from_url(url: str) -> str:
    """Extract and format tile coordinates from tessera URL.

    Args:
        url: URL like 'https://dl-2.tessera.wiki/v1/global_0.1_degree_representation/2024/grid_-74.05_41.05/grid_-74.05_41.05.npy'

    Returns:
        Formatted coordinate string like '74.05°W, 41.05°N' or original filename if parsing fails
    """
    # Extract grid coordinates from URL pattern: grid_LON_LAT
    match = re.search(r"grid_([-\d.]+)_([-\d.]+)", url)
    if not match:
        # Fallback to filename from URL
        return os.path.basename(url)

    lon_str, lat_str = match.groups()
    try:
        lon = float(lon_str)
        lat = float(lat_str)

        # Format with directions
        lon_dir = "W" if lon < 0 else "E"
        lat_dir = "S" if lat < 0 else "N"

        return f"{abs(lon):.2f}°{lon_dir}, {abs(lat):.2f}°{lat_dir}"
    except ValueError:
        # Fallback if parsing fails
        return os.path.basename(url)


class RichHTTPDownloader:
    """Custom HTTP downloader for Pooch that integrates with Rich progress bars."""

    def __init__(
        self,
        progress_callback: Optional[Callable] = None,
        description: str = "Downloading",
    ):
        """Initialize the downloader.

        Args:
            progress_callback: Function to call with (current, total, status)
            description: Description for the download operation
        """
        self.progress_callback = progress_callback
        self.description = description

    def __call__(self, url: str, output_file: str, known_hash: Optional[str] = None):
        """Download a file from the given URL.

        Args:
            url: URL to download from
            output_file: Path to save the downloaded file
            known_hash: Expected hash of the file (for verification)

        Returns:
            Path to the downloaded file
        """
        import requests

        # Ensure output directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Start download with streaming
        with requests.get(url, stream=True, timeout=30) as response:
            response.raise_for_status()

            # Get total file size from headers
            total_size = int(response.headers.get("content-length", 0))

            # Initialize progress
            downloaded = 0
            if self.progress_callback and total_size > 0:
                tile_name = extract_tile_coordinates_from_url(url)
                status = f"{self.description} {tile_name}"
                self.progress_callback(0, total_size, status)

            # Download in chunks
            chunk_size = 8192
            with open(output_file, "wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:  # Filter out keep-alive chunks
                        f.write(chunk)
                        downloaded += len(chunk)

                        # Update progress
                        if self.progress_callback and total_size > 0:
                            tile_name = extract_tile_coordinates_from_url(url)
                            percent = (downloaded / total_size) * 100
                            status = f"{self.description} {tile_name} ({percent:.0f}%)"
                            self.progress_callback(downloaded, total_size, status)

            # Final progress update
            if self.progress_callback and total_size > 0:
                tile_name = extract_tile_coordinates_from_url(url)
                self.progress_callback(
                    total_size, total_size, f"Downloaded {tile_name}"
                )

        return output_file


def create_rich_downloader(
    progress_callback: Optional[Callable] = None, description: str = "Downloading"
):
    """Create a Rich-integrated HTTP downloader for Pooch.

    Args:
        progress_callback: Function to call with (current, total, status)
        description: Description for the download operation

    Returns:
        HTTP downloader compatible with Pooch
    """
    return RichHTTPDownloader(
        progress_callback=progress_callback, description=description
    )


def should_show_pooch_progress() -> bool:
    """Determine if Pooch progress should be shown based on environment.

    Returns:
        True if progress should be shown, False otherwise
    """
    # Check if we're in a non-interactive environment
    if not os.isatty(0):  # stdin is not a terminal
        return False

    # Check environment variables that might indicate automation
    ci_indicators = [
        "CI",
        "CONTINUOUS_INTEGRATION",
        "GITHUB_ACTIONS",
        "GITLAB_CI",
        "JENKINS_URL",
        "BUILDKITE",
    ]

    if any(os.getenv(var) for var in ci_indicators):
        return False

    return True
