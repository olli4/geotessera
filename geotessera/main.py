"""Example script demonstrating basic GeoTessera usage.

This module shows how to use the Pooch library directly to fetch
Tessera embedding files. It's primarily a development example and
test script rather than a user-facing tool.

For normal usage, prefer the GeoTessera class or CLI commands.
"""

import importlib.resources
import pooch
from .core import TESSERA_BASE_URL


def main():
    """Demonstrate direct Pooch usage for fetching Tessera embeddings.

    This example shows the low-level approach to downloading embedding
    files using Pooch directly. It fetches a single embedding file for
    a specific location near Cambridge, UK.

    The process demonstrates:
    1. Creating a Pooch instance with Tessera's data URL
    2. Loading the registry file for year 2024
    3. Fetching a specific embedding file with progress bar
    4. Printing the local cache path where the file was saved

    Note:
        This is a development example. For production use, prefer:
        >>> from geotessera import GeoTessera
        >>> gt = GeoTessera()
        >>> embedding = gt.fetch_embedding(lat=52.05, lon=0.15)
    """
    version = "v1"
    POOCH = pooch.create(
        path=pooch.os_cache("geotessera"),
        base_url=f"{TESSERA_BASE_URL}/{version}/global_0.1_degree_representation/",
        version=version,
        registry=None,
    )

    with importlib.resources.open_text(
        "geotessera", "registry_2024.txt"
    ) as registry_file:
        POOCH.load_registry(registry_file)

    fname = POOCH.fetch("2024/grid_0.15_52.05/grid_0.15_52.05.npy", progressbar=True)
    print(fname)


if __name__ == "__main__":
    main()
