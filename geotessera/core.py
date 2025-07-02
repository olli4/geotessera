"""Core GeoTessera functionality for accessing geospatial embeddings."""
import os
from pathlib import Path
from typing import Optional, Union, List, Dict
import importlib.resources
import pooch


class GeoTessera:
    """Main class for accessing GeoTessera geospatial embeddings.
    
    This class provides methods to fetch and load geospatial embeddings
    from the GeoTessera dataset.
    
    Attributes:
        version: Version of the GeoTessera dataset to use (default: "v1")
        cache_dir: Directory to cache downloaded files
    """
    
    def __init__(self, version: str = "v1", cache_dir: Optional[Union[str, Path]] = None):
        """Initialize GeoTessera client.
        
        Args:
            version: Version of the dataset to use
            cache_dir: Directory to cache downloaded files. If None, uses system cache.
        """
        self.version = version
        self._cache_dir = cache_dir
        self._pooch = None
        self._initialize_pooch()
    
    def _initialize_pooch(self):
        """Initialize the Pooch downloader with registry."""
        cache_path = self._cache_dir if self._cache_dir else pooch.os_cache("geotessera")
        
        self._pooch = pooch.create(
            path=cache_path,
            base_url=f"https://dl-1.tessera.wiki/{self.version}/global_0.1_degree_representation/",
            version=self.version,
            registry=None,
        )
        
        # Load the registry file
        with importlib.resources.open_text("geotessera", "registry_2024.txt") as registry_file:
            self._pooch.load_registry(registry_file)
    
    def fetch_embedding(self, lat: float, lon: float, year: int = 2024, 
                       progressbar: bool = True) -> str:
        """Fetch embedding file for a specific location.
        
        Args:
            lat: Latitude coordinate
            lon: Longitude coordinate
            year: Year of the embedding (default: 2024)
            progressbar: Show download progress bar
            
        Returns:
            Path to the downloaded file
        """
        # Format coordinates to match file naming convention
        grid_name = f"grid_{lon:.2f}_{lat:.2f}"
        file_path = f"{year}/{grid_name}/{grid_name}.npy"
        
        return self._pooch.fetch(file_path, progressbar=progressbar)
    
    def get_embedding_path(self, lat: float, lon: float, year: int = 2024) -> str:
        """Get the local file path for an embedding after downloading.
        
        Args:
            lat: Latitude coordinate
            lon: Longitude coordinate
            year: Year of the embedding (default: 2024)
            
        Returns:
            Path to the embedding file
        """
        return self.fetch_embedding(lat, lon, year, progressbar=True)
    
    def list_available_embeddings(self) -> List[str]:
        """List all available embeddings in the registry.
        
        Returns:
            List of available embedding files
        """
        return list(self._pooch.registry.keys())
    
    def get_registry_info(self) -> Dict[str, str]:
        """Get information about all files in the registry.
        
        Returns:
            Dictionary mapping file paths to their checksums
        """
        return dict(self._pooch.registry)