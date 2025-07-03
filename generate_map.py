# generate_map.py
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import datetime
import requests
import numpy as np
from pathlib import Path

def fetch_grid_coordinates():
    """
    Fetch grid coordinates from the registry file.
    Returns a DataFrame with latitude and longitude columns.
    """
    # Fetch the registry file from the URL
    registry_url = "https://dl-1.tessera.wiki/v1/global_0.1_degree_representation/registry_2024.txt"
    
    print("Fetching registry file...")
    response = requests.get(registry_url)
    response.raise_for_status()
    
    # Parse the registry content
    coordinates = set()  # Use set to avoid duplicates
    
    for line in response.text.strip().split('\n'):
        if line:
            # Extract the file path (first part before the hash)
            file_path = line.split()[0]
            
            # Only process main grid files (not _scales files)
            if file_path.endswith('.npy') and not '_scales.npy' in file_path:
                # Extract grid name from path like "2024/grid_-0.05_10.05/grid_-0.05_10.05.npy"
                parts = file_path.split('/')
                if len(parts) >= 2:
                    grid_name = parts[1]  # e.g., "grid_-0.05_10.05"
                    
                    if grid_name.startswith('grid_'):
                        # Remove "grid_" prefix and split by underscore
                        coords = grid_name[5:].split('_')
                        if len(coords) == 2:
                            try:
                                lon = float(coords[0])
                                lat = float(coords[1])
                                coordinates.add((lat, lon))
                            except ValueError:
                                continue
    
    print(f"Found {len(coordinates)} unique grid points")
    
    # Convert to DataFrame
    coords_list = list(coordinates)
    df = pd.DataFrame(coords_list, columns=['Latitude', 'Longitude'])
    
    return df

def create_map():
    """
    Generate a world map with all available grid points from the registry.
    """
    # --- 1. Fetch grid coordinates ---
    df = fetch_grid_coordinates()
    
    # Convert pandas DataFrame to GeoDataFrame
    print("Creating GeoDataFrame...")
    gdf = geopandas.GeoDataFrame(
        df, geometry=geopandas.points_from_xy(df.Longitude, df.Latitude)
    )
    gdf.set_crs("EPSG:4326", inplace=True)  # Set the coordinate system to WGS84

    # --- 2. Plot the map ---
    # Check if world map shapefile exists
    world_map_path = Path('world_map/ne_110m_admin_0_countries.shp')
    if not world_map_path.exists():
        print("Warning: World map shapefile not found. Using built-in world map.")
        world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    else:
        world = geopandas.read_file(world_map_path)

    # Create the plotting area
    print("Creating map...")
    fig, ax = plt.subplots(1, 1, figsize=(20, 12))

    # Plot the world map base layer
    world.plot(ax=ax, color='lightgray', edgecolor='black', linewidth=0.5)

    # Plot grid points on the map with smaller markers
    gdf.plot(ax=ax, marker='o', color='red', markersize=2, alpha=0.8, label='Available')

    # --- 3. Customize and save the map ---
    # Add a title with the current time and grid count
    current_time_utc = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    ax.set_title(f'GeoTessera Grid Coverage Map\nTotal Grid Points: {len(df)}\nLast Updated: {current_time_utc}', 
                 fontdict={'fontsize': '18', 'fontweight': 'bold'})

    # Add legend
    ax.legend(loc='lower left', frameon=True, fancybox=True, shadow=True, fontsize=12)

    # Set axis labels
    ax.set_xlabel('Longitude', fontsize=14)
    ax.set_ylabel('Latitude', fontsize=14)

    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--')

    # Save the map as an image file
    plt.tight_layout()
    plt.savefig('map.png', dpi=300, bbox_inches='tight')
    print(f"Map generated successfully with {len(df)} grid points!")
    
    # Close the plot to free memory
    plt.close()

if __name__ == '__main__':
    create_map()