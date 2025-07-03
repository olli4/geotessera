# generate_map.py
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import datetime

def create_map():
    """
    Generate a world map with the current time and several fixed points.
    In practical applications, you should fetch your real-time data here.
    """
    # --- 1. Fetch your data ---
    # This is an example. You should replace it with your own data fetching logic.
    # For example: Fetching the latest earthquake locations from an API, your movement trajectory, etc.
    data = {
        'City': ['London', 'New York', 'Tokyo', 'Sydney'],
        'Latitude': [51.5074, 40.7128, 35.6895, -33.8688],
        'Longitude': [-0.1278, -74.0060, 139.6917, 151.2093]
    }
    df = pd.DataFrame(data)

    # Convert pandas DataFrame to GeoDataFrame
    gdf = geopandas.GeoDataFrame(
        df, geometry=geopandas.points_from_xy(df.Longitude, df.Latitude)
    )
    gdf.set_crs("EPSG:4326", inplace=True) # Set the coordinate system to WGS84

    # --- 2. Plot the map ---
    # Directly load a local shapefile
    world = geopandas.read_file('world_map/ne_110m_admin_0_countries.shp')

    # Create the plotting area
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    # Plot the world map base layer
    world.plot(ax=ax, color='lightgray', edgecolor='black')

    # Plot your data points on the map
    gdf.plot(ax=ax, marker='o', color='red', markersize=50, label='My Points')

    # --- 3. Customize and save the map ---
    # Add a title with the current time
    current_time_utc = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    ax.set_title(f'Real-time Map\nLast Updated: {current_time_utc}', fontdict={'fontsize': '16', 'fontweight': '3'})

    # Hide the axes
    ax.set_axis_off()

    # Save the map as an image file, bbox_inches='tight' trims excess whitespace
    # The filename 'map.png' is important and will be used later
    plt.savefig('map.png', dpi=150, bbox_inches='tight')
    print("Map generated successfully!")

if __name__ == '__main__':
    create_map()