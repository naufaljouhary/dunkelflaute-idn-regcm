"""
Topography Visualization Module
Author: Naufal Jouhary
Description: Generates a high-resolution topographic map of the Indonesian archipelago
using DEM data. Includes hillshading and custom ocean masking for publication-quality output.
"""

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LightSource, PowerNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def generate_topography_map(input_filepath, output_filepath, lon_range=(94, 142), lat_range=(-12, 10)):
    """
    Reads NetCDF DEM data, applies hillshading and ocean masking, and saves a spatial map.
    
    Parameters:
    - input_filepath (str): Path to the input NetCDF file.
    - output_filepath (str): Path to save the output PNG image.
    - lon_range (tuple): Longitude bounding box (min, max).
    - lat_range (tuple): Latitude bounding box (min, max).
    """
    print(f"Loading topographic data from: {input_filepath}...")

    # ==========================================
    # 1. LOAD AND SUBSET DATA
    # ==========================================
    ds = xr.open_dataset(input_filepath)
    ds_subset = ds.sel(lat=slice(lat_range[0], lat_range[1]), lon=slice(lon_range[0], lon_range[1]))

    topo = ds_subset['z'].values
    lat = ds_subset['lat'].values
    lon = ds_subset['lon'].values

    # ==========================================
    # 2. COLORMAP & SHADING LOGIC
    # ==========================================
    # Extract terrain colors, excluding the deep ocean blues (start from 0.25)
    terrain_cmap = plt.cm.terrain
    new_colors = terrain_cmap(np.linspace(0.25, 1.0, 256))
    land_cmap = mcolors.LinearSegmentedColormap.from_list("land_only", new_colors)
    norm = PowerNorm(gamma=0.4, vmin=0, vmax=5000)

    # Apply hillshading for 3D terrain effect
    ls = LightSource(azdeg=315, altdeg=45)
    topo_fill = np.nan_to_num(topo, nan=0) # Replace NaNs with 0 for hillshade calculation

    # Generate shaded RGBA array
    shaded_rgba = ls.shade(topo_fill, cmap=land_cmap, norm=norm, vert_exag=100, blend_mode='overlay')

    # Create ocean mask (elevations <= 0 or NaNs) and apply to Alpha channel (index 3)
    is_ocean = (topo <= 0) | (np.isnan(topo))
    rgba_img = shaded_rgba.copy()
    rgba_img[:, :, 3] = np.where(is_ocean, 0.0, 1.0) # 0.0 = transparent for ocean

    # ==========================================
    # 3. SPATIAL PLOTTING
    # ==========================================
    print("Generating spatial plot...")
    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_range[0], lon_range[1], lat_range[0], lat_range[1]], crs=ccrs.PlateCarree())

    # Set ocean background color
    ocean_color = '#d4f1f9'
    ax.set_facecolor(ocean_color)
    ax.add_feature(cfeature.OCEAN, color=ocean_color, zorder=0)

    # Plot the processed RGBA image
    img_extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    ax.imshow(rgba_img, extent=img_extent, transform=ccrs.PlateCarree(), origin='lower', zorder=1)

    # Add cartographic features (coastlines, borders, gridlines)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, color='black', zorder=2)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.6, color='black', zorder=2)
    
    gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.3, color='gray', zorder=3)
    gl.top_labels = True
    gl.bottom_labels = True
    gl.left_labels = True
    gl.right_labels = True
    gl.xlabel_style = {'size': 9}
    gl.ylabel_style = {'size': 9}

    # ==========================================
    # 4. COLORBAR SETUP
    # ==========================================
    sm = plt.cm.ScalarMappable(cmap=land_cmap, norm=norm)
    sm.set_array([])
    pos = ax.get_position()
    cax = fig.add_axes([pos.x0, pos.y0 - 0.08, pos.width, 0.025])
    cbar = plt.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.set_label('Elevation (m.a.s.l.)', fontsize=11)
    cbar.set_ticks([0, 200, 500, 1000, 2000, 3000, 4000, 5000])

    # ==========================================
    # 5. SAVE OUTPUT
    # ==========================================
    plt.savefig(output_filepath, dpi=300, bbox_inches='tight')
    print(f"Success! Map saved to: {output_filepath}")
    plt.close() # Good practice to close figure to free up memory

if __name__ == "__main__":
    # Define RELATIVE paths for reproducibility on any machine
    # Assuming script is run from the project root folder
    DATA_DIR = r'./data'
    OUTPUT_DIR = r'./outputs'
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    INPUT_FILE = os.path.join(DATA_DIR, 'GMTED_DEM_indo.nc')
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'topography_map_indonesia.png')
    
    # Execute function
    # Note: Before running, ensure 'GMTED_DEM_indo.nc' is placed in the './data' folder
    generate_topography_map(INPUT_FILE, OUTPUT_FILE)
