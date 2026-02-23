"""
CMIP6 Model Evaluation Suite
Author: Naufal Jouhary

Description:
Comprehensive spatial and statistical evaluation of CMIP6 Global Climate Models (GCMs) 
against ERA5 reanalysis data to select optimal boundary conditions for dynamical downscaling.

Data Provenance & Preprocessing Strategy:
-----------------------------------------
1. Reference Data (ERA5 Reanalysis):
   - Source: Copernicus Climate Change Service (C3S)
   - URL: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-timeseries
   - Variables: u10, v10 (converted to sfcWind), and SSRD (converted to rsds).
   - Temporal: Hourly data resampled to daily mean (daymean) and monthly mean (monmean).

2. Global Climate Models (CMIP6):
   - Source: Earth System Grid Federation (ESGF) nodes (e.g., DKRZ, CEDA, NCI).
   - Period: 1991-2020. 
   - Methodology Note: The 1991-2020 period is constructed by stitching the 'Historical' 
     experiment (1991-2014) with the 'SSP5-8.5' scenario (2015-2020). SSP5-8.5 is utilized 
     specifically as an extreme stress-test; if a GCM aligns well with ERA5 under the most 
     sensitive and highly variable scenario, its baseline physics are deemed robust without 
     excessive over/underestimation.
   - Spatial: Global data subsetted to the Indonesian Maritime Continent domain.

Execution:
----------
Run via terminal/CMD with arguments to save memory:
$ python 02_model_evaluation.py --module [taylor|tss|matrix|all] --var [sfcWind|rsds|all] --time [daily|monthly|climatology|all]

Example: 
$ python 02_model_evaluation.py --module taylor tss --time monthly
"""

import os
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from tabulate import tabulate
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. GLOBAL CONFIGURATION
# ==========================================
# Ganti ini jadi r"./data/1-seleksi_model" kalau mau dipush ke GitHub biar reproducible!
BASE_DIR = r"C:\IPBUniversity\CoolYeah\FASTTRACK\TESIS\data\1-seleksi_model\wget_script\analisis"
ERA5_DIR = os.path.join(BASE_DIR, "ERA5")
OUTPUT_DIR = r"C:\IPBUniversity\CoolYeah\FASTTRACK\TESIS\data\1-seleksi_model\output\github"

os.makedirs(OUTPUT_DIR, exist_ok=True)

GCM_MODELS = {
    "CanESM5": "canesm", "EC-Earth3": "ecearth", "IPSL-CM6A-LR": "ipsl",
    "MPI-ESM1-2-HR": "mpi-hr", "MPI-ESM1-2-LR": "mpi-lr",
    "NorESM2-LM": "noresm-lm", "NorESM2-MM": "noresm-mm"
}
MATRIX_MODELS = {"ERA5": "obs", **GCM_MODELS}
SEASONS = ["ANNUAL", "DJF", "MAM", "JJA", "SON"]
EXTENT = [94.5, 141.5, -11.5, 7]

MODEL_STYLES = {
    "CanESM5":      {"m": "o", "c": "black"},
    "EC-Earth3":    {"m": "s", "c": "dimgray"},
    "IPSL-CM6A-LR": {"m": "^", "c": "gray"},
    "MPI-ESM1-2-HR":{"m": "D", "c": "darkgray"},
    "MPI-ESM1-2-LR":{"m": "v", "c": "silver"},
    "NorESM2-LM":   {"m": "p", "c": "lightgray"},
    "NorESM2-MM":   {"m": "*", "c": "black"}
}

# ==========================================
# 2. CORE UTILITY FUNCTIONS
# ==========================================
def force_to_datetimeindex(ds):
    if hasattr(ds.time.values[0], 'calendar'):
        ds = ds.assign_coords(time=pd.to_datetime(ds.time.dt.strftime('%Y-%m-%d')))
    return ds

def get_taylor_stats(obs_data, mod_data):
    obs_adj, mod_adj = xr.align(obs_data, mod_data, join='inner')
    mod_adj = mod_adj.interp_like(obs_adj, method='nearest')
    o_flat, m_flat = obs_adj.values.flatten(), mod_adj.values.flatten()
    mask = ~np.isnan(o_flat) & ~np.isnan(m_flat)
    o_flat, m_flat = o_flat[mask], m_flat[mask]
    
    if len(o_flat) == 0: return None, None, None
    r = np.corrcoef(o_flat, m_flat)[0, 1]
    std_norm = np.std(m_flat) / np.std(o_flat)
    rmsd_norm = np.sqrt(1 + std_norm**2 - 2 * std_norm * r)
    return r, std_norm, rmsd_norm

def calculate_tss_eq5(obs, mod):
    r = xr.corr(obs, mod, dim='time')
    sigma_hat = mod.std(dim='time') / obs.std(dim='time')
    return (4 * (1 + r)**4) / (((sigma_hat + 1/sigma_hat)**2) * (1 + 1.0)**4)

def clean_metadata(ds):
    ds.attrs = {}
    if 'height' in ds.coords: ds = ds.drop_vars('height')
    return ds

# ==========================================
# 3. EVALUATION ENGINES
# ==========================================
def generate_taylor_diagrams(target_vars, target_modes):
    print(f"\n[Executing] Taylor Diagram Module (Modes: {', '.join(target_modes)})...")
    for var in target_vars:
        num_plots = len(target_modes)
        fig, axes = plt.subplots(1, num_plots, figsize=(8*num_plots, 9), subplot_kw={'projection': 'polar'})
        if num_plots == 1: axes = [axes]
        
        all_stats = [] 

        for idx, mode in enumerate(target_modes):
            ax = axes[idx]
            ax.set_thetamin(0); ax.set_thetamax(90)
            
            for rmsd in [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]:
                ax.add_artist(plt.Circle((1, 0), rmsd, transform=ax.transData._b, color='gray', linestyle='--', alpha=0.3, fill=False))
            t_ref = np.linspace(0, np.pi/2, 100)
            ax.plot(t_ref, np.ones_like(t_ref), color='lightgray', linewidth=2.0, alpha=0.8, zorder=5)
            ax.plot(0, 1, 'ko', markersize=12, label='Ref (ERA5)', zorder=20)

            for m_name, m_alias in GCM_MODELS.items():
                try:
                    ds_m = xr.open_dataset(os.path.join(BASE_DIR, f"{var}_{m_name}_1991-2020_Indo.nc"))[var]
                    ds_o_raw = xr.open_dataset(os.path.join(ERA5_DIR, f"{var}_era5-daily_{m_alias}-grid_1991-2020.nc"))
                    t_d = 'valid_time' if 'valid_time' in ds_o_raw.dims else 'time'
                    ds_o = ds_o_raw[var].rename({t_d: 'time'})

                    if mode == "daily":
                        ds_m, ds_o = ds_m.assign_coords(time=ds_m.time.dt.strftime('%Y-%m-%d').values), ds_o.assign_coords(time=ds_o.time.dt.strftime('%Y-%m-%d').values)
                    elif mode == "monthly":
                        ds_m, ds_o = ds_m.resample(time='1MS').mean().assign_coords(time=lambda x: x.time.dt.strftime('%Y-%m').values), ds_o.resample(time='1MS').mean().assign_coords(time=lambda x: x.time.dt.strftime('%Y-%m').values)
                    elif mode == "climatology":
                        ds_m, ds_o = ds_m.mean(dim='time'), ds_o.mean(dim='time')

                    r, sd, rmsd = get_taylor_stats(ds_o, ds_m)
                    if r is not None:
                        style = MODEL_STYLES[m_name]
                        ax.plot(np.arccos(r), sd, style['m'], color=style['c'], markersize=10, label=m_name if idx==(num_plots-1) else "", markeredgecolor='black', alpha=0.9, zorder=15)
                        all_stats.append([mode.capitalize(), m_name, round(r, 4), round(sd, 4), round(rmsd, 4)])
                except Exception: pass

            tick_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
            ax.set_xticks(np.arccos(tick_vals))
            ax.set_xticklabels([str(t) for t in tick_vals])
            ax.text(np.deg2rad(45), 1.55, "Correlation Coefficient", rotation=-45, ha='center', va='bottom', fontweight='bold', fontsize=11)
            
            sd_ticks = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
            ax.set_yticks(sd_ticks)
            ax.set_yticklabels([str(s) for s in sd_ticks])
            ax.set_rlabel_position(90)
            ax.text(np.deg2rad(-10), 0.75, "Normalized Standard Deviation", ha='center', va='bottom', fontweight='bold', fontsize=11)
            ax.set_ylim(0, 1.5); ax.set_title(f"{mode.capitalize()} Data", pad=35, fontweight='bold', fontsize=14)

        plt.suptitle(f"Taylor Diagram: {var} (1991-2020)", fontsize=16, y=1.05 if num_plots==1 else 1.0, fontweight='bold')
        axes[-1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, frameon=True)
        
        save_suffix = "_".join(target_modes)
        plt.savefig(os.path.join(OUTPUT_DIR, f"Taylor_Final_{var}_{save_suffix}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        if all_stats:
            pd.DataFrame(all_stats, columns=['Mode', 'Model', 'R', 'SD_norm', 'RMSD_norm']).to_csv(os.path.join(OUTPUT_DIR, f"Statistics_Taylor_{var}_{save_suffix}.csv"), index=False)
    print("  > Taylor Diagrams Output Successful.")

def generate_tss_maps(target_vars, target_modes):
    valid_tss_modes = [m for m in target_modes if m in ["daily", "monthly"]]
    
    if not valid_tss_modes:
        print("\n[Executing] Spatial TSS Maps Module...")
        print("  > SKIPPED: TSS cannot be calculated using 'climatology' mode (requires time dimension).")
        return

    print(f"\n[Executing] Spatial TSS Maps Module (Modes: {', '.join(valid_tss_modes)})...")
    for mode in valid_tss_modes:
        for var in target_vars:
            fig, axes = plt.subplots(4, 2, figsize=(16, 12.7), constrained_layout=True, subplot_kw={'projection': ccrs.PlateCarree()})
            fig.set_constrained_layout_pads(w_pad=0, h_pad=0, hspace=0.002, wspace=0.004)
            axes_flat, im = axes.flatten(), None 

            for i, (m_name, m_alias) in enumerate(GCM_MODELS.items()):
                try:
                    ds_m = force_to_datetimeindex(xr.open_dataset(os.path.join(BASE_DIR, f"{var}_{m_name}_1991-2020_Indo.nc"))[var]).assign_coords(time=lambda x: x.time.dt.floor('D')).drop_duplicates('time')
                    ds_o_raw = xr.open_dataset(os.path.join(ERA5_DIR, f"{var}_era5-daily_{m_alias}-grid_1991-2020.nc"))
                    ds_o = force_to_datetimeindex(ds_o_raw[var].rename({'valid_time' if 'valid_time' in ds_o_raw.dims else 'time': 'time'})).assign_coords(time=lambda x: x.time.dt.floor('D')).drop_duplicates('time')

                    if mode == "monthly": ds_m, ds_o = ds_m.resample(time='1MS').mean(), ds_o.resample(time='1MS').mean()
                    ds_m, ds_o = xr.align(ds_m, ds_o, join='inner')
                    ds_m = ds_m.reindex_like(ds_o, method='nearest')
                    
                    ax = axes_flat[i]
                    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
                    im = clean_metadata(calculate_tss_eq5(ds_o, ds_m)).plot(ax=ax, transform=ccrs.PlateCarree(), cmap='Spectral', vmin=0, vmax=0.7, add_colorbar=False, add_labels=False)
                    ax.coastlines(resolution='10m', color='black', linewidth=0.5)
                    ax.text(0.02, 0.05, f"{m_name}", transform=ax.transAxes, fontsize=10, fontweight='bold', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=2), ha='left', va='bottom')
                except Exception: axes_flat[i].set_visible(False)
            
            axes_flat[-1].set_visible(False)
            if im:
                cb = fig.colorbar(im, cax=fig.add_axes([0.0, -0.0175, 1, 0.012]), orientation='horizontal')
                cb.set_label(f'Taylor Skill Score (TSS)', fontsize=12, fontweight='bold', labelpad=5)
            
            plt.suptitle(f"Taylor Skill Score (TSS) - {var} {mode.capitalize()} (1991-2020)", fontsize=16, y=1.025, fontweight='bold')
            plt.savefig(os.path.join(OUTPUT_DIR, f"TSS_Seamless_Spectral_{mode}_{var}.png"), dpi=300, bbox_inches='tight', pad_inches=0.05)
            plt.close()
    print("  > TSS Maps Output Successful.")

def generate_climatology_matrices(target_vars):
    print("\n[Executing] Climatology Matrices Module (Seasonal)...")
    for var in target_vars:
        fig, axes = plt.subplots(len(SEASONS), len(MATRIX_MODELS), figsize=(40, 11.8), subplot_kw={'projection': ccrs.PlateCarree()})
        plt.subplots_adjust(left=0.06, right=0.98, bottom=0.12, top=0.90, hspace=0.002, wspace=0.01)
        cmap_choice, v_min, v_max, unit = ('rainbow', 0, 10, "m/s") if var == "sfcWind" else ('YlOrRd', 150, 300, "W/m²")
        im = None
        for r_idx, season in enumerate(SEASONS):
            for c_idx, (m_name, m_alias) in enumerate(MATRIX_MODELS.items()):
                try:
                    if m_name == "ERA5":
                        ds_raw = xr.open_dataset(os.path.join(ERA5_DIR, f"sfcWind_era5-daily_1991-2020.nc" if var == "sfcWind" else f"rsds_era5-daily_1991-2020.nc"))
                        if 'valid_time' in ds_raw.coords or 'valid_time' in ds_raw.dims: ds_raw = ds_raw.rename({'valid_time': 'time'})
                        ds = ds_raw[var] if var in ds_raw.data_vars else ds_raw[[v for v in ds_raw.data_vars if 'bnds' not in v][0]]
                        if var == "rsds" and ds.max() > 1000: ds = ds / 86400
                    else:
                        ds = xr.open_dataset(os.path.join(BASE_DIR, f"{var}_{m_name}_1991-2020_Indo.nc"))[var]
                    
                    ds = force_to_datetimeindex(ds).assign_coords(time=lambda x: x.time.dt.floor('D')).drop_duplicates('time')
                    ds_mean = clean_metadata(ds.mean('time') if season == "ANNUAL" else ds.groupby('time.season').mean('time').sel(season=season))

                    ax = axes[r_idx, c_idx]
                    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
                    im = ds_mean.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap_choice, vmin=v_min, vmax=v_max, add_colorbar=False, add_labels=False)
                    ax.coastlines(resolution='10m', color='black', linewidth=0.5); ax.gridlines(draw_labels=False, linestyle=':', alpha=0.1)

                    if r_idx == 0: ax.set_title(m_name, fontsize=18, fontweight='bold', pad=15)
                    if c_idx == 0: ax.text(-0.05, 0.5, season, transform=ax.transAxes, fontsize=20, fontweight='bold', rotation=90, va='center', ha='right')
                except Exception: axes[r_idx, c_idx].set_visible(False)

        if im:
            cb = fig.colorbar(im, cax=fig.add_axes([0.06, 0.08, 0.92, 0.02]), orientation='horizontal')
            cb.set_label(f'{var} ({unit})', fontsize=18, fontweight='bold'); cb.ax.tick_params(labelsize=14)

        plt.suptitle(f"{var.upper()}: 1991-2020", fontsize=28, y=0.98, fontweight='bold')
        plt.savefig(os.path.join(OUTPUT_DIR, f"GrandMatrix_Final_{var}.png"), dpi=300, bbox_inches='tight')
        plt.close()
    print("  > Climatology Matrices Output Successful.")

# ==========================================
# 4. CLI EXECUTION CONTROL
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CMIP6 Model Evaluation Suite")
    parser.add_argument("--module", nargs='+', choices=["taylor", "tss", "matrix", "all"], default=["all"], help="Select evaluation module to run.")
    parser.add_argument("--var", choices=["sfcWind", "rsds", "all"], default="all", help="Select climate variable to process.")
    parser.add_argument("--time", nargs='+', choices=["daily", "monthly", "climatology", "all"], default=["all"], help="Select temporal resolution for Taylor/TSS.")
    
    args, unknown = parser.parse_known_args()

    target_vars = ["sfcWind", "rsds"] if args.var == "all" else [args.var]
    target_modes = ["daily", "monthly", "climatology"] if "all" in args.time else args.time
    target_modules = ["taylor", "tss", "matrix"] if "all" in args.module else args.module
    
    print(f"{'='*50}\nCMIP6 EVALUATION PIPELINE INITIATED\nTarget Modules: {', '.join(target_modules).upper()}\nTarget Variable: {args.var.upper()}\nTime Modes: {', '.join(target_modes).upper()}\n{'='*50}")

    if "taylor" in target_modules: generate_taylor_diagrams(target_vars, target_modes)
    if "tss" in target_modules: generate_tss_maps(target_vars, target_modes)
    if "matrix" in target_modules: generate_climatology_matrices(target_vars)

    print(f"\n{'='*50}\nALL REQUESTED OPERATIONS COMPLETED. CHECK OUTPUT DIR.\n{'='*50}")