"""
Project: Continuous 60-Year CF Trend Analysis
Description: 
- Concatenates Historical (1991-2014) and SSP (2015-2050) datasets for continuous 60-year time series.
- Computes ensemble averaging dynamically.
- Calculates Theil-Sen slope and Mann-Kendall p-value for annual and seasonal capacity factors.
- Evaluates standard Wind, standard Solar PV, and daylight-only Solar PV.
"""

import xarray as xr
import numpy as np
import os
import warnings
from scipy.stats import theilslopes, kendalltau
import gc

warnings.filterwarnings("ignore")

in_dir = "../data/processed"
out_dir = "../data/processed/trends"
os.makedirs(out_dir, exist_ok=True)

models = ['EC-Earth3', 'NorESM']
ssps = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
seasons = ['DJF', 'MAM', 'JJA', 'SON']

# Configuration: (Input_Prefix, Input_Suffix, Output_Name, Internal_Var_Name)
vars_config = [
    ('CF_WIND', '', 'CF_WIND', 'cf_wind'),
    ('CF_SOLAR', '', 'CF_SOLAR', 'cf_solar'),
    ('CF_SOLAR', '_siang', 'CF_SOLAR_daylight', 'cf_solar')
]

def calc_trend_1d(y):
    if np.isnan(y).all() or len(np.unique(y)) == 1: 
        return np.nan, np.nan
    x = np.arange(len(y))
    slope, _, _, _ = theilslopes(y, x, 0.95)
    _, pval = kendalltau(x, y)
    return slope, pval

def apply_trend_to_da(da):
    slope, pval = xr.apply_ufunc(
        calc_trend_1d, da,
        input_core_dims=[['time']],
        output_core_dims=[[], []],
        vectorize=True,
        dask='forbidden'
    )
    return slope, pval

print("==========================================================")
print("Initializing Continuous 60-Year CF Trend Analysis")
print("==========================================================")

for prefix, suffix, out_name, var_name in vars_config:
    print(f"\n[ Processing 60-Year Trend: {out_name} ]")
    for ssp in ssps:
        
        da_models = []
        for mod in models:
            # Dynamically construct file paths
            f_hist = os.path.join(in_dir, f"{prefix}_1D_{mod}_hist{suffix}.nc")
            f_ssp = os.path.join(in_dir, f"{prefix}_1D_{mod}_{ssp}{suffix}.nc")
            
            if os.path.exists(f_hist) and os.path.exists(f_ssp):
                with xr.open_dataset(f_hist, engine='netcdf4') as ds_h, xr.open_dataset(f_ssp, engine='netcdf4') as ds_s:
                    v_key = var_name if var_name in ds_h.data_vars else list(ds_h.data_vars)[0]
                    da_h = ds_h[v_key].load()
                    da_s = ds_s[v_key].load()
                    
                    # Concatenate historical and future scenarios continuously
                    da_concat = xr.concat([da_h, da_s], dim='time')
                    da_models.append(da_concat)
        
        if len(da_models) != 2:
            print(f"   [SKIP] Incomplete ensemble data for HIST + {ssp.upper()}.")
            continue
            
        print(f"   Assembling ensemble for HIST + {ssp.upper()}...")
        # Compute continuous ensemble mean
        da_ens = (da_models[0] + da_models[1]) / 2.0
        ds_out = xr.Dataset()

        print("      -> Calculating Annual Trends (1991-2050)...")
        da_annual = da_ens.resample(time='1Y').mean()
        slope_ann, pval_ann = apply_trend_to_da(da_annual)
        ds_out['slope_annual'] = slope_ann
        ds_out['pval_annual'] = pval_ann

        print("      -> Calculating Seasonal Trends (1991-2050)...")
        da_seasonal = da_ens.resample(time='QS-DEC').mean()
        for season in seasons:
            da_s = da_seasonal.sel(time=da_seasonal.time.dt.season == season)
            slope_s, pval_s = apply_trend_to_da(da_s)
            ds_out[f'slope_{season}'] = slope_s
            ds_out[f'pval_{season}'] = pval_s

        # Save output
        out_file = os.path.join(out_dir, f"TREND_60YR_{out_name}_Ensemble_{ssp}.nc")
        ds_out.to_netcdf(out_file)
        print(f"   [SAVED] {os.path.basename(out_file)}")
        
        del da_models, da_ens, da_annual, da_seasonal, ds_out
        gc.collect()

print("\n[SUCCESS] 60-Year CF Trend Analysis completed.")
