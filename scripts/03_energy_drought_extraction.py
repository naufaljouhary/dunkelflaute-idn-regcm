"""
Project: Energy Drought Extraction Engine
Description: 
- Extracts Energy Drought Frequency (EDF) and Duration (EDD).
- Implements 1-day gap pooling methodology (Qu et al.) for consecutive drought events.
- Performs continuous 60-year Theil-Sen trend analysis across Historical and SSP scenarios.
"""

import xarray as xr
import numpy as np
import os
import gc
import warnings
from scipy.stats import theilslopes, kendalltau

warnings.filterwarnings("ignore")

class EnergyDroughtCalculator:
    def __init__(self):
        self.cf_dir = "../data/processed"
        self.out_dir = "../data/processed"
        self.trend_dir = os.path.join(self.out_dir, "trends")
        
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(self.trend_dir, exist_ok=True)
        
        self.models = ['EC-Earth3', 'NorESM']
        self.ssps = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
        
        # Configuration: (Prefix, Suffix, Output_Name, Internal_Var_Name)
        self.vars_config = [
            ('CF_WIND', '', 'WIND', 'cf_wind'),
            ('CF_SOLAR', '', 'SOLAR', 'cf_solar'),
            ('CF_SOLAR', '_siang', 'SOLAR_daylight', 'cf_solar')
        ]
        
    def get_ensemble_data(self, prefix, suffix, var_name, scen):
        das = []
        for model in self.models:
            path = os.path.join(self.cf_dir, f"{prefix}_1D_{model}_{scen}{suffix}.nc")
            if os.path.exists(path):
                with xr.open_dataset(path, engine='netcdf4') as ds:
                    v_key = var_name if var_name in ds.data_vars else list(ds.data_vars)[0]
                    das.append(ds[v_key].load())
        
        if len(das) != 2: return None
        return (das[0] + das[1]) / 2.0

    def extract_annual_edf(self, da_target, var_name):
        mask_10 = da_target < 0.1
        edf = mask_10.resample(time='1Y').sum(dim='time')
        
        edf.name = f"{var_name}_EDF"
        edf.attrs = {'long_name': 'Energy Drought Freq (CF < 0.1)', 'units': 'Days'}
        return edf

    def extract_annual_edd_with_pooling(self, da_target, p20, p40, var_name):
        # Broadcast thresholds to time dimension of da_target
        p20_vals = p20.values[np.newaxis, :, :]
        p40_vals = p40.values[np.newaxis, :, :]
        
        # Initialize arrays for annual EDD
        years = np.unique(da_target.time.dt.year)
        annual_max = np.zeros((len(years), da_target.shape[1], da_target.shape[2]))
        annual_mean = np.zeros((len(years), da_target.shape[1], da_target.shape[2]))
        
        for y_idx, year in enumerate(years):
            da_year = da_target.sel(time=str(year))
            cf_vals = da_year.values
            
            m20 = cf_vals < p20_vals
            m40 = cf_vals < p40_vals
            
            # Detect 1-day gaps for pooling (Qu et al. methodology)
            m20_prev = np.roll(m20, shift=1, axis=0)
            m20_prev[0, :, :] = False  
            
            m20_next = np.roll(m20, shift=-1, axis=0)
            m20_next[-1, :, :] = False 
            
            gaps = (~m20) & m20_prev & m20_next & m40
            m20_pooled = m20 | gaps
            
            # 1. MAX CONSECUTIVE DAYS (WORST CASE)
            max_count = np.zeros(m20_pooled.shape[1:], dtype=int)
            curr_count = np.zeros(m20_pooled.shape[1:], dtype=int)
            for t in range(m20_pooled.shape[0]):
                curr_count = (curr_count + 1) * m20_pooled[t]
                max_count = np.maximum(max_count, curr_count)
            annual_max[y_idx] = max_count
                
            # 2. MEAN CONSECUTIVE DAYS (AVERAGE EVENT DURATION)
            total_drought_days = m20_pooled.sum(axis=0)
            m20_prev_event = np.roll(m20_pooled, shift=1, axis=0)
            m20_prev_event[0, :, :] = False
            
            event_starts = m20_pooled & (~m20_prev_event)
            event_count = event_starts.sum(axis=0)
            
            mean_count = np.zeros_like(total_drought_days, dtype=float)
            valid = event_count > 0
            mean_count[valid] = total_drought_days[valid] / event_count[valid]
            annual_mean[y_idx] = mean_count
            
        time_coords = [np.datetime64(f"{year}-12-31") for year in years]
        
        edd_max = xr.DataArray(annual_max, coords=[time_coords, da_target.lat, da_target.lon], dims=['time', 'lat', 'lon'])
        edd_max.name = f"{var_name}_EDD_Max"
        
        edd_mean = xr.DataArray(annual_mean, coords=[time_coords, da_target.lat, da_target.lon], dims=['time', 'lat', 'lon'])
        edd_mean.name = f"{var_name}_EDD_Mean"
        
        return edd_max, edd_mean

    def calc_trend_1d(self, y):
        if np.isnan(y).all() or len(np.unique(y)) == 1: 
            return np.nan, np.nan
        x = np.arange(len(y))
        slope, _, _, _ = theilslopes(y, x, 0.95)
        _, pval = kendalltau(x, y)
        return slope, pval

    def apply_trend_to_da(self, da):
        slope, pval = xr.apply_ufunc(
            self.calc_trend_1d, da,
            input_core_dims=[['time']],
            output_core_dims=[[], []],
            vectorize=True,
            dask='forbidden'
        )
        return slope, pval

    def run(self):
        print("==========================================================")
        print("Initializing Energy Drought Extraction Engine")
        print("==========================================================")
        
        for prefix, suffix, out_name, var_name in self.vars_config:
            print(f"\n[ Processing: {out_name} ]")
            
            # --- 1. HISTORICAL BASELINE & THRESHOLDS ---
            da_hist = self.get_ensemble_data(prefix, suffix, var_name, 'hist')
            if da_hist is None:
                print(f"   [SKIP] Historical data for {out_name} is incomplete.")
                continue
                
            print("   Calculating historical thresholds (P20 & P40)...")
            p20 = da_hist.quantile(0.20, dim='time').drop_vars('quantile')
            p40 = da_hist.quantile(0.40, dim='time').drop_vars('quantile')
            
            print("   Extracting annual EDF & EDD (Historical)...")
            edf_hist = self.extract_annual_edf(da_hist, out_name)
            edd_max_hist, edd_mean_hist = self.extract_annual_edd_with_pooling(da_hist, p20, p40, out_name)
            
            # Save average historical metrics
            ds_hist_mean = xr.merge([edf_hist.mean('time'), edd_max_hist.mean('time'), edd_mean_hist.mean('time')])
            ds_hist_mean.to_netcdf(os.path.join(self.out_dir, f"Drought_Clim_{out_name}_hist.nc"))

            # --- 2. SSP PROJECTION & CONTINUOUS TREND ANALYSIS ---
            for ssp in self.ssps:
                da_ssp = self.get_ensemble_data(prefix, suffix, var_name, ssp)
                if da_ssp is None: continue
                
                print(f"   Extracting annual EDF & EDD ({ssp.upper()})...")
                edf_ssp = self.extract_annual_edf(da_ssp, out_name)
                edd_max_ssp, edd_mean_ssp = self.extract_annual_edd_with_pooling(da_ssp, p20, p40, out_name)
                
                # Save average SSP metrics
                ds_ssp_mean = xr.merge([edf_ssp.mean('time'), edd_max_ssp.mean('time'), edd_mean_ssp.mean('time')])
                ds_ssp_mean.to_netcdf(os.path.join(self.out_dir, f"Drought_Clim_{out_name}_{ssp}.nc"))

                # Continuous 60-Year Trend Analysis (HIST + SSP)
                print(f"   Calculating continuous 60-year trends (HIST + {ssp.upper()})...")
                
                # EDF Trend
                edf_60yr = xr.concat([edf_hist, edf_ssp], dim='time')
                slope_edf, pval_edf = self.apply_trend_to_da(edf_60yr)
                
                # EDD Max Trend
                edd_max_60yr = xr.concat([edd_max_hist, edd_max_ssp], dim='time')
                slope_edd_max, pval_edd_max = self.apply_trend_to_da(edd_max_60yr)
                
                # EDD Mean Trend
                edd_mean_60yr = xr.concat([edd_mean_hist, edd_mean_ssp], dim='time')
                slope_edd_mean, pval_edd_mean = self.apply_trend_to_da(edd_mean_60yr)
                
                # Assemble final trend dataset
                ds_trend = xr.Dataset({
                    f'{out_name}_EDF_slope': slope_edf, f'{out_name}_EDF_pval': pval_edf,
                    f'{out_name}_EDD_Max_slope': slope_edd_max, f'{out_name}_EDD_Max_pval': pval_edd_max,
                    f'{out_name}_EDD_Mean_slope': slope_edd_mean, f'{out_name}_EDD_Mean_pval': pval_edd_mean
                })
                
                trend_file = os.path.join(self.trend_dir, f"TREND_60YR_Drought_{out_name}_{ssp}.nc")
                ds_trend.to_netcdf(trend_file)
                print(f"   [SAVED] {os.path.basename(trend_file)}")
                
                del da_ssp, edf_ssp, edd_max_ssp, edd_mean_ssp, edf_60yr, edd_max_60yr, edd_mean_60yr, ds_trend
                gc.collect()

            del da_hist, p20, p40, edf_hist, edd_max_hist, edd_mean_hist
            gc.collect()

        print("\n[SUCCESS] Energy drought and trend analysis completed.")

if __name__ == "__main__":
    EnergyDroughtCalculator().run()
