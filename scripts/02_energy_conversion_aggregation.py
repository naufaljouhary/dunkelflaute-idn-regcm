"""
Project: Capacity Factor (CF) Conversion and Aggregation
Description: 
- Converts bias-corrected climate variables into Wind and Solar PV Capacity Factors.
- Processes 6-hourly data and aggregates to daily means.
- Features memory-efficient spatial tiling to prevent OOM errors.
"""

import xarray as xr
import numpy as np
import os
import sys
import warnings
import shutil
import gc

warnings.filterwarnings("ignore")

class CapacityFactorEngine:
    def __init__(self):
        self.path_in = "../data/processed"
        self.path_out = "../data/processed"
        
        self.tmp_dir = os.path.join(self.path_out, "tmp_cf_chkpt")
        os.makedirs(self.path_out, exist_ok=True)
        os.makedirs(self.tmp_dir, exist_ok=True)
        
        self.models = ['EC-Earth3', 'NorESM']
        self.scenarios = ['hist', 'ssp126', 'ssp245', 'ssp370', 'ssp585']
        
        # Siemens SWT-3.6-130 Power Curve
        self.V_CURVE = np.array([3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 
                                 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0])
        self.P_CURVE = np.array([43.0, 184.0, 421.0, 778.0, 1269.0, 1901.0, 2630.0, 3261.0, 3534.0, 
                                 3593.0, 3600.0, 3600.0, 3600.0, 3600.0, 3600.0, 3600.0, 3600.0, 
                                 3600.0, 3600.0, 3600.0, 3600.0, 3600.0, 3600.0])
        self.P_RATED = 3600.0

    def calc_wind_curve(self, da):
        """Vectorized Power Curve interpolation using xarray ufunc"""
        return xr.apply_ufunc(
            lambda x: np.interp(x, self.V_CURVE, self.P_CURVE, left=0.0, right=0.0) / self.P_RATED,
            da
        )

    def process_wind(self, model, scen):
        print(f"Processing Wind CF: {model} | {scen}")
        
        in_ws100 = os.path.join(self.path_in, f"ws100_{model}_{scen}_corrected.nc")
        out_6h = os.path.join(self.path_out, f"CF_WIND_6H_{model}_{scen}.nc")
        out_1d = os.path.join(self.path_out, f"CF_WIND_1D_{model}_{scen}.nc")
        
        if os.path.exists(out_6h) and os.path.exists(out_1d):
            print("   [SKIP] Wind CF data already exists.")
            return

        ds_ws100 = xr.open_dataset(in_ws100, engine='netcdf4')
        lat_len, lon_len = ds_ws100.dims['lat'], ds_ws100.dims['lon']
        tile_size = 50 
        
        # Output container
        da_cf_6h = xr.full_like(ds_ws100['ws100'], np.nan).load()
        
        for y in range(0, lat_len, tile_size):
            for x in range(0, lon_len, tile_size):
                y_e, x_e = min(y + tile_size, lat_len), min(x + tile_size, lon_len)
                
                # Checkpoint handler
                chkpt_6h = f"{self.tmp_dir}/wind_6h_{model}_{scen}_y{y}_x{x}.nc"
                if os.path.exists(chkpt_6h):
                    with xr.open_dataarray(chkpt_6h) as temp_da:
                        da_cf_6h.values[:, y:y_e, x:x_e] = temp_da.values
                    continue
                
                # Load spatial tile
                tile_ws100 = ds_ws100['ws100'][:, y:y_e, x:x_e].load()
                
                # Calculate CF
                tile_cf = self.calc_wind_curve(tile_ws100)
                
                # Save to memory and checkpoint
                da_cf_6h.values[:, y:y_e, x:x_e] = tile_cf.values
                tile_cf.to_netcdf(chkpt_6h)
                
                del tile_ws100, tile_cf
                gc.collect()

        # Format attributes and save 6-hourly data
        da_cf_6h.name = 'cf_wind'
        da_cf_6h.attrs['units'] = '1'
        da_cf_6h.attrs['long_name'] = '6-Hourly Wind Capacity Factor (Siemens SWT-3.6-130)'
        da_cf_6h.to_netcdf(out_6h)
        print(f"   [SAVED] {os.path.basename(out_6h)}")

        # Daily resampling
        print("   Aggregating to daily means (1D)...")
        da_cf_1d = da_cf_6h.resample(time='1D').mean(keep_attrs=True)
        da_cf_1d.attrs['long_name'] = 'Daily Mean Wind Capacity Factor'
        da_cf_1d.to_netcdf(out_1d)
        print(f"   [SAVED] {os.path.basename(out_1d)}")
        
        ds_ws100.close()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        os.makedirs(self.tmp_dir, exist_ok=True)

    def process_solar(self, model, scen):
        print(f"Processing Solar CF: {model} | {scen}")
        
        in_rsds = os.path.join(self.path_in, f"rsds_{model}_{scen}_corrected.nc")
        in_t2m = os.path.join(self.path_in, f"t2m_{model}_{scen}_corrected.nc")
        in_ws10 = os.path.join(self.path_in, f"ws10_{model}_{scen}_corrected.nc")
        
        out_6h = os.path.join(self.path_out, f"CF_SOLAR_6H_{model}_{scen}.nc")
        out_1d = os.path.join(self.path_out, f"CF_SOLAR_1D_{model}_{scen}.nc")
        
        if os.path.exists(out_6h) and os.path.exists(out_1d):
            print("   [SKIP] Solar CF data already exists.")
            return

        ds_rsds = xr.open_dataset(in_rsds, engine='netcdf4')
        ds_t2m = xr.open_dataset(in_t2m, engine='netcdf4')
        ds_ws10 = xr.open_dataset(in_ws10, engine='netcdf4')
        
        lat_len, lon_len = ds_rsds.dims['lat'], ds_rsds.dims['lon']
        tile_size = 50 
        
        da_cf_6h = xr.full_like(ds_rsds['rsds'], np.nan).load()
        
        for y in range(0, lat_len, tile_size):
            for x in range(0, lon_len, tile_size):
                y_e, x_e = min(y + tile_size, lat_len), min(x + tile_size, lon_len)
                
                chkpt_6h = f"{self.tmp_dir}/solar_6h_{model}_{scen}_y{y}_x{x}.nc"
                if os.path.exists(chkpt_6h):
                    with xr.open_dataarray(chkpt_6h) as temp_da:
                        da_cf_6h.values[:, y:y_e, x:x_e] = temp_da.values
                    continue
                
                t_rsds = ds_rsds['rsds'][:, y:y_e, x:x_e].load()
                t_tas = ds_t2m['t2m'][:, y:y_e, x:x_e].load() - 273.15 # Convert to Celsius
                t_ws10 = ds_ws10['ws10'][:, y:y_e, x:x_e].load()
                
                # TamizhMani thermodynamic model
                T_cell = 4.3 + (0.943 * t_tas) + (0.028 * t_rsds) - (1.528 * t_ws10)
                P_r = 1.0 - 0.005 * (T_cell - 25.0)
                
                # Actual Capacity Factor
                tile_cf = P_r * (t_rsds / 1000.0)
                
                # Physical filters (Nighttime = 0, Max = 1)
                tile_cf = tile_cf.where(t_rsds > 0, 0.0).clip(min=0, max=1)
                
                da_cf_6h.values[:, y:y_e, x:x_e] = tile_cf.values
                tile_cf.to_netcdf(chkpt_6h)
                
                del t_rsds, t_tas, t_ws10, T_cell, P_r, tile_cf
                gc.collect()

        da_cf_6h.name = 'cf_solar'
        da_cf_6h.attrs['units'] = '1'
        da_cf_6h.attrs['long_name'] = '6-Hourly Solar PV Capacity Factor (TamizhMani)'
        da_cf_6h.to_netcdf(out_6h)
        print(f"   [SAVED] {os.path.basename(out_6h)}")

        print("   Aggregating to daily means (1D)...")
        da_cf_1d = da_cf_6h.resample(time='1D').mean(keep_attrs=True)
        da_cf_1d.attrs['long_name'] = 'Daily Mean Solar PV Capacity Factor'
        da_cf_1d.to_netcdf(out_1d)
        print(f"   [SAVED] {os.path.basename(out_1d)}")
        
        ds_rsds.close()
        ds_t2m.close()
        ds_ws10.close()
        
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        os.makedirs(self.tmp_dir, exist_ok=True)

    def run(self):
        print("==========================================================")
        print("Initializing Capacity Factor Conversion Engine")
        print("==========================================================")
        try:
            for model in self.models:
                for scen in self.scenarios:
                    self.process_wind(model, scen)
                    self.process_solar(model, scen)
            print("\n[SUCCESS] Capacity Factor pipeline completed.")
        except Exception as e:
            print("\n[FATAL ERROR] Execution failed:")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    CapacityFactorEngine().run()
