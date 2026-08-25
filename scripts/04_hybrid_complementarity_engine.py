"""
Project: Hybrid Complementarity Engine
Description: 
- Calculates Spearman's Rank Correlation Coefficient (SRCC) and Composite Variability Index (CVI).
- Evaluates complementarity for both standard 24-hour solar PV and daylight-only solar PV against wind power.
"""

import xarray as xr
import numpy as np
import os
import warnings
from scipy.stats import spearmanr
import gc

warnings.filterwarnings("ignore")

class ComplementarityCalculator:
    def __init__(self):
        self.cf_dir = "../data/processed"
        self.out_dir = "../data/processed"
        os.makedirs(self.out_dir, exist_ok=True)
        
        self.models = ['EC-Earth3', 'NorESM']
        self.scenarios = ['hist', 'ssp126', 'ssp245', 'ssp370', 'ssp585']
        
        # Configuration: (Solar_Suffix, Output_Modifier, Display_Name)
        self.solar_configs = [
            ('', '', 'SOLAR 24H'),                 # Wind vs Standard Solar
            ('_siang', '_siang', 'SOLAR DAYLIGHT') # Wind vs Daylight Solar
        ]

    def calculate_srcc(self, da_wind, da_solar):
        def _spearman_1d(w, s):
            if np.isnan(w).all() or np.isnan(s).all(): return np.nan
            if np.std(w) == 0 or np.std(s) == 0: return np.nan
            return spearmanr(w, s)[0]

        return xr.apply_ufunc(
            _spearman_1d, da_wind, da_solar,
            input_core_dims=[['time'], ['time']],
            vectorize=True, dask='forbidden', output_dtypes=[float]
        )

    def calculate_cvi(self, da_wind, da_solar):
        std_sum = (da_wind + da_solar).std(dim='time')
        sum_std = da_wind.std(dim='time') + da_solar.std(dim='time')
        sum_std = sum_std.where(sum_std > 0, np.nan)
        return std_sum / sum_std

    def run(self):
        print("==========================================================")
        print("Initializing Complementarity Calculation")
        print("==========================================================")
        
        for sol_suffix, out_mod, display_name in self.solar_configs:
            print(f"\n==========================================================")
            print(f"Evaluating Complementarity: WIND VS {display_name}")
            print("==========================================================")
            
            for scen in self.scenarios:
                print(f"\n[ Processing Scenario: {scen.upper()} ]")
                
                # Load Ensemble
                w_das, s_das = [], []
                for model in self.models:
                    w_path = os.path.join(self.cf_dir, f"CF_WIND_1D_{model}_{scen}.nc")
                    s_path = os.path.join(self.cf_dir, f"CF_SOLAR_1D_{model}_{scen}{sol_suffix}.nc")
                    
                    if os.path.exists(w_path) and os.path.exists(s_path):
                        with xr.open_dataset(w_path, engine='netcdf4') as dw, xr.open_dataset(s_path, engine='netcdf4') as ds:
                            w_das.append(dw['cf_wind'].load())
                            s_das.append(ds['cf_solar'].load())
                
                if len(w_das) != 2:
                    print(f"   [SKIP] Incomplete ensemble data for {scen.upper()}.")
                    continue
                    
                print("   -> Assembling ensemble on-the-fly...")
                w_ens = (w_das[0] + w_das[1]) / 2.0
                s_ens = (s_das[0] + s_das[1]) / 2.0
                w_ens, s_ens = xr.align(w_ens, s_ens, join='inner')

                # --- Calculation ---
                print("   -> Calculating Climatological & Seasonal SRCC and CVI...")
                srcc_clim = self.calculate_srcc(w_ens, s_ens)
                cvi_clim = self.calculate_cvi(w_ens, s_ens)
                
                srcc_seas = []
                cvi_seas = []
                for season in ['DJF', 'MAM', 'JJA', 'SON']:
                    w_s = w_ens.sel(time=w_ens.time.dt.season == season)
                    s_s = s_ens.sel(time=s_ens.time.dt.season == season)
                    srcc_seas.append(self.calculate_srcc(w_s, s_s).assign_coords(season=season))
                    cvi_seas.append(self.calculate_cvi(w_s, s_s).assign_coords(season=season))

                # --- Save ---
                srcc_clim.name, cvi_clim.name = 'srcc', 'cvi'
                da_srcc_seas = xr.concat(srcc_seas, dim='season')
                da_cvi_seas = xr.concat(cvi_seas, dim='season')
                da_srcc_seas.name = 'srcc'
                da_cvi_seas.name = 'cvi'
                
                f_srcc_clim = os.path.join(self.out_dir, f"SRCC{out_mod}_Clim_{scen}.nc")
                f_cvi_clim = os.path.join(self.out_dir, f"CVI{out_mod}_Clim_{scen}.nc")
                f_srcc_seas = os.path.join(self.out_dir, f"SRCC{out_mod}_Season_{scen}.nc")
                f_cvi_seas = os.path.join(self.out_dir, f"CVI{out_mod}_Season_{scen}.nc")
                
                srcc_clim.to_netcdf(f_srcc_clim)
                cvi_clim.to_netcdf(f_cvi_clim)
                da_srcc_seas.to_netcdf(f_srcc_seas)
                da_cvi_seas.to_netcdf(f_cvi_seas)
                
                print("   [SAVED] Complementarity metrics successfully generated.")
                del w_ens, s_ens, srcc_clim, cvi_clim, srcc_seas, cvi_seas, da_srcc_seas, da_cvi_seas, w_das, s_das
                gc.collect()

if __name__ == "__main__":
    ComplementarityCalculator().run()
