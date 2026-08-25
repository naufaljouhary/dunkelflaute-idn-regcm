"""
Project: Sovereign Complementarity Calculator (HPC Worker)
Description: Menghitung SRCC & CVI lalu menyimpan ke NetCDF di folder Bab 5
Update: ⚡ DUAL-MODE COMPLEMENTARITY: Menghitung 2 pasang pilar (Wind vs Solar 24H) & (Wind vs Solar Siang).
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
        self.cf_dir = "/mgpfs/home/njouhary/TESIS/4_capacity_factor/output/rrtm_ncld1"
        self.out_dir = "/mgpfs/home/njouhary/TESIS/5_complementarity_analysis/rrtm_ncld1"
        os.makedirs(self.out_dir, exist_ok=True)
        
        self.models = ['EC-Earth3', 'NorESM']
        self.scenarios = ['hist', 'ssp126', 'ssp245', 'ssp370', 'ssp585']
        
        # ⚡ THE MENTOR'S TWEAK: Tuple Configuration (Solar_Suffix, Output_Modifier, Display_Name) ⚡
        self.solar_configs = [
            ('', '', 'SOLAR 24H'),                 # Wind vs Solar Standar
            ('_siang', '_siang', 'SOLAR DAYLIGHT') # Wind vs Solar Siang
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
        print("⚙️ STARTING COMPLEMENTARITY CALCULATION (OUTPUT: BAB 5)")
        print("==========================================================")
        
        for sol_suffix, out_mod, display_name in self.solar_configs:
            print(f"\n==========================================================")
            print(f"🔄 PILAR KOMPLEMENTARITAS: WIND VS {display_name}")
            print("==========================================================")
            
            for scen in self.scenarios:
                print(f"\n[ MENGHITUNG SKENARIO: {scen.upper()} ]")
                
                # Load Ensemble
                w_das, s_das = [], []
                for model in self.models:
                    # Angin selalu pakai 1D standar
                    w_path = os.path.join(self.cf_dir, f"CF_WIND_1D_{model}_{scen}.nc")
                    # Surya pakai string dinamis sesuai suffix
                    s_path = os.path.join(self.cf_dir, f"CF_SOLAR_1D_{model}_{scen}{sol_suffix}.nc")
                    
                    if os.path.exists(w_path) and os.path.exists(s_path):
                        with xr.open_dataset(w_path, engine='netcdf4') as dw, xr.open_dataset(s_path, engine='netcdf4') as ds:
                            w_das.append(dw['cf_wind'].load())
                            s_das.append(ds['cf_solar'].load())
                
                if len(w_das) != 2:
                    print(f"   ⏩ {scen.upper()}: Data ensemble tidak lengkap (butuh 2 model). Skip.")
                    continue
                    
                print("   -> Merakit ensemble on-the-fly...")
                w_ens = (w_das[0] + w_das[1]) / 2.0
                s_ens = (s_das[0] + s_das[1]) / 2.0
                w_ens, s_ens = xr.align(w_ens, s_ens, join='inner')

                # --- Kalkulasi ---
                print("   -> Menghitung Climatology & Seasonal SRCC + CVI...")
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
                
                # Naming output dinamis (SRCC_siang_Clim_hist.nc atau SRCC_Clim_hist.nc)
                f_srcc_clim = os.path.join(self.out_dir, f"SRCC{out_mod}_Clim_{scen}.nc")
                f_cvi_clim = os.path.join(self.out_dir, f"CVI{out_mod}_Clim_{scen}.nc")
                f_srcc_seas = os.path.join(self.out_dir, f"SRCC{out_mod}_Season_{scen}.nc")
                f_cvi_seas = os.path.join(self.out_dir, f"CVI{out_mod}_Season_{scen}.nc")
                
                srcc_clim.to_netcdf(f_srcc_clim)
                cvi_clim.to_netcdf(f_cvi_clim)
                da_srcc_seas.to_netcdf(f_srcc_seas)
                da_cvi_seas.to_netcdf(f_cvi_seas)
                
                print(f"   ✅ Sukses diletakkan di direktori 5_complementarity_analysis")
                del w_ens, s_ens, srcc_clim, cvi_clim, srcc_seas, cvi_seas, da_srcc_seas, da_cvi_seas, w_das, s_das
                gc.collect()

if __name__ == "__main__":
    ComplementarityCalculator().run()
