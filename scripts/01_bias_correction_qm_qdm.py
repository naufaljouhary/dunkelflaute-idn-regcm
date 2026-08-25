"""
Project: Bias Correction with Quantile Delta Mapping (QDM)
Description: 
- Implements a 91-day sliding window.
- Automatically splits future projections (36 years) into Block A (24 years) & Block B (24 years) to match historical length.
- Applies macroscopic jittering for rsds (0.1-1.0) and a physical cap at 1200 W/m2.
- Includes a resume filter to skip existing outputs.
- Memory optimization via spatial tiling (30x30) and garbage collection.
"""

import xarray as xr
import numpy as np
import os
import glob
import argparse
import warnings
import gc 
from xclim.sdba import QuantileDeltaMapping, Grouper

warnings.filterwarnings("ignore")

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('target_type', type=str)
    parser.add_argument('model_name', type=str)
    parser.add_argument('--in_dir', default=os.getenv('INPUT_DIR', "../data/raw"))
    parser.add_argument('--out_dir', default=os.getenv('OUTPUT_DIR', "../data/processed"))
    return parser.parse_args()

def fix_metadata(ds):
    rename_map = {}
    for d in list(ds.dims) + list(ds.coords):
        if d in ['valid_time', 'Time', 't']: rename_map[d] = 'time'
        if d in ['jx', 'longitude']: rename_map[d] = 'lon'
        if d in ['iy', 'latitude']: rename_map[d] = 'lat'
        
    if rename_map: 
        ds = ds.rename(rename_map)
        
    parasites = ['height', 'crs', 'expver', 'xlon', 'ylat', 'rlon', 'rlat', 'time_bnds', 'lat_bnds', 'lon_bnds']
    ds = ds.drop_vars(parasites, errors='ignore')
    
    keep = ['time', 'lat', 'lon']
    drop = [c for c in ds.coords if c not in keep]
    ds = ds.drop_vars(drop, errors='ignore')
    return ds.squeeze()

class BiasCorrectionQDM:
    def __init__(self, args):
        self.target_type, self.model_name = args.target_type, args.model_name
        self.in_dir, self.out_dir = args.in_dir, args.out_dir
        self.tmp_dir = os.path.join(self.out_dir, f"tmp_{self.target_type}_{self.model_name}")
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(self.tmp_dir, exist_ok=True)
        
        self.var_map = {
            'ws10':  {'comps': ['uas', 'vas'], 'unit': 'm/s'},
            'ws100': {'comps': ['ua100m', 'va100m'], 'unit': 'm/s'},
            'rsds':  {'comps': ['rsds'], 'unit': 'W/m^2'},
            't2m':   {'comps': ['tas'], 'unit': 'degree_Celsius'}
        }
        self.scenarios = ['hist', 'ssp126', 'ssp245', 'ssp370', 'ssp585']

    def apply_jitter(self, da):
        if self.target_type == 'rsds':
            threshold = 1.0 
            noise = xr.DataArray(
                np.random.uniform(0.1, 1.0, da.shape), 
                dims=da.dims, coords=da.coords
            )
        else:
            threshold = 0.001 
            noise = xr.DataArray(
                np.random.uniform(1e-6, 1e-4, da.shape), 
                dims=da.dims, coords=da.coords
            )
            
        out = xr.where(da <= threshold, noise, da)
        out.name = self.target_type
        out.attrs = da.attrs.copy()
        return out

    def robust_load(self, var_name, model_filter):
        files = sorted(glob.glob(f"{self.in_dir}/{var_name}_*{model_filter}*.nc"))
        if not files: return None
        ds = xr.open_mfdataset(files, combine='nested', concat_dim='time', preprocess=fix_metadata)
        
        if hasattr(ds.time.dt, 'calendar') and ds.time.dt.calendar != 'noleap':
            ds = ds.convert_calendar('noleap', align_on='random')
            
        da = ds[var_name]
        da.encoding.clear()
        
        if self.target_type == 'rsds': 
            da = da.clip(min=0.0)
        elif self.target_type == 't2m':
            if float(da.isel(time=0).mean().compute().item()) > 200.0: 
                da = da - 273.15
                
        da.attrs['units'] = self.var_map[self.target_type]['unit']
        da.name = self.target_type 
        return da.transpose('time', 'lat', 'lon')

    def get_data(self, components, model_filter):
        das = [self.robust_load(c, model_filter) for c in components]
        if any(d is None for d in das): return None
        
        if len(das) == 2:
            res = np.sqrt(das[0]**2 + das[1]**2)
            res.encoding.clear()
            res.attrs['units'] = self.var_map[self.target_type]['unit']
            res.name = self.target_type
            return res
            
        return das[0]

    def process(self):
        print(f"Initializing QDM | Target: {self.target_type} | Model: {self.model_name}")
        
        # Input filtering and slicing
        da_targets = {}
        for s in self.scenarios:
            data_scen = self.get_data(self.var_map[self.target_type]['comps'], f"{self.model_name}_{s}")
            if data_scen is None:
                print(f"[WARNING] Input {s.upper()} is missing or incomplete. Skipping.")
                continue
                
            if s == 'hist':
                final_out_path = os.path.join(self.out_dir, f"{self.target_type}_{self.model_name}_{s}_corrected.nc")
                if os.path.exists(final_out_path):
                    print(f"[SKIP] Final output for {s.upper()} already exists.")
                else:
                    da_targets[s] = data_scen
            else:
                # Split future projections into two blocks to match historical length
                out_A = os.path.join(self.out_dir, f"{self.target_type}_{self.model_name}_{s}_blockA_corrected.nc")
                out_B = os.path.join(self.out_dir, f"{self.target_type}_{self.model_name}_{s}_blockB_corrected.nc")
                
                if os.path.exists(out_A):
                    print(f"[SKIP] {s.upper()} Block A (2015-2038) already processed.")
                else:
                    da_targets[f"{s}_blockA"] = data_scen.sel(time=slice('2015-01-01', '2038-12-31'))
                    
                if os.path.exists(out_B):
                    print(f"[SKIP] {s.upper()} Block B (2027-2050) already processed.")
                else:
                    da_targets[f"{s}_blockB"] = data_scen.sel(time=slice('2027-01-01', '2050-12-31'))

        if not da_targets:
            print("[DONE] All blocks processed successfully.")
            return
            
        print(f"Blocks scheduled for adjustment: {list(da_targets.keys())}")

        # Load training data (Historical baseline)
        da_ref = self.get_data(self.var_map[self.target_type]['comps'], 'ERA5_hist')
        da_mod_hist = self.get_data(self.var_map[self.target_type]['comps'], f"{self.model_name}_hist")
        
        if da_ref is None or da_mod_hist is None:
            print("[FATAL ERROR] Missing ERA5_hist or GCM_hist data. Cannot train QDM.")
            return
        
        da_ref = da_ref.isel(time=slice(0, len(da_mod_hist.time)))
        da_ref['time'] = da_mod_hist.time.values
        
        tile_size, lat_len, lon_len = 30, da_ref.shape[1], da_ref.shape[2]
        kind = '+' if self.target_type == 't2m' else '*'
        
        for y in range(0, lat_len, tile_size):
            for x in range(0, lon_len, tile_size):
                y_e, x_e = min(y + tile_size, lat_len), min(x + tile_size, lon_len)
                
                if all(os.path.exists(f"{self.tmp_dir}/tile_{s}_y{y}_x{x}.nc") for s in da_targets.keys()): 
                    continue
                
                ref_t = da_ref[:, y:y_e, x:x_e].load()
                mod_h_t = da_mod_hist[:, y:y_e, x:x_e].load() 
                
                if int(ref_t.count()) == 0:
                    for s in da_targets.keys():
                        dummy = xr.full_like(da_targets[s][:, y:y_e, x:x_e], np.nan, dtype=np.float32)
                        dummy.name = self.target_type
                        dummy.attrs['units'] = self.var_map[self.target_type]['unit']
                        dummy.encoding.clear() 
                        dummy.transpose('time', 'lat', 'lon').to_netcdf(f"{self.tmp_dir}/tile_{s}_y{y}_x{x}.nc")
                    del ref_t, mod_h_t
                    gc.collect() 
                    continue

                if kind == '*':
                    ref_t = self.apply_jitter(ref_t)
                    mod_h_t = self.apply_jitter(mod_h_t)

                qdm = QuantileDeltaMapping.train(
                    ref_t, 
                    mod_h_t, 
                    nquantiles=100, 
                    group=Grouper("time.dayofyear", window=91), 
                    kind=kind
                )
                
                for s in da_targets.keys():
                    mod_f_t = da_targets[s][:, y:y_e, x:x_e].load()
                    if kind == '*': mod_f_t = self.apply_jitter(mod_f_t)
                        
                    adj = qdm.adjust(mod_f_t).transpose('time', 'lat', 'lon')
                    adj.name = self.target_type
                    adj.attrs['units'] = self.var_map[self.target_type]['unit']
                    adj.encoding.clear() 
                    adj.to_netcdf(f"{self.tmp_dir}/tile_{s}_y{y}_x{x}.nc")
                
                del ref_t, mod_h_t, qdm
                gc.collect() 
        
        print("--> Assembly Phase: Reconstructing final dataset...")
        for s in da_targets.keys():
            da_final = xr.full_like(da_targets[s], np.nan, dtype=np.float32).load()
            for y in range(0, lat_len, tile_size):
                for x in range(0, lon_len, tile_size):
                    with xr.open_dataarray(f"{self.tmp_dir}/tile_{s}_y{y}_x{x}.nc") as da_t:
                        da_final.values[:, y:min(y+tile_size, lat_len), x:min(x+tile_size, lon_len)] = da_t.transpose('time', 'lat', 'lon').values
            
            if self.target_type == 'rsds': 
                da_final = da_final.clip(min=0.0, max=1200.0)
                
            da_final.name = self.target_type
            da_final.attrs['units'] = self.var_map[self.target_type]['unit']
            da_final.attrs.pop('coordinates', None)
            da_final.attrs.pop('grid_mapping', None)
            
            da_final['lat'].attrs = {'standard_name': 'latitude', 'units': 'degrees_north', 'axis': 'Y'}
            da_final['lon'].attrs = {'standard_name': 'longitude', 'units': 'degrees_east', 'axis': 'X'}
            da_final['time'].attrs = {'standard_name': 'time', 'axis': 'T'}
            
            da_final.encoding.clear() 
            
            out_path = os.path.join(self.out_dir, f"{self.target_type}_{self.model_name}_{s}_corrected.nc")
            if os.path.exists(out_path): os.remove(out_path)
            da_final.to_netcdf(out_path)
            print(f"[SUCCESS] Assembled output saved to: {out_path}")

if __name__ == "__main__": 
    BiasCorrectionQDM(get_config()).process()
