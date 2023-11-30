from pathlib import Path
import shutil

import pandas as pd
import xarray as xr

from remake import Remake, TaskRule
from remake.util import format_path as fmtp

from mcs_prime import mcs_prime_config_util as cu

# slurm_config = {'queue': 'short-serial', 'mem': 64000, 'max_runtime': '20:00:00'}
slurm_config = {'account': 'short4hr', 'queue': 'short-serial-4hr', 'mem': 64000}
extract_east_asia = Remake(config=dict(slurm=slurm_config, content_checks=False))

case_dates = [
    pd.Timestamp(2019, 4, 20),
    pd.Timestamp(2019, 8, 25),
]

case_dates_hourly = [
    pd.date_range(d - pd.Timedelta(days=2), d + pd.Timedelta(days=2), freq='H')
    for d in case_dates
]


SFC_ERA5VARS = ['cape', 'tcwv']
ML_ERA5VARS = ['u', 'v', 't', 'q', 'lnsp', 'z']


class ERA5SfcExtractHKregion(TaskRule):
    @staticmethod
    def rule_inputs(case):
        e5times = case_dates_hourly[case]
        e5inputs = {f'era5_{t}_{var}': cu.era5_sfc_fmtp(var, t.year, t.month, t.day, t.hour)
                    for t in e5times
                    for var in SFC_ERA5VARS}
        return e5inputs

    @staticmethod
    def rule_outputs(case):
        e5times = case_dates_hourly[case]
        e5outputs = {f'era5_{t}_{var}': (cu.PATHS['outdir'] / 'regional_ERA5_data' / 'HongKong' /
                                         f'{t.year}' / f'{t.month:02d}' / f'{t.day:02d}' /
                                         (f'ecmwf-era5_oper_an_sfc_{t.year}{t.month:02d}{t.day:02d}'
                                          f'{t.hour:02d}00.hong_kong.{var}.nc'))
                     for t in e5times
                     for var in SFC_ERA5VARS}
        return e5outputs

    var_matrix = {
        'case': [0, 1]
    }

    def rule_run(self):
        e5times = case_dates_hourly[self.case]
        hk_lat = 22.33
        hk_lon = 114.16

        paths = self.inputs.values()

        lat_slice = slice(hk_lat + 10, hk_lat - 10)
        lon_slice = slice(hk_lon - 10, hk_lon + 10)

        print('open data')
        for t in e5times:
            for var in SFC_ERA5VARS:
                print(t, var)
                path = self.inputs[f'era5_{t}_{var}']
                print(path)
                e5data = (xr.open_dataset(path).sel(latitude=lat_slice, longitude=lon_slice)
                          .load())
                print(e5data)
                outpath = self.outputs[f'era5_{t}_{var}']
                print(outpath)
                cu.to_netcdf_tmp_then_copy(e5data, outpath)


class ERA5MlExtractHKregion(TaskRule):
    @staticmethod
    def rule_inputs(case):
        e5times = case_dates_hourly[case]
        e5inputs = {f'era5_{t}_{var}': cu.era5_ml_fmtp(var, t.year, t.month, t.day, t.hour)
                    for t in e5times
                    for var in ML_ERA5VARS}
        return e5inputs

    @staticmethod
    def rule_outputs(case):
        e5times = case_dates_hourly[case]
        e5outputs = {f'era5_{t}_{var}': (cu.PATHS['outdir'] / 'regional_ERA5_data' / 'HongKong' /
                                         f'{t.year}' / f'{t.month:02d}' / f'{t.day:02d}' /
                                         (f'ecmwf-era5_oper_an_ml_{t.year}{t.month:02d}{t.day:02d}'
                                          f'{t.hour:02d}00.hong_kong.{var}.nc'))
                     for t in e5times
                     for var in ML_ERA5VARS}
        return e5outputs

    var_matrix = {
        'case': [0, 1]
    }

    def rule_run(self):
        e5times = case_dates_hourly[self.case]
        hk_lat = 22.33
        hk_lon = 114.16

        paths = self.inputs.values()

        lat_slice = slice(hk_lat + 10, hk_lat - 10)
        lon_slice = slice(hk_lon - 10, hk_lon + 10)

        print('open data')
        for t in e5times:
            for var in ML_ERA5VARS:
                print(t, var)
                path = self.inputs[f'era5_{t}_{var}']
                print(path)
                e5data = (xr.open_dataset(path).sel(latitude=lat_slice, longitude=lon_slice)
                          .load())
                print(e5data)
                outpath = self.outputs[f'era5_{t}_{var}']
                print(outpath)
                cu.to_netcdf_tmp_then_copy(e5data, outpath)
