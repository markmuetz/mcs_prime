from pathlib import Path
import shutil

import pandas as pd
import xarray as xr

from remake import Remake, TaskRule
from remake.util import format_path as fmtp

from mcs_prime import PATHS

# slurm_config = {'queue': 'short-serial', 'mem': 64000, 'max_runtime': '20:00:00'}
slurm_config = {'account': 'short4hr', 'queue': 'short-serial-4hr', 'mem': 64000}
extract_east_asia = Remake(config=dict(slurm=slurm_config, content_checks=False))

years = [2004, 2020]
months = range(1, 13)


ERA5VARS = ['cape', 'tcwv']
FMT_PATH_ERA5P_SHEAR = (PATHS['outdir'] /
                        'era5_processed/{year}/{month:02d}/{day:02d}' /
                        ('ecmwf-era5_oper_an_ml_{year}{month:02d}{day:02d}'
                         '{hour:02d}00.proc_shear.nc'))



def to_netcdf_tmp_then_copy(ds, outpath, encoding=None):
    if encoding is None:
        encoding = {}
    tmpdir = Path('/work/scratch-nopw/mmuetz')
    assert outpath.is_absolute()
    tmppath = tmpdir / Path(*outpath.parts[1:])
    tmppath.parent.mkdir(exist_ok=True, parents=True)

    ds.to_netcdf(tmppath, encoding=encoding)
    shutil.move(tmppath, outpath)


def get_e5times(year, month):
    start = pd.Timestamp(year, month, 1)
    end = start + pd.DateOffset(months=1)
    e5times = pd.date_range(start, end, freq='H')
    return e5times


class ERA5ExtractEastAsia(TaskRule):
    def rule_inputs(year, month):
        e5times = get_e5times(year, month)
        e5inputs = {f'era5_{t}_{var}': (PATHS['era5dir'] /
                                        f'data/oper/an_sfc/{t.year}/{t.month:02d}/{t.day:02d}' /
                                        (f'ecmwf-era5_oper_an_sfc_{t.year}{t.month:02d}{t.day:02d}'
                                         f'{t.hour:02d}00.{var}.nc'))
                    for t in e5times
                    for var in ERA5VARS}
        return e5inputs

    def rule_outputs(year, month):
        e5times = get_e5times(year, month)[:-1]
        e5outputs = {f'era5_{t}': (PATHS['outdir'] / 'regional_ERA5_data' / 'east_asia' /
                                   f'{t.year}' / f'{t.month:02d}' / f'{t.day:02d}' /
                                   (f'ecmwf-era5_oper_an_sfc_{t.year}{t.month:02d}{t.day:02d}'
                                    f'{t.hour:02d}30.east_asia.nc'))
                     for t in e5times + pd.Timedelta(minutes=30)}
        return e5outputs

    var_matrix = {
        'year': years,
        'month': months,
    }

    def rule_run(self):
        e5times = get_e5times(self.year, self.month)
        interp_times = e5times[:-1] + pd.Timedelta(minutes=30)
        print(e5times[0], e5times[-1])
        print(interp_times[0], interp_times[-1])
        paths = self.inputs.values()

        # N = 24
        # interp_times = interp_times[:N]
        # paths = list(paths)[:N + 1]

        # Similar to Li 2023 region but slightly larger.
        lat_slice = slice(55, 0)
        lon_slice = slice(95, 165)

        print('load data')
        e5data = (xr.open_mfdataset(paths).sel(latitude=lat_slice, longitude=lon_slice)
                  .interp(time=interp_times).load())
        for t in interp_times:
            print(t)
            outpath = self.outputs[f'era5_{t}']
            to_netcdf_tmp_then_copy(e5data.sel(time=t), outpath)


class ERA5ShearExtractEastAsia(TaskRule):
    def rule_inputs(year, month):
        e5times = get_e5times(year, month)
        e5inputs = {f'era5_{t}': fmtp(FMT_PATH_ERA5P_SHEAR,
                                      year=t.year,
                                      month=t.month,
                                      day=t.day,
                                      hour=t.hour)
                    for t in e5times}
        return e5inputs

    def rule_outputs(year, month):
        e5times = get_e5times(year, month)[:-1]
        e5outputs = {f'era5_{t}': (PATHS['outdir'] / 'regional_ERA5_data' / 'east_asia' /
                                   f'{t.year}' / f'{t.month:02d}' / f'{t.day:02d}' /
                                   (f'ecmwf-era5_oper_an_sfc_{t.year}{t.month:02d}{t.day:02d}'
                                    f'{t.hour:02d}30.shear.east_asia.nc'))
                     for t in e5times + pd.Timedelta(minutes=30)}
        return e5outputs

    var_matrix = {
        'year': years,
        'month': months,
    }

    def rule_run(self):
        e5times = get_e5times(self.year, self.month)
        interp_times = e5times[:-1] + pd.Timedelta(minutes=30)
        print(e5times[0], e5times[-1])
        print(interp_times[0], interp_times[-1])
        paths = self.inputs.values()

        # N = 24
        # interp_times = interp_times[:N]
        # paths = list(paths)[:N + 1]

        # Similar to Li 2023 region but slightly larger.
        lat_slice = slice(55, 0)
        lon_slice = slice(95, 165)

        print('load data')
        e5data = (xr.open_mfdataset(paths, concat_dim='time', combine='nested')
                  .sel(latitude=lat_slice, longitude=lon_slice)
                  .interp(time=interp_times).load())
        for t in interp_times:
            print(t)
            outpath = self.outputs[f'era5_{t}']
            to_netcdf_tmp_then_copy(e5data.sel(time=t), outpath)

