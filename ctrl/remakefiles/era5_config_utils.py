from pathlib import Path
import pickle
import psutil
import shutil

import pandas as pd

from mcs_prime import PATHS

# YEARS = list(range(2000, 2021))
YEARS = [2020]
MONTHS = range(1, 13)
# MONTHS = range(1, 2)

ERA5VARS = ['cape', 'tcwv']
LS_REGIONS = ['all', 'land', 'ocean']

DATES = pd.date_range(f'{YEARS[0]}-01-01', f'{YEARS[-1]}-12-31')
DATE_KEYS = [(y, m, d) for y, m, d in zip(DATES.year, DATES.month, DATES.day)]

PATH_ERA5_MODEL_LEVELS = PATHS['datadir'] / 'ERA5/ERA5_L137_model_levels_table.csv'
PATH_REGRIDDER = (PATHS['outdir'] / 'conditional_era5_histograms' /
                  'regridder' /
                  'bilinear_1200x3600_481x1440_peri.nc')
PATH_ERA5_LAND_SEA_MASK = (PATHS['era5dir'] /
                           'data/invariants/ecmwf-era5_oper_an_sfc_200001010000.lsm.inv.nc')

FMT_PATH_ERA5_ML = (PATHS['era5dir'] /
                    'data/oper/an_ml/{year}/{month:02d}/{day:02d}' /
                    ('ecmwf-era5_oper_an_ml_{year}{month:02d}{day:02d}'
                     '{hour:02d}00.{var}.nc'))
FMT_PATH_ERA5_SFC = (PATHS['era5dir'] /
                     'data/oper/an_sfc/{year}/{month:02d}/{day:02d}' /
                     ('ecmwf-era5_oper_an_sfc_{year}{month:02d}{day:02d}'
                      '{hour:02d}00.{var}.nc'))
FMT_PATH_ERA5P_SHEAR = (PATHS['outdir'] /
                        'era5_processed/{year}/{month:02d}/{day:02d}' /
                        ('ecmwf-era5_oper_an_ml_{year}{month:02d}{day:02d}'
                         '{hour:02d}00.proc_shear.nc'))
FMT_PATH_ERA5P_VIMFD = (PATHS['outdir'] /
                        'era5_processed/{year}/{month:02d}/{day:02d}' /
                        ('ecmwf-era5_oper_an_ml_{year}{month:02d}{day:02d}'
                         '{hour:02d}00.proc_vimfd.nc'))
FMT_PATH_PIXEL_ON_ERA5 = (PATHS['outdir'] / 'mcs_track_pixel_on_era5_grid' /
                          '{year}/{month:02d}/{day:02d}' /
                          'mcstrack_on_era5_grid_{year}{month:02d}{day:02d}{hour:02d}30.nc')
FMT_PATH_ERA5_MEANFIELD = (PATHS['outdir'] / 'conditional_era5_histograms' /
                           '{year}' /
                           'era5_mean_field_{year}_{month:02d}.nc')
FMT_PATH_PRECURSOR_COND_HIST = (PATHS['outdir'] / 'conditional_era5_histograms' /
                                '{year}' /
                                'coretb_precursor{precursor_time}_hourly_hist_{year}_{month:02d}.nc')
FMT_PATH_COND_HIST_HOURLY = (PATHS['outdir'] / 'conditional_era5_histograms' /
                             '{year}' / 'core{core_method}_hourly_hist_{year}_{month:02d}.nc')
FMT_PATH_COND_HIST_GRIDPOINT = (PATHS['outdir'] / 'conditional_era5_histograms' /
                                '{year}' / 'gridpoint_hist_{year}_{month:02d}.nc')
FMT_PATH_COND_HIST_MEANFIELD = (PATHS['outdir'] / 'conditional_era5_histograms' /
                                '{year}' / 'meanfield_hist_{year}_{month:02d}.nc')
FMT_PATH_COMBINED_COND_HIST_GRIDPOINT = (PATHS['outdir'] / 'conditional_era5_histograms' /
                                         '{year}' / 'gridpoint_hist_{year}.nc')


def fmt_mcs_stats_path(year):
    if year == 2000:
        start_date = '20000601'
    else:
        start_date = f'{year}0101'
    return PATHS['statsdir'] / f'mcs_tracks_final_extc_{start_date}.0000_{year + 1}0101.0000.nc'


def fmt_mcs_pixel_path(year, month, day, hour):
    if year == 2000:
        start_date = '20000601'
    else:
        start_date = f'{year}0101'
    # Not all files exist! Do not include those that don't.
    return (PATHS['pixeldir'] /
            f'{start_date}.0000_{year + 1}0101.0000'
            f'/{year}/{month:02d}/{day:02d}' /
            f'mcstrack_{year}{month:02d}{day:02d}_{hour:02d}30.nc')

def gen_pixel_times_for_day(year, month, day):
    start = pd.Timestamp(year, month, day, 0, 30)
    end = start + pd.Timedelta(days=1) - pd.Timedelta(hours=1)
    pixel_times = pd.date_range(start, end, freq='H')
    return pixel_times


class PixelInputsCache:
    """Cache pixel inputs and whether they exist

    It is a slow process working out whether all the pixel files exist or not.
    Cache results so that I don't have to do this in any rule_inputs/rule_outputs,
    which dramatically slows down creating instances of tasks."""
    def __init__(self):
        self.cache_path = Path('.pixel_inputs_cache.pkl')
        if not self.cache_path.exists():
            print('creating pixel inputs cache')
            self.all_pixel_inputs = self.create_cache()
            self.cache_inputs()
        else:
            print('loading pixel inputs cache')
            self.all_pixel_inputs = self.load_cache()

    def create_cache(self):
        all_pixel_inputs = {}
        for (year, month, day) in DATE_KEYS:
            hourly_pixel_times = gen_pixel_times_for_day(year, month, day)
            # Not all files exist! Do not include those that don't.
            pixel_paths = [fmt_mcs_pixel_path(t.year, t.month, t.day, t.hour)
                           for t in hourly_pixel_times]
            pixel_times = []
            pixel_inputs = {}
            for time, path in zip(hourly_pixel_times, pixel_paths):
                if path.exists():
                    pixel_times.append(time)
                    pixel_inputs[f'pixel_{time}'] = path
            all_pixel_inputs[(year, month, day)] = pixel_times, pixel_inputs
        return all_pixel_inputs

    def cache_inputs(self):
        with self.cache_path.open('wb') as fp:
            pickle.dump(self.all_pixel_inputs, fp)

    def load_cache(self):
        with self.cache_path.open('rb') as fp:
            return pickle.load(fp)

pixel_inputs_cache = PixelInputsCache()


def print_mem_usage(format='bytes'):
    factor_map = {
        'bytes': 1,
        'B': 1,
        'kB': 1e3,
        'MB': 1e6,
        'GB': 1e9,
        'TB': 1e12,
    }
    factor = factor_map[format]
    process = psutil.Process()
    mem_usage_bytes = process.memory_info().rss
    print(f'{mem_usage_bytes / factor} {format}')


def gen_era5_times_for_month(year, month, include_precursor_offset=True):
    start = pd.Timestamp(year, month, 1)
    end = start + pd.DateOffset(months=1) - pd.Timedelta(hours=1)
    # Make sure there are enough precursor values.
    if include_precursor_offset and year == 2020 and month == 1:
        start -= pd.Timedelta(hours=6)
    # This is the final year - I need to make sure that
    # the first ERA5 file for 2021 is also made for e.g. CalcERA5Shear.
    if year == 2020 and month == 12:
        end += pd.Timedelta(hours=1)
    e5times = pd.date_range(start, end, freq='H')
    return e5times


def to_netcdf_tmp_then_copy(ds, outpath, encoding=None):
    if encoding is None:
        encoding = {}
    tmpdir = Path('/work/scratch-nopw/mmuetz')
    assert outpath.is_absolute()
    tmppath = tmpdir / Path(*outpath.parts[1:])
    tmppath.parent.mkdir(exist_ok=True, parents=True)

    ds.to_netcdf(tmppath, encoding=encoding)
    shutil.move(tmppath, outpath)

