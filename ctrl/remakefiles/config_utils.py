import pickle
import shutil
from hashlib import sha1
from pathlib import Path

import pandas as pd
import psutil
from mcs_prime import PATHS

# YEARS = list(range(2000, 2021))
YEARS = [2020]
MONTHS = range(1, 13)
# MONTHS = range(1, 2)

RADII = [1, 100, 200, 500, 1000]

ERA5VARS = ['cape', 'tcwv']
SHEAR_ERA5VARS = ['LLS_shear', 'L2M_shear', 'MLS_shear']
VIMFD_ERA5VARS = ['vimfd']
PROC_ERA5VARS = SHEAR_ERA5VARS + VIMFD_ERA5VARS
EXTENDED_ERA5VARS = ERA5VARS + PROC_ERA5VARS

LS_REGIONS = ['all', 'land', 'ocean']

DATES = pd.date_range(f'{YEARS[0]}-01-01', f'{YEARS[-1]}-12-31')
DATE_KEYS = [(y, m, d) for y, m, d in zip(DATES.year, DATES.month, DATES.day)]
DATE_MONTH_KEYS = [(y, m) for y in YEARS for m in MONTHS]

PATH_ERA5_MODEL_LEVELS = PATHS['datadir'] / 'ERA5/ERA5_L137_model_levels_table.csv'
PATH_REGRIDDER = PATHS['outdir'] / 'conditional_era5_histograms' / 'regridder' / 'bilinear_1200x3600_481x1440_peri.nc'
PATH_ERA5_LAND_SEA_MASK = PATHS['era5dir'] / 'data/invariants/ecmwf-era5_oper_an_sfc_200001010000.lsm.inv.nc'
PATH_LAT_LON_DISTS = PATHS['outdir'] / 'mcs_local_envs' / 'lat_lon_distances.nc'

FMT_PATH_ERA5_ML = (
    PATHS['era5dir']
    / 'data/oper/an_ml/{year}/{month:02d}/{day:02d}'
    / ('ecmwf-era5_oper_an_ml_{year}{month:02d}{day:02d}' '{hour:02d}00.{var}.nc')
)
FMT_PATH_ERA51_ML = (
    PATHS['era5dir']
    / 'data/oper/an_ml/{year}/era5.1_{year}_data/{month:02d}/{day:02d}'
    / ('ecmwf-era51_oper_an_ml_{year}{month:02d}{day:02d}' '{hour:02d}00.{var}.nc')
)
FMT_PATH_ERA5_SFC = (
    PATHS['era5dir']
    / 'data/oper/an_sfc/{year}/{month:02d}/{day:02d}'
    / ('ecmwf-era5_oper_an_sfc_{year}{month:02d}{day:02d}' '{hour:02d}00.{var}.nc')
)
FMT_PATH_ERA51_SFC = (
    PATHS['era5dir']
    / 'data/oper/an_sfc/{year}/era5.1_{year}_data/{month:02d}/{day:02d}'
    / ('ecmwf-era51_oper_an_sfc_{year}{month:02d}{day:02d}' '{hour:02d}00.{var}.nc')
)
FMT_PATH_ERA5P_SHEAR = (
    PATHS['outdir']
    / 'era5_processed/{year}/{month:02d}/{day:02d}'
    / ('ecmwf-era5_oper_an_ml_{year}{month:02d}{day:02d}' '{hour:02d}00.proc_shear.nc')
)
FMT_PATH_ERA5P_VIMFD = (
    PATHS['outdir']
    / 'era5_processed/{year}/{month:02d}/{day:02d}'
    / ('ecmwf-era5_oper_an_ml_{year}{month:02d}{day:02d}' '{hour:02d}00.proc_vimfd.nc')
)
FMT_PATH_PIXEL_ON_ERA5 = (
    PATHS['outdir']
    / 'mcs_track_pixel_on_era5_grid'
    / '{year}/{month:02d}/{day:02d}'
    / 'mcstrack_on_era5_grid_{year}{month:02d}{day:02d}{hour:02d}30.nc'
)
FMT_PATH_ERA5_MEANFIELD = (
    PATHS['outdir'] / 'conditional_era5_histograms' / '{year}' / 'era5_mean_field_{year}_{month:02d}.nc'
)
FMT_PATH_PRECURSOR_COND_HIST = (
    PATHS['outdir']
    / 'conditional_era5_histograms'
    / '{year}'
    / 'coretb_precursor{precursor_time}_hourly_hist_{year}_{month:02d}.nc'
)
FMT_PATH_COND_HIST_HOURLY = (
    PATHS['outdir'] / 'conditional_era5_histograms' / '{year}' / 'core{core_method}_hourly_hist_{year}_{month:02d}.nc'
)
FMT_PATH_COND_MCS_LIFECYCLE_HIST_HOURLY = (
    PATHS['outdir'] / 'conditional_era5_histograms' / '{year}' / 'lifecycle_hourly_hist_{year}_{month:02d}.nc'
)
FMT_PATH_COND_HIST_GRIDPOINT = (
    PATHS['outdir'] / 'conditional_era5_histograms' / '{year}' / 'gridpoint_hist_{year}_{month:02d}.nc'
)
FMT_PATH_COND_HIST_MEANFIELD = (
    PATHS['outdir'] / 'conditional_era5_histograms' / '{year}' / 'meanfield_hist_{year}_{month:02d}.nc'
)
FMT_PATH_COMBINED_COND_HIST_GRIDPOINT = (
    PATHS['outdir'] / 'conditional_era5_histograms' / '{year}' / 'gridpoint_hist_{year}.nc'
)
FMT_PATH_CHECK_LAT_LON_DISTS_FIG = (
    PATHS['figdir'] / 'mcs_local_envs' / 'check_figs' / 'lat_lon_distances_{lat}_{lon}_{radius}.png'
)
FMT_PATH_MCS_LOCAL_ENV = (
    PATHS['outdir']
    / 'mcs_local_envs'
    / '{year}'
    / '{month:02d}'
    / 'mcs_local_env_{mode} {year}_{month:02d}_{day:02d}.nc'
)
FMT_PATH_LIFECYCLE_MCS_LOCAL_ENV = (
    PATHS['outdir'] / 'mcs_local_envs' / '{year}' / '{month:02d}' / 'lifecycle_mcs_local_env_{year}_{month:02d}.nc'
)
FMT_PATH_CHECK_MCS_LOCAL_ENV = PATHS['figdir'] / 'mcs_local_envs' / 'check_figs' / 'mcs_local_env_r{radius}km.png'
FMT_PATH_COMBINE_MCS_LOCAL_ENV = (
    PATHS['outdir'] / 'mcs_local_envs' / '{year}' / '{month:02d}' / 'mcs_local_env_{mode}_{year}_{month:02d}.nc'
)


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
    return (
        PATHS['pixeldir'] / f'{start_date}.0000_{year + 1}0101.0000'
        f'/{year}/{month:02d}/{day:02d}' / f'mcstrack_{year}{month:02d}{day:02d}_{hour:02d}30.nc'
    )


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
        hexkey = sha1(str((YEARS, MONTHS)).encode()).hexdigest()
        self.cache_path = Path(f'.pixel_inputs_cache_{hexkey}.pkl')
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
            pixel_paths = [fmt_mcs_pixel_path(t.year, t.month, t.day, t.hour) for t in hourly_pixel_times]
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
    print(f'{mem_usage_bytes / factor:.2f} {format}')


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


def lon_180_to_360(v):
    return np.where(v < 0, v + 360, v)


def lon_360_to_180(v):
    return np.where(v > 180, v - 360, v)


def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


def find_nearest_circular(array, value):
    # Convert array and value to radians
    array_rad = np.radians(array)
    value_rad = np.radians(value)

    # Compute the circular distance between value and each element in array
    diff = np.arctan2(np.sin(value_rad - array_rad), np.cos(value_rad - array_rad))

    # Find the index of the minimum circular distance
    idx = np.abs(diff).argmin()
    return idx


def get_bins(var):
    if var == 'cape':
        bins = np.linspace(0, 5000, 101)
    elif var == 'tcwv':
        bins = np.linspace(0, 100, 101)
    elif var[-5:] == 'shear':
        bins = np.linspace(0, 100, 101)
    elif var == 'vimfd':
        bins = np.linspace(-2e-3, 2e-3, 101)
    hist_mids = (bins[1:] + bins[:-1]) / 2
    return bins, hist_mids
