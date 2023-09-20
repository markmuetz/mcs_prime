import inspect
import socket
import warnings
import pickle
import shutil
import traceback
from hashlib import sha1
from pathlib import Path

from cartopy.util import add_cyclic_point
from IPython.display import clear_output
import numpy as np
import pandas as pd
import psutil
import xarray as xr

import remake
from remake import util


# Define paths on different systems. The most important ones are "jasmin".
ALL_PATHS = {
    "mistakenot": {
        "datadir": Path("/home/markmuetz/Datasets/MCS_PRIME"),
        "statsdir": Path("/home/markmuetz/Datasets/MCS_PRIME/MCS_database/MCS_Global/stats"),
        "pixeldir": Path("/home/markmuetz/Datasets/MCS_PRIME/MCS_database/MCS_Global/mcstracking"),
        "simdir": Path("/home/markmuetz/mirrors/jasmin/gws/nopw/j04/mcsprime/mmuetz/data/UM_sims/"),
        "outdir": Path("/home/markmuetz/MCS_PRIME_output/output"),
        "figdir": Path("/home/markmuetz/MCS_PRIME_output/figs"),
        "dropboxdir": Path("/home/markmuetz/Dropbox/Academic/Projects/MCS_PRIME/Shared/MCS_PRIME_figures"),
    },
    "jasmin": {
        "datadir": Path("/gws/nopw/j04/mcs_prime/mmuetz/data/"),
        "statsdir": Path("/gws/nopw/j04/mcs_prime/mmuetz/data/MCS_Global/stats"),
        "pixeldir": Path("/gws/nopw/j04/mcs_prime/mmuetz/data/MCS_Global/mcstracking"),
        "outdir": Path("/gws/nopw/j04/mcs_prime/mmuetz/data/mcs_prime_output"),
        "figdir": Path("/gws/nopw/j04/mcs_prime/mmuetz/data/mcs_prime_figs"),
        "era5dir": Path("/badc/ecmwf-era5"),
    },
}


def _short_hostname():
    hostname = socket.gethostname()
    if "." in hostname and hostname.split(".")[1] == "jasmin":
        return "jasmin"
    return hostname


hostname = _short_hostname()
if hostname[:4] == "host" or hostname == "jupyter-mmuetz":
    hostname = "jasmin"

if hostname not in ALL_PATHS:
    raise Exception(f"Unknown hostname: {hostname}")

# PATHS is system specific.
PATHS = ALL_PATHS[hostname]
for k, path in PATHS.items():
    if not path.exists():
        warnings.warn(f"Warning: path missing {k}: {path}")

# These are from Zhe Feng's tracking dataset.
# Generated using:
# dict((int(v[0]), v[1].strip()) for v in [l.strip().split(':') for l in tracks.dstracks.track_status.attrs['comments'].split(';')]))
STATUS_DICT = {
    0: "Track stops",
    1: "Simple track continuation",
    2: "This is the bigger cloud in simple merger",
    3: "This is the bigger cloud from a simple split that stops at this time",
    4: "This is the bigger cloud from a split and this cloud continues to the next time",
    5: "This is the bigger cloud from a split that subsequently is the big cloud in a merger",
    13: "This cloud splits at the next time step",
    15: "This cloud is the bigger cloud in a merge that then splits at the next time step",
    16: "This is the bigger cloud in a split that then splits at the next time step",
    18: "Merge-split at same time (big merge, splitter, and big split)",
    21: "This is the smaller cloud in a simple merger",
    24: "This is the bigger cloud of a split that is then the small cloud in a merger",
    31: "This is the smaller cloud in a simple split that stops",
    32: "This is a small split that continues onto the next time step",
    33: "This is a small split that then is the bigger cloud in a merger",
    34: "This is the small cloud in a merger that then splits at the next time step",
    37: "Merge-split at same time (small merge, splitter, big split)",
    44: "This is the smaller cloud in a split that is smaller cloud in a merger at the next time step",
    46: "Merge-split at same time (big merge, splitter, small split)",
    52: "This is the smaller cloud in a split that is smaller cloud in a merger at the next time step",
    65: "Merge-split at same time (smaller merge, splitter, small split)",
}

# Which years will be active for processing/analysis.
# YEARS = list(range(2000, 2021))
YEARS = [2018]
MONTHS = range(1, 13)
# MONTHS = [2]

# Radii for MCS local environments (mcs_local_envs.py).
RADII = [1, 100, 200, 500, 1000]

# Reusable list of ERA5 base and derived variables (internal names).
ERA5VARS = ['cape', 'tcwv']
SHEAR_ERA5VARS = ['shear_0', 'shear_1', 'shear_2', 'shear_3']
VIMFD_ERA5VARS = ['vertically_integrated_moisture_flux_div']
LAYER_MEANS_ERA5VARS = ['RHlow', 'RHmid', 'theta_e_mid']
DELTA_ERA5VARS = ['delta_3h_cape', 'delta_3h_tcwv']
PROC_ERA5VARS = SHEAR_ERA5VARS + VIMFD_ERA5VARS + LAYER_MEANS_ERA5VARS + DELTA_ERA5VARS
EXTENDED_ERA5VARS = ERA5VARS + PROC_ERA5VARS

# Different regions.
LS_REGIONS = ['all', 'land', 'ocean']

# Some tasks work on a year/month/day, others on year/month. Construct suitable time ranges.
DATES = pd.date_range(f'{YEARS[0]}-01-01', f'{YEARS[-1]}-12-31')
DATE_KEYS = [(y, m, d) for y, m, d in zip(DATES.year, DATES.month, DATES.day)]
YEARS_MONTHS = [(y, m) for y in YEARS for m in MONTHS]

# Define static paths. Done in a system-agnostic way.
PATH_ERA5_MODEL_LEVELS = PATHS['datadir'] / 'ERA5/ERA5_L137_model_levels_table.csv'
PATH_REGRIDDER = PATHS['outdir'] / / 'pixel_to_era5_regridder' / 'bilinear_1200x3600_481x1440_peri.nc'
PATH_ERA5_LAND_SEA_MASK = PATHS['era5dir'] / 'data/invariants/ecmwf-era5_oper_an_sfc_200001010000.lsm.inv.nc'
PATH_LAT_LON_DISTS = PATHS['outdir'] / 'mcs_local_envs' / 'lat_lon_distances.nc'

# Dynamic format paths
# ====================
#
# Define dynamic format paths. Done in a system-agnostic way.
# Reusable across different remakefiles because they are a top-level import.
# Each one can be used by using remake.format_path(FMT_PATH, year=2000...)
# Every variable in the format path must be passed in. Extra variables will be silently ignored.

# ERA5 base paths.
FMT_PATH_ERA5_ML = (
    PATHS['era5dir']
    / 'data/oper/an_ml/{year}/{month:02d}/{day:02d}'
    / 'ecmwf-era5_oper_an_ml_{year}{month:02d}{day:02d}{hour:02d}00.{var}.nc'
)
# From (JASMIN): /badc/ecmwf-era5/data/oper/an_ml/2001/00README
#
# ERA5 Model level data - 2000-2006: WARNING
# ==========================================
# ERA5 data for 2000-2006 was found to suffer from statospheric cold biases.
# As such users should make use of the ERA5.1 data for this period.
# Please see the associated dataset catalgoue records in the CEDA Data Catalogue for links to the ERA5.1 datasets.
#
# See for more details: https://www.ecmwf.int/en/elibrary/81149-global-stratospheric-temperature-bias-and-other-stratospheric-aspects-era5-and
FMT_PATH_ERA51_ML = (
    PATHS['era5dir']
    / 'data/oper/an_ml/{year}/era5.1_{year}_data/{month:02d}/{day:02d}'
    / 'ecmwf-era51_oper_an_ml_{year}{month:02d}{day:02d}{hour:02d}00.{var}.nc'
)
FMT_PATH_ERA5_SFC = (
    PATHS['era5dir']
    / 'data/oper/an_sfc/{year}/{month:02d}/{day:02d}'
    / 'ecmwf-era5_oper_an_sfc_{year}{month:02d}{day:02d}{hour:02d}00.{var}.nc'
)
# Bias ONLY affected stratosphere - no change in surface vars.
# FMT_PATH_ERA51_SFC = (
#     PATHS['era5dir']
#     / 'data/oper/an_sfc/{year}/era5.1_{year}_data/{month:02d}/{day:02d}'
#     / 'ecmwf-era51_oper_an_sfc_{year}{month:02d}{day:02d}{hour:02d}00.{var}.nc'
# )

# ERA5 processed paths (ERA5P)
# ============================
# era5_processed
FMT_PATH_ERA5P_SHEAR = (
    PATHS['outdir']
    / 'era5_processed/{year}/{month:02d}/{day:02d}'
    / 'ecmwf-era5_oper_an_ml_{year}{month:02d}{day:02d}{hour:02d}00.proc_shear.nc'
)
FMT_PATH_ERA5P_VIMFD = (
    PATHS['outdir']
    / 'era5_processed/{year}/{month:02d}/{day:02d}'
    / 'ecmwf-era5_oper_an_ml_{year}{month:02d}{day:02d}{hour:02d}00.proc_vimfd.nc'
)
FMT_PATH_ERA5P_LAYER_MEANS = (
    PATHS['outdir']
    / 'era5_processed/{year}/{month:02d}/{day:02d}'
    / 'ecmwf-era5_oper_an_ml_{year}{month:02d}{day:02d}{hour:02d}00.proc_layer_means.nc'
)
FMT_PATH_ERA5P_DELTA = (
    PATHS['outdir']
    / 'era5_processed/{year}/{month:02d}/{day:02d}'
    / 'ecmwf-era5_oper_an_ml_{year}{month:02d}{day:02d}{hour:02d}00.proc_delta.nc'
)
FMT_PATH_ERA5_MEANFIELD = (
    PATHS['outdir'] / 'era5_processed' / '{year}' / 'era5_mean_field_{year}_{month:02d}.nc'
)

# MCS pixel data regridded to ERA5 grid (i.e. coarsened).
# =======================================================
# mcs_track_pixel_on_era5_grid
FMT_PATH_PIXEL_ON_ERA5 = (
    PATHS['outdir']
    / 'mcs_track_pixel_on_era5_grid'
    / '{year}/{month:02d}/{day:02d}'
    / 'mcstrack_on_era5_grid_{year}{month:02d}{day:02d}{hour:02d}30.nc'
)

# Conditional histograms paths.
# =============================
# conditional_era5_histograms
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

# MCS local envs paths.
# =====================
# mcs_local_envs
FMT_PATH_CHECK_LAT_LON_DISTS_FIG = (
    PATHS['figdir'] / 'mcs_local_envs' / 'check_figs' / 'lat_lon_distances_{lat}_{lon}_{radius}.png'
)
FMT_PATH_MCS_LOCAL_ENV = (
    PATHS['outdir']
    / 'mcs_local_envs'
    / '{year}'
    / '{month:02d}'
    / 'mcs_local_env_{mode}_{year}_{month:02d}_{day:02d}.nc'
)
FMT_PATH_LIFECYCLE_MCS_LOCAL_ENV = (
    PATHS['outdir'] / 'mcs_local_envs' / '{year}' / '{month:02d}' / 'lifecycle_mcs_local_env_{year}_{month:02d}.nc'
)
FMT_PATH_CHECK_MCS_LOCAL_ENV = PATHS['figdir'] / 'mcs_local_envs' / 'check_figs' / 'mcs_local_env_r{radius}km.png'
FMT_PATH_COMBINE_MCS_LOCAL_ENV = (
    PATHS['outdir'] / 'mcs_local_envs' / '{year}' / 'monthly_mcs_local_env_{mode}_{year}_{month:02d}.nc'
)

# Path formatting helpers.
# ========================
def fmt_mcs_stats_path(year):
    """Format MCS stats path handling different start date for 2000"""
    if year == 2000:
        start_date = '20000601'
    else:
        start_date = f'{year}0101'
    return PATHS['statsdir'] / f'mcs_tracks_final_extc_{start_date}.0000_{year + 1}0101.0000.nc'


def fmt_mcs_pixel_path(year, month, day, hour):
    """Format MCS pixel path handling different start date for 2000"""
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
    """Generate Pixel times for one day"""
    start = pd.Timestamp(year, month, day, 0, 30)
    end = start + pd.Timedelta(days=1) - pd.Timedelta(hours=1)
    pixel_times = pd.date_range(start, end, freq='H')
    return pixel_times


class PixelInputsCache:
    """Cache pixel inputs and whether they exist

    It is a slow process working out whether all the pixel files exist or not.
    Cache results so that I don't have to do this in any rule_inputs/rule_outputs,
    which dramatically slows down creating instances of tasks.

    Generated for a given choice of YEARS, MONTHS - will be regenerated
    if these changed and hasn't already been generated."""

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


def round_times_to_nearest_second(dstracks, fields):
    """Round times in dstracks.base_time to the nearest second.

    Sometimes the dstracks dataset has minor inaccuracies in the time, e.g.
    '2000-06-01T00:30:00.000013440' (13440 ns). Remove these.

    :param dstracks: xarray.Dataset to convert.
    :param fields: list of fields to convert.
    :return: None
    """

    def remove_time_incaccuracy(t):
        # To make this an array operation, you have to use the ns version of datetime64, like so:
        return (np.round(t.astype(int) / 1e9) * 1e9).astype("datetime64[ns]")

    for field in fields:
        dstracks[field].load()
        tmask = ~np.isnan(dstracks[field].values)
        dstracks[field].values[tmask] = remove_time_incaccuracy(dstracks[field].values[tmask])


def print_mem_usage(format='bytes'):
    """Helper to print Python script memory usage in chosen format"""
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


def gen_era5_times_for_month(year, month):
    """Generate ERA5 times for one month."""
    start = pd.Timestamp(year, month, 1)
    end = start + pd.DateOffset(months=1) - pd.Timedelta(hours=1)
    e5times = pd.date_range(start, end, freq='H')
    return e5times


def to_netcdf_tmp_then_copy(ds, outpath, encoding=None):
    """Helper to write output to a JASMIN scratch dir, then copy to desired location.

    Writing to this disk then copying is *MASSIVELY* faster.
    I think it is to do with writing NetCDF4 data to GWS is slow.

    Also, writes detailed metadata to netcdf.

    See: https://help.jasmin.ac.uk/article/176-storage
    Changed to nopw2 in July 23:
    /work/scratch-nopw[RO from 11July2023]
    """
    if encoding is None:
        encoding = {}
    tmpdir = Path('/work/scratch-nopw2/mmuetz')
    assert outpath.is_absolute()
    tmppath = tmpdir / Path(*outpath.parts[1:])
    tmppath.parent.mkdir(exist_ok=True, parents=True)

    # Add some metadata to the netcdf file.
    stack = next(traceback.walk_stack(None))
    frame = stack[0]
    calling_file = frame.f_globals['__file__']
    calling_obj = frame.f_locals['self']
    calling_obj_doc = calling_obj.__doc__
    calling_class_name = calling_obj.__class__.__name__
    output_path_actual = util.tmp_to_actual_path(outpath)

    nodename = socket.gethostname()
    remake_version = remake.__version__

    metadata_attrs = {
        'created by': f'{calling_file}: {calling_class_name}',
        'calling file source': Path(calling_file).read_text(),
        f'remake version': remake_version,
        f'task': f'{calling_obj}',
        f'task doc': f'{calling_obj_doc}',
        'created on': str(pd.Timestamp.now()),
        'nodename': nodename,
        'hostname': hostname,
        'output path': str(output_path_actual),
    }
    ds.attrs.update(metadata_attrs)

    ds.to_netcdf(tmppath, encoding=encoding)
    shutil.move(tmppath, outpath)


def lon_180_to_360(v):
    """Convert a lon from -180 to 180 format to 0 to 360"""
    return np.where(v < 0, v + 360, v)


def lon_360_to_180(v):
    """Convert a lon from 0 to 360 format to -180 to 180"""
    return np.where(v > 180, v - 360, v)


def find_nearest(array, value):
    """Find the index of the nearest value in an np.array"""
    idx = (np.abs(array - value)).argmin()
    return idx


def find_nearest_circular(array, value):
    """Find the index of the nearest value in a circular np.array"""
    # Convert array and value to radians
    array_rad = np.radians(array)
    value_rad = np.radians(value)

    # Compute the circular distance between value and each element in array
    diff = np.arctan2(np.sin(value_rad - array_rad), np.cos(value_rad - array_rad))

    # Find the index of the minimum circular distance
    idx = np.abs(diff).argmin()
    return idx


def get_bins(var):
    """Bins used for histograms for each of these variables"""
    if var == 'cape':
        bins = np.linspace(0, 5000, 101)
    elif var == 'tcwv':
        bins = np.linspace(0, 100, 101)
    elif var.startswith('shear'):
        bins = np.linspace(0, 100, 101)
    elif var == 'vertically_integrated_moisture_flux_div':
        bins = np.linspace(-2e-3, 2e-3, 101)
    elif var == 'RHlow':
        bins = np.linspace(0, 1, 101)
    elif var == 'RHmid':
        bins = np.linspace(0, 1, 101)
    elif var == 'theta_e_mid':
        bins = np.linspace(200, 400, 101)
    elif var == 'delta_3h_cape':
        bins = np.linspace(-1000, 1000, 101)
    elif var == 'delta_3h_tcwv':
        bins = np.linspace(-100, 100, 101)
    hist_mids = (bins[1:] + bins[:-1]) / 2
    return bins, hist_mids


def load_lsmask(path):
    """Load the land-sea mask from the given path"""
    lsmask = {}
    for lsreg in LS_REGIONS:
        # Build appropriate land-sea mask for region.
        da_lsmask = xr.load_dataarray(path)
        if lsreg == 'all':
            # All ones.
            lsmask['all'] = da_lsmask[0].sel(latitude=slice(60, -60)).values >= 0
        elif lsreg == 'land':
            # LSM has land == 1.
            lsmask['land'] = da_lsmask[0].sel(latitude=slice(60, -60)).values > 0.5
        elif lsreg == 'ocean':
            # LSM has ocean == 0.
            lsmask['ocean'] = da_lsmask[0].sel(latitude=slice(60, -60)).values <= 0.5
        else:
            raise ValueError(f'Unknown region: {lsreg}')
    return lsmask


def xr_add_cyclic_point(da, lon_name='longitude'):
    """Pad data array in longitude dimension.

    * Taken from https://stackoverflow.com/a/60913814/54557
    * Modified to handle data on model levels as well.
    * Use add_cyclic_point to pad input data.
    * Relies on min lon value in da being 0.

    :param da: xr.DataArray with dimensions including longitude
    :param lon_name: name of longitude dimension in da
    :returns: padded copy of da
    """
    lon_idx = da.dims.index(lon_name)
    wrap_data, wrap_lon = add_cyclic_point(da.values, coord=da.longitude, axis=lon_idx)

    # Copy old coords and modify longitude.
    new_coords = {dim: da.coords[dim] for dim in da.dims}
    new_coords[lon_name] = wrap_lon

    # Generate output DataArray with new data but same structure as input.
    out_da = xr.DataArray(data=wrap_data, name=da.name, coords=new_coords, dims=da.dims, attrs=da.attrs)
    return out_da


def gen_region_masks(logger, pixel_on_e5, tracks, core_method='tb'):
    """Core function to generate MCS region masks from Pixel and tracks data

    Uses Zhe Feng's dataset.
    * Pixel data is on ERA5 grid.
    * Can use either tb (brightness T) or precip to define core based on threshold.

    Scheme:
    * Loop over times, getting cloudnumbers of MCSs from tracks dataset.
    * Use these to define 5 mutually exclusive regions based on threshold values.
      * MCS core
      * MCS shield
      * Cloud core
      * Cloud shield
      * env
    * Return a 3D array of these (time, lon lat).
    """
    mcs_core_shield_mask = []
    # Looping over subset of times.
    for i, time in enumerate(pixel_on_e5.time.values):
        pdtime = pd.Timestamp(time)
        time = pdtime.to_pydatetime()
        if i % 24 == 0:
            print(time)

        # Get cloudnumbers (cns) for tracks at given time.
        ts = tracks.tracks_at_time(time)
        # tmask is a 2d mask that spans multiple tracks, getting
        # the cloudnumbers at *one time only*, that can be
        # used to get cloudnumbers.
        tmask = (ts.dstracks.base_time == pdtime).values
        if tmask.sum() == 0:
            logger.info(f'No times matched in tracks DB for {pdtime}')
            cns = np.array([])
        else:
            # Each cloudnumber can be used to link to the corresponding
            # cloud in the pixel data.
            cns = ts.dstracks.cloudnumber.values[tmask]
            # Nicer to have sorted values.
            cns.sort()

        # Tracked MCS shield (N.B. close to Tb < 241K but expanded by precip regions).
        # INCLUDES CONV CORE (if using core_method='tb')
        if len(pixel_on_e5.time.values) == 1:
            # No time dim.
            mcs_core_shield_mask.append(pixel_on_e5.cloudnumber.isin(cns).values)
        else:
            mcs_core_shield_mask.append(pixel_on_e5.cloudnumber[i].isin(cns).values)

    mcs_core_shield_mask = np.array(mcs_core_shield_mask)
    if core_method == 'tb':
        # Convective core Tb < 225K.
        core_mask = pixel_on_e5.tb.values < 225
    elif core_method == 'precip':
        core_mask = pixel_on_e5.precipitation.values > 2  # mm/hr
    # Non-MCS clouds (Tb < 241K). INCLUDES CONV CORE.
    # OPERATOR PRECEDENCE! Brackets are vital here.
    cloud_core_shield_mask = (pixel_on_e5.cloudnumber.values > 0) & ~mcs_core_shield_mask
    # MCS conv core only.
    mcs_core_mask = mcs_core_shield_mask & core_mask
    # Cloud conv core only.
    cloud_core_mask = cloud_core_shield_mask & core_mask
    # Env is everything outside of these two regions.
    env_mask = ~mcs_core_shield_mask & ~cloud_core_shield_mask

    # Remove conv core from shields.
    mcs_shield_mask = mcs_core_shield_mask & ~mcs_core_mask
    cloud_shield_mask = cloud_core_shield_mask & ~cloud_core_mask

    # Verify mutual exclusivity and that all points are covered.
    assert (
        mcs_core_mask.astype(int)
        + mcs_shield_mask.astype(int)
        + cloud_core_mask.astype(int)
        + cloud_shield_mask.astype(int)
        + env_mask.astype(int)
        == 1
    ).all()

    return mcs_core_mask, mcs_shield_mask, cloud_core_mask, cloud_shield_mask, env_mask
