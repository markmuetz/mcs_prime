"""Remakefile to calculate local environments based on MCS track location at varying radii.

Coppy of ctrl/remakefiles/mcs_local_envs.
Modified to include divergence.

Idea is that if you know where an MCS is, you can look at different radii from that to sample
its ERA5 environment. So it is complementary to era5_histograms, which uses actual positions
of MCS masks, but provides information about spatial scales. It can also be used to investigate
the precursor environments (see LifecycleMcsLocalEnv).
"""
from itertools import product
from timeit import default_timer as timer

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from geopy import distance
from mcs_prime import PATHS, McsTracks, PixelData
from remake import Remake, TaskRule
from remake.util import format_path as fmtp

import mcs_prime.mcs_prime_config_util as cu

# slurm_config = {'queue': 'short-serial', 'mem': 64000, 'max_runtime': '10:00:00'}
slurm_config = {'account': 'short4hr', 'queue': 'short-serial-4hr', 'mem': 32000}
mcs_local_envs = Remake(config=dict(slurm=slurm_config, content_checks=False))

YEARS = [2020]
levels = [
    114,  # 850 hPa
    105,  # 700 hPa
    101,  # 600 hPa
]

DIV_ERA5VARS = (
    [f'div_ml{level}' for level in levels] +
    [f'vertically_integrated_div_ml{level}_surf' for level in levels]
)

FMT_PATH_MCS_ENV_COND_REVS_LIFECYCLE_MCS_LOCAL_ENV = (
    PATHS['outdir'] / 'mcs_env_cond_reviews' / 'mcs_local_envs' / '{year}' / '{month:02d}' / 'lifecycle_mcs_local_env_{year}_{month:02d}.{var}.nc'
)

def e5_data_inputs(e5times, var):
    """Generate input paths for all ERA5 variables, base and processed"""
    if var in cu.ERA5VARS + cu.DL_ERA5VARS:
        inputs = {
            f'era5_{t}_{var}': cu.era5_sfc_fmtp(var, t.year, t.month, t.day, t.hour)
            for t in e5times
        }
    elif var in cu.SHEAR_ERA5VARS:
        inputs = {
            f'era5_shear_{t}': fmtp(cu.FMT_PATH_ERA5P_SHEAR, year=t.year, month=t.month, day=t.day, hour=t.hour)
            for t in e5times
        }
    elif var in cu.VIMFD_ERA5VARS:
        inputs = {
            f'era5_vimfd_{t}': fmtp(cu.FMT_PATH_ERA5P_VIMFD, year=t.year, month=t.month, day=t.day, hour=t.hour)
            for t in e5times
        }
    elif var in cu.LAYER_MEANS_ERA5VARS:
        inputs = {
            f'era5_layer_means_{t}': fmtp(cu.FMT_PATH_ERA5P_LAYER_MEANS, year=t.year, month=t.month, day=t.day, hour=t.hour)
            for t in e5times
        }
    elif var in cu.DELTA_ERA5VARS:
        inputs = {
            f'era5_delta_{t}': fmtp(cu.FMT_PATH_ERA5P_DELTA, year=t.year, month=t.month, day=t.day, hour=t.hour)
            for t in e5times
        }
    elif var in DIV_ERA5VARS:
        inputs = {
            f'era5_div_{t}': fmtp(cu.FMT_PATH_ERA5P_DIV, year=t.year, month=t.month, day=t.day, hour=t.hour)
            for t in e5times
        }
    return inputs


def open_e5_data(logger, e5times, inputs):
    """Load the ERA5 data"""
    mcs_times = e5times[:-1] + pd.Timedelta(minutes=30)

    e5paths = [v for k, v in inputs.items() if k.startswith('era5_')]
    print(f'Open ERA5 {e5paths[0].name} - {e5paths[-1].name}')
    e5ds = xr.open_mfdataset(e5paths).sel(latitude=slice(60, -60)).interp(time=mcs_times)

    return e5ds


def get_dist(da, lat, lon):
    """Rotate the distance data to the desired lon, using the correct lat value

    There is some inaccuracy here as it only uses the closes possible lat/lon,
    not the actual lat/lon provided. For an ERA5 grid this means there might be up
    to 0.25/2 = 0.125deg ~ 12km of error.
    """
    lat_idx = cu.find_nearest(da.dist_lat.values, lat)
    lon_idx = cu.find_nearest_circular(da.lon.values, lon)
    return lat_idx, lon_idx, np.roll(da.values[lat_idx], lon_idx, axis=1)


class LifecycleMcsLocalEnv(TaskRule):
    """Capture the env at various radii over the lifecycle of MCSs

    Includes the precursor env. This is captured up to 24hr before MCS init. During this
    time, the lat/lon of MCS init are used. At and after MCS init, the lat/lon of the MCS track are used.
    """

    @staticmethod
    def rule_inputs(year, month, var):
        # Both have one extra value at start/end because I need to interp to half hourly.
        # so that I can calc precursor env.
        start = pd.Timestamp(year, month, 1) - pd.Timedelta(hours=25)
        # to account for latest possible time in tracks dataset (400 hours)
        end = start + pd.DateOffset(months=1) + pd.Timedelta(hours=401)
        e5times = pd.date_range(start, end, freq='H')

        inputs = e5_data_inputs(e5times, var)
        inputs['tracks'] = cu.fmt_mcs_stats_path(year)
        # Already generated by main analysis.
        inputs['dists'] = cu.PATH_LAT_LON_DISTS

        return inputs

    # Don't trash existing output!
    rule_outputs = {'lifecycle_mcs_local_env': FMT_PATH_MCS_ENV_COND_REVS_LIFECYCLE_MCS_LOCAL_ENV}

    var_matrix = {
        'year': YEARS,
        'month': cu.MONTHS,
        'var': DIV_ERA5VARS,
    }
    config = {'slurm': {'mem': 400000, 'partition': 'high-mem', 'max_runtime': '24:00:00'}}

    def rule_run(self):
        # Start from first precursor time.
        start = pd.Timestamp(self.year, self.month, 1) - pd.Timedelta(hours=25)
        # to account for latest possible time in tracks dataset (400 hours)
        end = start + pd.DateOffset(months=1) + pd.Timedelta(hours=401)
        e5times = pd.date_range(start, end, freq='H')
        print(e5times)

        # This is not the most memory-efficient way of doing this.
        # BUT it allows me to load all the data once, and then access
        # it in any order efficiently (because it's all loaded into RAM).
        # Note also that not all values of this are necessarily used.
        # E.g. the last time will only be used if there is a duration=400h
        # MCS (unlikely).
        # Normal trick of interpolating to mcs_times.
        e5ds = open_e5_data(self.logger, e5times, self.inputs)
        print(e5ds)
        var = self.var
        # e5ds can contain multiple vars. Load the one we want.
        print(f'loading {var}')
        e5ds[var].load()

        tracks = McsTracks.open(self.inputs['tracks'], None)
        time = pd.DatetimeIndex(tracks.dstracks.start_basetime)
        dstracks = tracks.dstracks.isel(tracks=(time.month == self.month))
        print(dstracks)

        dists = xr.load_dataarray(self.inputs['dists'])
        print(dists)

        percentiles = [10, 25, 50, 75, 90]
        # Construct output hists dataset.
        coords = {
            'tracks': dstracks.tracks,
            'radius': cu.RADII,
            'times': list(range(-24, 400)),
            'percentile': percentiles,
        }
        data_vars = {}

        coords['percentile'] = percentiles
        blank_mean_data = np.full((len(dstracks.tracks), len(cu.RADII), 424), np.nan)
        blank_percentile_data = np.full((len(dstracks.tracks), len(cu.RADII), len(percentiles), 424), np.nan)

        units = e5ds[var].units
        attrs = {'description': f'mean value of {var} over precursor time and MCS lifecycle', 'units': units}
        data_vars[f'mean_{var}'] = (('tracks', 'radius', 'times'), blank_mean_data, attrs)
        data_vars[f'percentile_{var}'] = (('tracks', 'radius', 'percentile', 'times'), blank_percentile_data)

        dsout = xr.Dataset(
            data_vars=data_vars,
            coords=coords,
        )
        dsout.radius.attrs['units'] = 'km'
        print(dsout)

        # Main loop. For each track, find its lat/lon at each of its
        # times (incl. precursor times, when first lat/lon is used). Then
        # use this to mask out an area based on the different radii. Compute a hist
        # at each time for each track and for each radius, and store this.
        for track_idx, track_id in enumerate(dstracks.tracks.values):
            print(track_id)
            track = dstracks.sel(tracks=track_id)
            duration = track.track_duration.values.item()

            track_start = pd.Timestamp(track.start_basetime.values.item())
            precursor_start = track_start - pd.Timedelta(hours=24)
            track_end = pd.Timestamp(track.base_time.values[duration - 1])
            precursor_times = pd.date_range(precursor_start, track_start - pd.Timedelta(hours=1), freq='H')
            times = np.concatenate([precursor_times, pd.DatetimeIndex(track.base_time.values[:duration])])

            # extended_idx can be used to index e.g. track.meanlat IF it's >= 0
            # < 0 corresponds to precursor period.
            extended_idx = range(-24, duration)
            assert len(times) == len(extended_idx)
            for i, time in zip(extended_idx, times):
                print(time)

                idx = 0 if i < 0 else i

                # If in precursor period, use the first lat/lon vals.
                lat = track.meanlat.values[idx]
                lon = track.meanlon.values[idx]

                lat_idx, lon_idx, dist = get_dist(dists, lat, lon)
                data = e5ds.sel(time=time)[var].values
                for j, r in enumerate(cu.RADII):
                    dist_mask = dist < r
                    # Note, to index the times dim, I need to add 24 to i (starts at -24).
                    masked_data = data[dist_mask]
                    dsout[f'mean_{var}'].values[track_idx, j, i + 24] = np.nanmean(masked_data)
                    dsout[f'percentile_{var}'].values[track_idx, j, :, i + 24] = np.nanpercentile(masked_data, percentiles)
        cu.to_netcdf_tmp_then_copy(dsout, self.outputs['lifecycle_mcs_local_env'])