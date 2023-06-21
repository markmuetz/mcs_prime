import datetime as dt
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

import config_utils as cu

# slurm_config = {'queue': 'short-serial', 'mem': 64000, 'max_runtime': '10:00:00'}
slurm_config = {'account': 'short4hr', 'queue': 'short-serial-4hr', 'mem': 32000}
mcs_local_envs = Remake(config=dict(slurm=slurm_config, content_checks=False))

TODOS = '''
TODOS
* Make sure filenames are consistent
* Make sure variables names are sensible/consistent
* Docstrings for all fns, classes
* Validate all data
* Consistent attrs for all created .nc files
* Units on data vars etc.
'''
print(TODOS)


class GenLatLonDistance(TaskRule):
    rule_inputs = {'cape': fmtp(cu.FMT_PATH_ERA5_SFC, year=2020, month=1, day=1, hour=0, var='cape')}
    rule_outputs = {'dists': cu.PATH_LAT_LON_DISTS}

    def rule_run(self):
        e5cape = xr.open_dataarray(self.inputs['cape']).sel(latitude=slice(60, -60)).isel(time=0)
        lats = e5cape.latitude.values
        lons = e5cape.longitude.values
        lat_dists = []

        # For each lat, calc the distance from a point to each other point over domain.
        # These are done at 0deg lon, as they can be rotated to any abitrary lon easily.
        for lat in lats:
            print(lat)
            dists_flat = []
            for lat_lon in product(lats, lons):
                # Use great_circle because it is a lot faster than
                # distance.distance. Just calc at 0deg lon.
                dists_flat.append(distance.great_circle(lat_lon, (lat, 0)).km)
            dists = np.array(dists_flat).reshape(len(lats), len(lons))
            lat_dists.append((lat, dists))

        dist_lats = [l for l, d in lat_dists]
        da = xr.DataArray(
            data=np.array([d for l, d in lat_dists]),
            dims=['dist_lat', 'lat', 'lon'],
            coords=dict(
                dist_lat=dist_lats,
                lat=lats,
                lon=lons,
            ),
        )
        da.to_netcdf(self.outputs['dists'])


def get_dist(da, lat, lon):
    lat_idx = cu.find_nearest(da.dist_lat.values, lat)
    lon_idx = cu.find_nearest_circular(da.lon.values, lon)
    return lat_idx, lon_idx, np.roll(da.values[lat_idx], lon_idx, axis=1)


class CheckLatLonDistance(TaskRule):
    rule_inputs = {'dists': cu.PATH_LAT_LON_DISTS}
    rule_outputs = {'fig': cu.FMT_PATH_CHECK_LAT_LON_DISTS_FIG}

    var_matrix = {
        ('lat', 'lon', 'radius'): [
            (0, 0, 200),
            (10, 20, 500),
            (30, 350, 1000),
            (-40, 360, 500),
        ]
    }

    def rule_run(self):
        dists = xr.open_dataarray(self.inputs['dists'])
        lats = dists.lat.values
        lons = dists.lon.values
        lat = self.lat
        lon = self.lon

        lat_idx, lon_idx, dist = get_dist(dists, lat, lon)
        fig, ax = plt.subplots(
            1,
            1,
            subplot_kw=dict(projection=ccrs.PlateCarree()),
        )

        ax.set_title(f'{lat}, {lon}, {self.radius}')
        ax.contourf(lons, lats, dist < self.radius)
        ax.coastlines()
        ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

        plt.savefig(self.outputs['fig'])


class McsLocalEnv(TaskRule):
    @staticmethod
    def rule_inputs(year, month, day, mode):
        start = dt.datetime(year, month, day)
        # Note there are 25 of these so I can get ERA5 data on the hour either side
        # of MCS dataset data (on the half hour).
        e5times = pd.date_range(start, start + dt.timedelta(days=1), freq='H')

        inputs = {
            f'era5_{t}_{var}': fmtp(cu.FMT_PATH_ERA5_SFC, year=t.year, month=t.month, day=t.day, hour=t.hour, var=var)
            for t in e5times
            for var in cu.ERA5VARS
        }

        inputs['tracks'] = cu.fmt_mcs_stats_path(year)
        inputs['dists'] = cu.PATH_LAT_LON_DISTS

        return inputs

    rule_outputs = {'mcs_local_env': cu.FMT_PATH_MCS_LOCAL_ENV}

    var_matrix = {
        ('year', 'month', 'day'): cu.DATE_KEYS,
        'mode': ['init', 'lifecycle'],
    }

    def rule_run(self):
        tracks = McsTracks.open(self.inputs['tracks'], None)

        dists = xr.load_dataarray(self.inputs['dists'])

        start = dt.datetime(self.year, self.month, self.day)
        # Note there are 25 of these so I can get ERA5 data on the hour either side
        # of MCS dataset data (on the half hour).
        e5times = pd.date_range(start, start + dt.timedelta(days=1), freq='H')

        mcs_start = dt.datetime(self.year, self.month, self.day, 0, 30)
        # 24 of these on the half hour.
        mcs_times = pd.date_range(mcs_start, mcs_start + dt.timedelta(hours=23), freq='H')

        e5paths = [self.inputs[f'era5_{t}_{v}'] for t in e5times for v in cu.ERA5VARS]
        # Interp to MCS times (on the half hour).
        e5ds = xr.open_mfdataset(e5paths).sel(latitude=slice(60, -60)).interp(time=mcs_times).load()

        data_vars = {}
        blank_data = np.zeros((1, len(cu.RADII), len(e5ds.latitude), len(e5ds.longitude)))
        for v in cu.ERA5VARS:
            # Give both arrayes a time dimention with the mean MCS time for easier combining of files.
            data_vars[f'{v}'] = (('time', 'latitude', 'longitude'), e5ds[v].mean(dim='time').values[None, :, :])
            data_vars[f'mcs_local_{v}'] = (('time', 'radius', 'latitude', 'longitude'), blank_data.copy())
        data_vars['dist_mask_sum'] = (('time', 'radius', 'latitude', 'longitude'), blank_data.copy().astype(int))

        dsout = xr.Dataset(
            data_vars=data_vars,
            coords=dict(
                time=[mcs_times.mean()],
                radius=cu.RADII,
                latitude=e5ds.latitude,
                longitude=e5ds.longitude,
            ),
        )
        dsout.radius.attrs['units'] = 'km'
        print(dsout)

        mcs_local_vars = {}
        for v in cu.ERA5VARS:
            mcs_local_vars[v] = []

        for pdtime in mcs_times:
            print(pdtime)
            nptime = pdtime.to_numpy()

            if self.mode == 'init':
                time_mask = tracks.dstracks.base_time.values[:, 0] == nptime
                mcs_lats = tracks.dstracks.meanlat.values[:, 0][time_mask]
                mcs_lons = tracks.dstracks.meanlon.values[:, 0][time_mask]
            elif self.mode == 'lifecycle':
                time_mask = tracks.dstracks.base_time.values == nptime
                mcs_lats = tracks.dstracks.meanlat.values[time_mask]
                mcs_lons = tracks.dstracks.meanlon.values[time_mask]

            e5data = e5ds.sel(time=pdtime)

            mcs_local_mask = np.zeros((len(cu.RADII), len(e5ds.latitude), len(e5ds.longitude)), dtype=bool)

            for lat, lon in zip(mcs_lats, mcs_lons):
                lat_idx, lon_idx, dist = get_dist(dists, lat, lon)
                for i, r in enumerate(cu.RADII):
                    dist_mask = dist < r
                    mcs_local_mask[i, :, :] |= dist_mask

            dsout.dist_mask_sum[0, :, :, :].values += mcs_local_mask.astype(int)

            for v in cu.ERA5VARS:
                # Broadcast variable into correct shape for applying masks at different radii.
                mcs_local_var = np.ones(len(cu.RADII))[:, None, None] * np.copy(e5data[v].values)[None, :, :]
                mcs_local_var[~mcs_local_mask] = np.nan
                mcs_local_vars[v].append(mcs_local_var)

        for v in cu.ERA5VARS:
            # mcs_local_vars[v] is a list of arrays. List is time dim. Convert to
            # array with shape (ntime, nradius, nlat, nlon).
            data = np.array(mcs_local_vars[v])
            assert data.shape == (len(mcs_times), len(cu.RADII), len(e5ds.latitude), len(e5ds.longitude))
            # Collapse on the time dimension, then give arrage a time dimension with
            # the mean MCS time for easier combining of files.
            dsout[f'mcs_local_{v}'].values = np.nanmean(data, axis=0)[None, :, :, :]

        # TODO: set encoding of dist_mask_sum to int.
        dsout.to_netcdf(self.outputs['mcs_local_env'])


class LifecycleMcsLocalEnvHist(TaskRule):
    @staticmethod
    def rule_inputs(year, month):
        # Both have one extra value at start/end because I need to interp to half hourly.
        # so that I can calc precursor env.
        start = pd.Timestamp(year, month, 1) - pd.Timedelta(hours=25)
        # to account for latest possible time in tracks dataset (400 hours)
        end = start + pd.DateOffset(months=1) + pd.Timedelta(hours=401)
        e5times = pd.date_range(start, end, freq='H')

        inputs = {
            f'era5_{t}_{var}': fmtp(cu.FMT_PATH_ERA5_SFC, year=t.year, month=t.month, day=t.day, hour=t.hour, var=var)
            for t in e5times
            for var in cu.ERA5VARS
        }

        inputs['tracks'] = cu.fmt_mcs_stats_path(year)
        inputs['dists'] = cu.PATH_LAT_LON_DISTS

        return inputs

    rule_outputs = {'lifecycle_mcs_local_env': cu.FMT_PATH_LIFECYCLE_MCS_LOCAL_ENV}

    var_matrix = {
        'year': cu.YEARS,
        'month': cu.MONTHS,
    }

    def rule_run(self):
        # Start from first precursor time.
        start = pd.Timestamp(self.year, self.month, 1) - pd.Timedelta(hours=25)
        # to account for latest possible time in tracks dataset (400 hours)
        end = start + pd.DateOffset(months=1) + pd.Timedelta(hours=401)
        e5times = pd.date_range(start, end, freq='H')
        mcs_times = e5times[:-1] + pd.Timedelta(minutes=30)

        e5paths = [self.inputs[f'era5_{t}_{var}'] for t in e5times for var in cu.ERA5VARS]

        # This is not the most memory-efficient way of doing this.
        # BUT it allows me to load all the data once, and then access
        # it in any order efficiently (because it's all loaded into RAM).
        # Note also that not all values of this are necessarily used.
        # E.g. the last time will only be used if there is a duration=400h
        # MCS (unlikely).
        # Normal trick of interpolating to mcs_times.
        e5ds = xr.open_mfdataset(e5paths).sel(latitude=slice(60, -60)).interp(time=mcs_times).load()

        tracks = McsTracks.open(self.inputs['tracks'], None)
        time = pd.DatetimeIndex(tracks.dstracks.start_basetime)
        dstracks = tracks.dstracks.isel(tracks=(time.month == self.month))

        dists = xr.load_dataarray(self.inputs['dists'])

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
        for var in cu.ERA5VARS:
            bins, hist_mids = cu.get_bins(var)
            coords.update({f'{var}_hist_mids': hist_mids, f'{var}_bins': bins})
            # Starts off as np.nan.
            blank_hist_data = np.full((len(dstracks.tracks), len(cu.RADII), len(hist_mids), 424), np.nan)
            blank_mean_data = np.full((len(dstracks.tracks), len(cu.RADII), 424), np.nan)
            blank_percentile_data = np.full((len(dstracks.tracks), len(cu.RADII), len(percentiles), 424), np.nan)

            data_vars[f'hist_{var}'] = (('tracks', 'radius', f'{var}_hist_mids', 'times'), blank_hist_data)
            data_vars[f'mean_{var}'] = (('tracks', 'radius', 'times'), blank_mean_data)
            data_vars[f'percentile_{var}'] = (('tracks', 'radius', 'percentile', 'times'), blank_percentile_data)

        dsout = xr.Dataset(
            data_vars=data_vars,
            coords=coords,
        )
        dsout.radius.attrs['units'] = 'km'

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
                for j, r in enumerate(cu.RADII):
                    dist_mask = dist < r
                    for var in cu.ERA5VARS:
                        bins = dsout[f'{var}_bins'].values
                        # Go ahead and compute histogram.
                        # Note, to index the times dim, I need to add 24 to i (starts at -24).
                        data = e5ds.sel(time=time)[var].values[dist_mask]
                        dsout[f'hist_{var}'].values[track_idx, j, :, i + 24] = np.histogram(data, bins=bins)[0]
                        dsout[f'mean_{var}'].values[track_idx, j, i + 24] = data.mean()
                        dsout[f'percentile_{var}'].values[track_idx, j, :, i + 24] = np.percentile(data, percentiles)

        cu.to_netcdf_tmp_then_copy(dsout, self.outputs['lifecycle_mcs_local_env'])


class CheckMcsLocalEnv(TaskRule):
    rule_inputs = {
        'mcs_local_env': fmtp(McsLocalEnv.rule_outputs['mcs_local_env'], year=2020, month=1, day=1, mode='init'),
        'tracks': cu.fmt_mcs_stats_path(2020),
    }
    rule_outputs = {'fig': cu.FMT_PATH_CHECK_MCS_LOCAL_ENV}

    var_matrix = {'radius': cu.RADII}

    def rule_run(self):
        tracks = McsTracks.open(self.inputs['tracks'], None)
        mcs_local_env = xr.open_dataset(self.inputs['mcs_local_env']).sel(radius=self.radius)

        mcs_start = dt.datetime(2020, 1, 1, 0, 30)
        # 24 of these on the half hour.
        mcs_times = pd.date_range(mcs_start, mcs_start + dt.timedelta(hours=23), freq='H')

        fig, ax = plt.subplots(
            1,
            1,
            subplot_kw=dict(projection=ccrs.PlateCarree()),
        )
        fig.set_size_inches((1920 / 100, 1080 / 100))
        ax.set_title(f'{self.radius}')
        ax.contourf(mcs_local_env.longitude, mcs_local_env.latitude, mcs_local_env.mcs_local_cape[0])

        for pdtime in mcs_times:
            nptime = pdtime.to_numpy()

            time_mask = tracks.dstracks.base_time.values[:, 0] == nptime
            mcs_lats = tracks.dstracks.meanlat.values[:, 0][time_mask]
            mcs_lons = tracks.dstracks.meanlon.values[:, 0][time_mask]

            ax.scatter(mcs_lons, mcs_lats, marker='x')

        ax.coastlines()
        ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

        plt.savefig(self.outputs['fig'])


class CombineMonthlyMcsLocalEnv(TaskRule):
    @staticmethod
    def rule_inputs(year, month, mode):
        start = pd.Timestamp(year, month, 1)
        end = start + pd.DateOffset(months=1) - pd.Timedelta(days=1)
        times = pd.date_range(start, end, freq='D')

        inputs = {
            f'era5_{t}': fmtp(McsLocalEnv.rule_outputs['mcs_local_env'], year=year, month=month, day=t.day, mode=mode)
            for t in times
        }
        return inputs

    rule_outputs = {'mcs_local_env': cu.FMT_PATH_COMBINE_MCS_LOCAL_ENV}

    var_matrix = {
        ('year', 'month'): cu.DATE_MONTH_KEYS,
        'mode': ['init', 'lifecycle'],
    }

    def rule_run(self):
        paths = self.inputs.values()
        ds = xr.open_mfdataset(paths)

        dsout = ds.mean(dim='time').load()
        dsout = dsout.expand_dims({'time': 1})
        dsout = dsout.assign_coords({'time': [pd.Timestamp(ds.time.mean().item())]})

        dsout.to_netcdf(self.outputs['mcs_local_env'])
