import datetime as dt
from itertools import product

import cartopy.crs as ccrs
from geopy import distance
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


from remake import Remake, TaskRule
from remake.util import format_path as fmtp

from mcs_prime import PATHS, McsTracks, PixelData

# slurm_config = {'queue': 'short-serial', 'mem': 64000, 'max_runtime': '10:00:00'}
slurm_config = {'account': 'short4hr', 'queue': 'short-serial-4hr', 'mem': 32000}
mcs_local_envs = Remake(config=dict(slurm=slurm_config, content_checks=False))

years = list(range(2000, 2021))
# years = [2020]
months = range(1, 13)

# For testing - 5 days.
# DATES = pd.date_range(f'{years[0]}-01-01', f'{years[0]}-01-05')
DATES = pd.date_range(f'{years[0]}-01-01', f'{years[-1]}-12-31')
DATE_KEYS = [(y, m, d) for y, m, d in zip(DATES.year, DATES.month, DATES.day)]
DATE_MONTH_KEYS = [(y, m) for y in years for m in months]


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


class GenLatLonDistance(TaskRule):
    rule_inputs = {'cape': (PATHS['era5dir'] / 'data/oper/an_sfc/'
                            '2020/01/01'
                            '/ecmwf-era5_oper_an_sfc_'
                            '202001010000.cape.nc')}
    rule_outputs = {'dists': (PATHS['outdir'] / 'mcs_local_envs' /
                              'lat_lon_distances.nc')}

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
            )
        )
        da.to_netcdf(self.outputs['dists'])


def get_dist(da, lat, lon):
    lat_idx = find_nearest(da.dist_lat.values, lat)
    lon_idx = find_nearest_circular(da.lon.values, lon)
    return lat_idx, lon_idx, np.roll(da.values[lat_idx], lon_idx, axis=1)


class CheckLatLonDistance(TaskRule):
    rule_inputs = {'dists': (PATHS['outdir'] / 'mcs_local_envs' /
                             'lat_lon_distances.nc')}
    rule_outputs = {'fig': (PATHS['figdir'] / 'mcs_local_envs' /
                            'check_figs' /
                            'lat_lon_distances_{lat}_{lon}_{radius}.png')}

    var_matrix = {('lat', 'lon', 'radius'): [
        (0, 0, 200),
        (10, 20, 500),
        (30, 350, 1000),
        (-40, 360, 500),
    ]}

    def rule_run(self):
        dists = xr.open_dataarray(self.inputs['dists'])
        lats = dists.lat.values
        lons = dists.lon.values
        lat = self.lat
        lon = self.lon

        lat_idx, lon_idx, dist = get_dist(dists, lat, lon)
        fig, ax = plt.subplots(
            1, 1,
            subplot_kw=dict(projection=ccrs.PlateCarree()),
        )

        ax.set_title(f'{lat}, {lon}, {self.radius}')
        ax.contourf(lons, lats, dist < self.radius)
        ax.coastlines()
        ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

        plt.savefig(self.outputs['fig'])


ERA5VARS = ['cape', 'tcwv']


class McsLocalEnv(TaskRule):
    @staticmethod
    def rule_inputs(year, month, day):
        start = dt.datetime(year, month, day)
        # Note there are 25 of these so I can get ERA5 data on the hour either side
        # of MCS dataset data (on the half hour).
        e5times = pd.date_range(start, start + dt.timedelta(days=1), freq='H')

        inputs = {f'era5_{t}_{var}': (PATHS['era5dir'] /
                                      f'data/oper/an_sfc/{t.year}/{t.month:02d}/{t.day:02d}' /
                                      (f'ecmwf-era5_oper_an_sfc_{t.year}{t.month:02d}{t.day:02d}'
                                       f'{t.hour:02d}00.{var}.nc'))
                  for t in e5times
                  for var in ERA5VARS}

        if year == 2000:
            start_date = '20000601'
        else:
            start_date = f'{year}0101'
        inputs['tracks'] = (PATHS['statsdir'] /
                                    f'mcs_tracks_final_extc_{start_date}.0000_{year + 1}0101.0000.nc')
        inputs['dists'] = (PATHS['outdir'] / 'mcs_local_envs' / 'lat_lon_distances.nc')

        return inputs

    rule_outputs = {'mcs_local_env': (PATHS['outdir'] / 'mcs_local_envs' /
                                      '{year}' / '{month:02d}' /
                                      'mcs_local_env_{year}_{month:02d}_{day:02d}.nc')}

    var_matrix = {
        ('year', 'month', 'day'): DATE_KEYS,
    }

    def rule_run(self):
        tracks = McsTracks.open(self.inputs['tracks'], None)

        dists = xr.open_dataarray(self.inputs['dists'])

        start = dt.datetime(self.year, self.month, self.day)
        # Note there are 25 of these so I can get ERA5 data on the hour either side
        # of MCS dataset data (on the half hour).
        e5times = pd.date_range(start, start + dt.timedelta(days=1), freq='H')

        mcs_start = dt.datetime(self.year, self.month, self.day, 0, 30)
        # 24 of these on the half hour.
        mcs_times = pd.date_range(mcs_start, mcs_start + dt.timedelta(hours=23), freq='H')

        e5paths = [self.inputs[f'era5_{t}_{v}']
                   for t in e5times
                   for v in ERA5VARS]
        # Interp to MCS times (on the half hour).
        e5ds = (xr.open_mfdataset(e5paths).sel(latitude=slice(60, -60))
                .interp(time=mcs_times).load())

        mcs_local_vars = {}
        for v in ERA5VARS:
            mcs_local_vars[v] = []

        for i, pdtime in enumerate(mcs_times):
            nptime = pdtime.to_numpy()
            time = pdtime.to_pydatetime()
            print(time)

            time_mask = tracks.dstracks.base_time.values[:, 0] == nptime
            mcs_init_lats = tracks.dstracks.meanlat.values[:, 0][time_mask]
            mcs_init_lons = tracks.dstracks.meanlon.values[:, 0][time_mask]

            e5data = e5ds.sel(time=pdtime)

            mcs_local_mask = np.zeros_like(e5data.cape.values, dtype=bool)

            for lat, lon in zip(mcs_init_lats, mcs_init_lons):
                lat_idx, lon_idx, dist = get_dist(dists, lat, lon)
                dist_mask = dist < 500
                mcs_local_mask |= dist_mask

            for v in ERA5VARS:
                mcs_local_var = np.copy(e5data[v].values)
                mcs_local_var[~mcs_local_mask] = np.nan
                mcs_local_vars[v].append(mcs_local_var)

        data_vars = {}
        for v in ERA5VARS:
            # Give both arrayes a time dimention with the mean MCS time for easier combining of files.
            data_vars[f'{v}'] = (('time', 'latitude', 'longitude'),
                                 e5ds[v].mean(dim='time').values[None, :, :])
            data_vars[f'mcs_local_{v}'] = (('time', 'latitude', 'longitude'),
                                           np.nanmean(np.array(mcs_local_vars[v]), axis=0)[None, :, :])
        dsout = xr.Dataset(
            data_vars=data_vars,
            coords=dict(
                time=[mcs_times.mean()],
                latitude=e5data.latitude,
                longitude=e5data.longitude,
            )
        )
        dsout.to_netcdf(self.outputs['mcs_local_env'])


class CombineMonthlyMcsLocalEnv(TaskRule):
    @staticmethod
    def rule_inputs(year, month):
        start = dt.datetime(year, month, 1)
        endmonth = month + 1
        endyear = year
        if endmonth == 13:
            endmonth = 1
            endyear += 1
        end = dt.datetime(endyear, endmonth, 1) - dt.timedelta(days=1)
        times = pd.date_range(start, end, freq='D')

        inputs = {f'era5_{t}': fmtp(McsLocalEnv.rule_outputs['mcs_local_env'],
                                    year=year, month=month, day=t.day)
                  for t in times}
        return inputs

    rule_outputs = {'mcs_local_env': (PATHS['outdir'] / 'mcs_local_envs' /
                                      '{year}' / '{month:02d}' /
                                      'mcs_local_env_{year}_{month:02d}.nc')}

    var_matrix = {
        ('year', 'month'): DATE_MONTH_KEYS,
    }

    def rule_run(self):
        paths = self.inputs.values()
        ds = xr.open_mfdataset(paths)

        dsout = ds.mean(dim='time').load()
        dsout = dsout.expand_dims({'time': 1})
        dsout = dsout.assign_coords({'time': [pd.Timestamp(ds.time.mean().item())]})

        dsout.to_netcdf(self.outputs['mcs_local_env'])

