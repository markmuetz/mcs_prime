import datetime as dt
from itertools import chain
from pathlib import Path
from timeit import default_timer

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
from scipy import stats
import xarray as xr

from remake import Remake, TaskRule
from remake.util import format_path as fmtp

from mcs_prime import PATHS, McsTracks, PixelData
from mcs_prime.util import round_times_to_nearest_second

slurm_config = {'account': 'short4hr', 'queue': 'short-serial-4hr', 'mem': 16000}
# slurm_config = {'queue': 'short-serial', 'mem': 16000}
track_era5_prop = Remake(config=dict(slurm=slurm_config))

date = dt.date(2019, 1, 1)
days = []
# while date.day <= 10:
while date.year == 2019:
    days.append(date)
    date += dt.timedelta(days=1)


class TrackERA5PropagationCorr(TaskRule):
    enabled = False
    rule_inputs = {}
    rule_outputs = {'daily_track_era5_data': (PATHS['outdir'] / 'track_era5_prop' / '{year}' / '{month:02d}' / '{day:02d}' /
                                              'daily_track_era5_data_{year}{month:02d}{day:02}.nc')}

    var_matrix = {('year', 'month', 'day'): [(date.year, date.month, date.day) for date in days]}

    def rule_run(self):
        e5datadir = PATHS['era5dir'] / f'data/oper/an_ml/{self.year}/{self.month:02d}/{self.day:02d}'
        stats_year_path = PATHS['statsdir'] / f'mcs_tracks_final_extc_{self.year}0101.0000_{self.year + 1}0101.0000.nc'
        dstracks = xr.open_dataset(stats_year_path)
        round_times_to_nearest_second(dstracks)

        start = default_timer()
        dstracks.base_time.load()
        dstracks.meanlon.load()
        dstracks.meanlat.load()
        dstracks.movement_distance_x.load()
        dstracks.movement_distance_y.load()

        # N.B. not all hours have to have data in them. Skip one that don't.
        datasets = []
        for h in range(24):
            print(h)
            # track data every hour on the half hour.
            track_time = dt.datetime(self.year, self.month, self.day, h, 30)
            # ERA5 data every hour on the hour.
            e5time = dt.datetime(self.year, self.month, self.day, h, 0)
            # Cannot interp track data - get ERA5 before and after and interp
            # using e.g. ...mean(dim=time).
            paths = [e5datadir / (f'ecmwf-era5_oper_an_ml_{t.year}{t.month:02d}{t.day:02d}'
                                  f'{t.hour:02d}00.{var}.nc')
                     for var in ['u', 'v']
                     for t in [e5time, e5time + dt.timedelta(hours=1)]]
            # Only want selected levels,
            # and lat limited to that of tracks, and midpoint time value.
            e5uv = (xr.open_mfdataset(paths).sel(latitude=slice(60, -60))
                    .isel(level=levels).mean(dim='time').load())
            e5u = e5uv.u
            e5v = e5uv.v

            track_point_mask = dstracks.base_time == pd.Timestamp(track_time)
            track_point_lon = dstracks.meanlon.values[track_point_mask]
            track_point_lat = dstracks.meanlat.values[track_point_mask]

            # km/h -> m/s.
            track_point_vel_x = dstracks.movement_distance_x.values[track_point_mask] / 3.6
            track_point_vel_y = dstracks.movement_distance_y.values[track_point_mask] / 3.6

            # This is the same way you select values along a transect.
            # But here I just want at unconnected points.
            lon = xr.DataArray(track_point_lon, dims='track_point')
            lat = xr.DataArray(track_point_lat, dims='track_point')

            if len(lon):
                # N.B. no interp.
                track_point_era5_u = e5u.sel(longitude=lon, latitude=lat, method='nearest').values
                track_point_era5_v = e5v.sel(longitude=lon, latitude=lat, method='nearest').values

                N = len(track_point_lon)
                times = [track_time] * N
                ds = xr.Dataset(data_vars=dict(
                        point_time=('index', times),
                        track_point_era5_u=(['level', 'index'], track_point_era5_u),
                        track_point_era5_v=(['level', 'index'], track_point_era5_v),
                        track_point_vel_x=('index', track_point_vel_x),
                        track_point_vel_y=('index', track_point_vel_y),
                    ),
                    coords=dict(
                        index=range(len(times)),
                        level=e5u.level,
                    ),
                )
                datasets.append(ds)

            e5u.close()
            e5v.close()

        ds = xr.concat(datasets, dim='index')
        ds['index'] = np.arange(len(ds.index))
        end = default_timer()
        print(end - start)
        ds.to_netcdf(self.outputs['daily_track_era5_data'])


class TrackERA5LinkData(TaskRule):
    rule_inputs = {}
    rule_outputs = {'track_era5_linked_data': (PATHS['outdir'] / 'track_era5_prop' / '{year}' / '{month:02d}' / '{day:02d}' /
                                               'track_era5_linked_data_{year}{month:02d}{day:02}.nc')}

    var_matrix = {('year', 'month', 'day'): [(date.year, date.month, date.day) for date in days]}

    def rule_run(self):
        e5datadir = PATHS['era5dir'] / f'data/oper/an_ml/'
        stats_year_path = PATHS['statsdir'] / f'mcs_tracks_final_extc_{self.year}0101.0000_{self.year + 1}0101.0000.nc'
        dstracks = xr.open_dataset(stats_year_path)
        round_times_to_nearest_second(dstracks)

        start = default_timer()
        dstracks.base_time.load()
        dstracks.meanlon.load()
        dstracks.meanlat.load()
        dstracks.movement_distance_x.load()
        dstracks.movement_distance_y.load()

        # Choose some levels.
        # levels = [-61, -41, -21, -1]
        levels = [77, 97, 117, 137]

        datasets = []
        for h in range(24):
            print(h)
            # track data every hour on the half hour.
            track_time = dt.datetime(self.year, self.month, self.day, h, 30)
            # ERA5 data every hour on the hour.
            e5time = dt.datetime(self.year, self.month, self.day, h, 0)
            # Cannot interp track data - get ERA5 before and after and interp using e.g. ...mean(dim=time).

            paths = [e5datadir / (f'{t.year}/{t.month:02d}/{t.day:02d}/'
                                  f'ecmwf-era5_oper_an_ml_{t.year}{t.month:02d}{t.day:02d}'
                                  f'{t.hour:02d}00.{var}.nc')
                     for var in ['u', 'v']
                     for t in [e5time, e5time + dt.timedelta(hours=1)]]
            # Only want levels 38-137 (lowest 100 levels), and lat limited to that of tracks, and midpoint time value.
            e5uv = xr.open_mfdataset(paths).sel(latitude=slice(60, -60)).sel(level=slice(38, None)).mean(dim='time').load()

            e5u = e5uv.u
            e5v = e5uv.v

            track_point_mask = dstracks.base_time == pd.Timestamp(track_time)
            track_point_lon = dstracks.meanlon.values[track_point_mask]
            track_point_lat = dstracks.meanlat.values[track_point_mask]

            track_point_vel_x = dstracks.movement_distance_x.values[track_point_mask] / 3.6 # km/h -> m/s.
            track_point_vel_y = dstracks.movement_distance_y.values[track_point_mask] / 3.6

            # Filter out NaNs.
            nanmask = ~np.isnan(track_point_vel_x)
            track_point_lon = track_point_lon[nanmask]
            track_point_lat = track_point_lat[nanmask]
            track_point_vel_x = track_point_vel_x[nanmask]
            track_point_vel_y = track_point_vel_y[nanmask]

            lon = xr.DataArray(track_point_lon, dims='track_point')
            lat = xr.DataArray(track_point_lat, dims='track_point')

            # N.B. no interp., mean over time does interpolation around half hour.
            track_point_era5_u = e5u.sel(longitude=lon, latitude=lat, method='nearest').values
            track_point_era5_v = e5v.sel(longitude=lon, latitude=lat, method='nearest').values

            e5u.close()
            e5v.close()
            e5uv.close()

            if len(lon):
                # N.B. no interp.
                track_point_era5_u = e5u.sel(longitude=lon, latitude=lat, method='nearest')
                track_point_era5_v = e5v.sel(longitude=lon, latitude=lat, method='nearest')

                # Can now calculate squared diff with judicious use of array broadcasting.
                sqdiff = ((track_point_era5_u.values - track_point_vel_x[None, :])**2 +
                          (track_point_era5_v.values - track_point_vel_y[None, :])**2)

                # What does idx hold? It is the level index of the minimum squared difference
                # between the track point velocity and ERA5 winds. The index starts at level 38 (i.e. 0 index == level 38).
                idx = np.argmin(sqdiff, axis=0)

                N = len(track_point_lon)
                times = [track_time] * N
                ds = xr.Dataset(data_vars=dict(
                        point_time=('index', times),
                        meanlon=('index', lon.values),
                        meanlat=('index', lat.values),
                        track_point_era5_u=(['level', 'index'], track_point_era5_u.sel(level=levels).values),
                        track_point_era5_v=(['level', 'index'], track_point_era5_v.sel(level=levels).values),
                        track_point_vel_x=('index', track_point_vel_x),
                        track_point_vel_y=('index', track_point_vel_y),
                        min_diff_level=('index', e5u.level.values[idx]),
                        min_sq_diff=('index', np.min(sqdiff, axis=0))
                    ),
                    coords=dict(
                        index=np.arange(len(times)),
                        level=e5u.level.sel(level=levels).values,
                    ),
                )
                datasets.append(ds)

        ds = xr.concat(datasets, dim='index')
        ds['index'] = np.arange(len(ds.index))
        ds.to_netcdf(self.outputs['track_era5_linked_data'])


class PlotDay(TaskRule):
    rule_inputs = TrackERA5LinkData.rule_outputs
    rule_outputs = {'fig': (PATHS['figdir'] / 'track_era5_prop'
                            / '{year}' / '{month:02d}' / '{day:02d}' /
                            'daily_track_era5_data_{year}{month:02d}{day:02}.png')}
    var_matrix = {('year', 'month', 'day'): [(2019, 6, 1)]}

    def rule_run(self):
        ds = xr.open_dataset(self.inputs['track_era5_linked_data'])
        levels = ds.level

        fig, axes = plt.subplots(2, len(levels))
        fig.suptitle('ERA5 vs MCS for 2019/6/1')
        for i in range(len(levels)):
            ax0, ax1 = axes[:, i]
            ax0.scatter(ds.track_point_era5_u[i].values, ds.track_point_vel_x.values)
            ax1.scatter(ds.track_point_era5_v[i].values, ds.track_point_vel_y.values)
            ax0.axis('equal')
            ax1.axis('equal')
            ax0.set_xlim((-50, 50))
            ax0.set_ylim((-50, 50))
            ax0.plot([-50, 50], [-50, 50])
            ax1.plot([-50, 50], [-50, 50])
            nanmask0 = ~np.isnan(ds.track_point_vel_x.values)
            nanmask1 = ~np.isnan(ds.track_point_vel_y.values)
            res0 = stats.linregress(ds.track_point_era5_u[i].values[nanmask0], ds.track_point_vel_x.values[nanmask0])
            res1 = stats.linregress(ds.track_point_era5_v[i].values[nanmask1], ds.track_point_vel_y.values[nanmask1])
            x0min, x0max = ds.track_point_era5_u[i].values.min(),ds.track_point_era5_u[i].values.max()
            x1min, x1max = ds.track_point_era5_v[i].values.min(),ds.track_point_era5_v[i].values.max()
            x0 = np.array([-50, 50])
            x1 = np.array([-50, 50])
            print(res0)
            print(res1)
            label0 = f'm={res0.slope:.2f}\nr2={res0.rvalue**2:.2f}'
            label1 = f'm={res1.slope:.2f}\nr2={res1.rvalue**2:.2f}'
            ax0.plot(x0, res0.intercept + res0.slope * x0, 'r', label=label0)
            ax1.plot(x1, res1.intercept + res1.slope * x1, 'r', label=label1)
            ax0.legend()
            ax1.legend()

            ax0.set_title(f'model level: {ds.level[i].values.item()}')
            ax0.set_xlabel('ERA5 u')
            ax1.set_xlabel('ERA5 v')
            if i == 0:
                ax0.set_ylabel('Track u')
                ax1.set_ylabel('Track v')

        plt.show()
        plt.savefig(self.outputs['fig'])


class PlotHist(TaskRule):
    @staticmethod
    def rule_inputs(norm):
        inputs = {
            (date.year, date.month, date.day): fmtp(
                TrackERA5LinkData.rule_outputs['track_era5_linked_data'],
                year=date.year,
                month=date.month,
                day=date.day)
            for date in days
        }
        return inputs
    rule_outputs = {'hist_track_era5_data_fig': (PATHS['figdir'] / 'track_era5_prop' /
                                                 'hist_track_era5.{norm}.png')}
    var_matrix = {'norm': ['none', 'lognorm']}

    def rule_run(self):
        ds = xr.open_mfdataset(self.inputs.values(), concat_dim='index', combine='nested')
        # Re-index.
        ds['index'] = np.arange(len(ds.index))

        levels = ds.level
        nanmask0 = ~np.isnan(ds.track_point_vel_x.values)
        nanmask1 = ~np.isnan(ds.track_point_vel_y.values)

        # Taken from analysing narrow bin histograms (corresponds to dx/dy of ~10km).
        dbin = 2.77
        bins = np.arange(-10 * dbin - dbin / 2, 11 * dbin + dbin / 2, dbin)
        extent = bins[[0, -1, 0, -1]]
        fig, axes = plt.subplots(2, len(levels))
        fig.set_size_inches(20, 6)
        fig.suptitle('ERA5 vs MCS')

        for i in range(len(levels)):
            hist0, _, _ = np.histogram2d(
                ds.track_point_era5_u[i].values[nanmask0],
                ds.track_point_vel_x.values[nanmask0],
                bins=bins,
            )
            hist1, _, _ = np.histogram2d(
                ds.track_point_era5_v[i].values[nanmask1],
                ds.track_point_vel_y.values[nanmask1],
                bins=bins,
            )
            ax0, ax1 = axes[:, i]
            imshow_kwargs = {} if self.norm == 'none' else {'norm': LogNorm()}
            ax0.imshow(hist0.T, origin='lower', extent=extent, **imshow_kwargs)
            im = ax1.imshow(hist1.T, origin='lower', extent=extent, **imshow_kwargs)
            # ax0.scatter(ds.track_point_era5_u[i].values, ds.track_point_vel_x.values, marker='x')
            # ax1.scatter(ds.track_point_era5_v[i].values, ds.track_point_vel_y.values, marker='x')
            res0 = stats.linregress(
                ds.track_point_era5_u[i].values[nanmask0],
                ds.track_point_vel_x.values[nanmask0]
            )
            res1 = stats.linregress(
                ds.track_point_era5_v[i].values[nanmask1],
                ds.track_point_vel_y.values[nanmask1]
            )
            x0min, x0max = ds.track_point_era5_u[i].values.min(), ds.track_point_era5_u[i].values.max()
            x1min, x1max = ds.track_point_era5_v[i].values.min(), ds.track_point_era5_v[i].values.max()
            x0 = np.array(bins[[0, -1]])
            x1 = np.array(bins[[0, -1]])

            label0 = f'm={res0.slope:.2f}\nr2={res0.rvalue**2:.2f}'
            label1 = f'm={res1.slope:.2f}\nr2={res1.rvalue**2:.2f}'
            ax0.plot(x0, res0.intercept + res0.slope * x0, 'r', label=label0)
            ax1.plot(x1, res1.intercept + res1.slope * x1, 'r', label=label1)
            ax0.legend()
            ax1.legend()

            ax0.set_title(f'model level: {ds.level[i].values.item()}')
            ax0.set_xlabel('ERA5 u')
            ax1.set_xlabel('ERA5 v')
            if i == 0:
                ax0.set_ylabel('Track u')
                ax1.set_ylabel('Track v')
        # plt.colorbar(im)

        fig.subplots_adjust(left=0.05, right=0.9, bottom=0.5, top=0.95)
        plt.show()
        plt.savefig(self.outputs['hist_track_era5_data_fig'])

