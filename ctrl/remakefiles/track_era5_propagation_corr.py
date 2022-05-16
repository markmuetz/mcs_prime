import datetime as dt
from itertools import chain
from pathlib import Path
from timeit import default_timer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import xarray as xr

from remake import Remake, TaskRule
from remake.util import format_path as fmtp

from mcs_prime import PATHS, McsTracks, PixelData
from mcs_prime.util import round_times_to_nearest_second

slurm_config = {'account': 'short4hr', 'queue': 'short-serial-4hr', 'mem': 16000}
track_era5_prop = Remake(config=dict(slurm=slurm_config))

date = dt.date(2019, 1, 1)
days = []
# while date.year == 2019:
while date.day <= 10:
    days.append(date)
    date += dt.timedelta(days=1)


class TrackERA5PropagationCorr(TaskRule):
    rule_inputs = {}
    rule_outputs = {'daily_track_era5_data': (PATHS['outdir'] / 'track_era5_prop' / '{year}' / '{month:02d}' / '{day:02d}' /
                                              'daily_track_era5_data_{year}{month:02d}{day:02}.nc')}

    var_matrix = {('year', 'month', 'day'): [(date.year, date.month, date.day) for date in days]}

    def rule_run(self):
        e5datadir = Path(f'/badc/ecmwf-era5/data/oper/an_ml/{self.year}/{self.month:02d}/{self.day:02d}')
        stats_year_path = PATHS['statsdir'] / f'mcs_tracks_final_extc_{self.year}0101.0000_{self.year + 1}0101.0000.nc'
        dstracks = xr.open_dataset(stats_year_path)
        round_times_to_nearest_second(dstracks)

        uv = {}
        start = default_timer()
        dstracks.base_time.load()
        dstracks.meanlon.load()
        dstracks.meanlat.load()
        dstracks.movement_distance_x.load()
        dstracks.movement_distance_y.load()
        # Choose some levels.
        levels = [-61, -41, -21, -1]

        for h in range(24):
            print(h)
            e5u = xr.open_dataarray(e5datadir / f'ecmwf-era5_oper_an_ml_{self.year}{self.month:02d}{self.day:02d}{h:02d}00.u.nc')
            e5v = xr.open_dataarray(e5datadir / f'ecmwf-era5_oper_an_ml_{self.year}{self.month:02d}{self.day:02d}{h:02d}00.v.nc')

            # TODO: N.B. 30min offset between track, ERA5 - fix by interp.
            time = dt.datetime(self.year, self.month, self.day, h, 30)
            track_point_mask = dstracks.base_time == pd.Timestamp(time)
            track_point_lon = dstracks.meanlon.values[track_point_mask]
            track_point_lat = dstracks.meanlat.values[track_point_mask]

            track_point_vel_x = dstracks.movement_distance_x.values[track_point_mask] / 3.6 # km/h -> m/s.
            track_point_vel_y = dstracks.movement_distance_y.values[track_point_mask] / 3.6 # km/h -> m/s.

            # This is the same way you select values along a transect.
            # But here I just want at unconnected points.
            lon = xr.DataArray(track_point_lon, dims='track_point')
            lat = xr.DataArray(track_point_lat, dims='track_point')

            # N.B. no interp.
            track_point_era5_u = (e5u.isel(time=0).isel(level=levels)
                                  .sel(longitude=lon, latitude=lat, method='nearest').values)
            track_point_era5_v = (e5v.isel(time=0).isel(level=levels)
                                  .sel(longitude=lon, latitude=lat, method='nearest').values)

            N = len(track_point_lon)
            uv[h] = dict(
                time=time,
                N=N,
                track_point_era5_u=track_point_era5_u,
                track_point_era5_v=track_point_era5_v,
                track_point_vel_x=track_point_vel_x,
                track_point_vel_y=track_point_vel_y,
            )
            e5u.close()
            e5v.close()

        times = list(chain.from_iterable([[uv[h]['time']] * uv[h]['N'] for h in range(24)]))
        track_point_era5_u = np.concatenate([uv[h]['track_point_era5_u'] for h in range(24)], axis=1)
        track_point_era5_v = np.concatenate([uv[h]['track_point_era5_v'] for h in range(24)], axis=1)
        track_point_vel_x = np.concatenate([uv[h]['track_point_vel_x'] for h in range(24)])
        track_point_vel_y = np.concatenate([uv[h]['track_point_vel_y'] for h in range(24)])
        ds = xr.Dataset(data_vars=dict(
                point_time=('index', times),
                track_point_era5_u=(['level', 'index'], track_point_era5_u),
                track_point_era5_v=(['level', 'index'], track_point_era5_v),
                track_point_vel_x=('index', track_point_vel_x),
                track_point_vel_y=('index', track_point_vel_y),
            ),
            coords=dict(
                index=range(len(times)),
                level=e5u.level[levels],
            ),
        )
        # print(uv)
        end = default_timer()
        print(end - start)
        ds.to_netcdf(self.outputs['daily_track_era5_data'])


class PlotDay(TaskRule):
    rule_inputs = TrackERA5PropagationCorr.rule_outputs
    rule_outputs = {'daily_track_era5_data_fig': (PATHS['figdir'] / 'track_era5_prop' / '{year}' / '{month:02d}' / '{day:02d}' /
                                                  'daily_track_era5_data_{year}{month:02d}{day:02}.png')}
    var_matrix = {('year', 'month', 'day'): [(2019, 6, 1)]}

    def rule_run(self):
        ds = xr.open_dataset(self.inputs['daily_track_era5_data'])
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
        plt.savefig(self.outputs['daily_track_era5_data_fig'])


class PlotHist(TaskRule):
    @staticmethod
    def rule_inputs():
        inputs = {
            (date.year, date.month, date.day): fmtp(
                TrackERA5PropagationCorr.rule_outputs['daily_track_era5_data'],
                year=date.year,
                month=date.month,
                day=date.day)
            for date in days
        }
        return inputs
    rule_outputs = {'hist_track_era5_data_fig': (PATHS['figdir'] / 'track_era5_prop' /
                                                 'hist_track_era5.png')}

    def rule_run(self):
        ds = xr.open_mfdataset(self.inputs.values(), concat_dim='index', combine='nested')
        # Re-index.
        ds['index'] = np.arange(len(ds.index))

        levels = ds.level
        nanmask0 = ~np.isnan(ds.track_point_vel_x.values)
        nanmask1 = ~np.isnan(ds.track_point_vel_y.values)

        bins = np.linspace(-25, 25, 20)
        extent = (-25, 25, -25, 25)
        fig, axes = plt.subplots(2, len(levels))
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
            ax0.imshow(hist0.T, origin='lower', extent=extent)
            ax1.imshow(hist1.T, origin='lower', extent=extent)
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
            x0 = np.array([-25, 25])
            x1 = np.array([-25, 25])

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
        plt.savefig(self.outputs['hist_track_era5_data_fig'])

