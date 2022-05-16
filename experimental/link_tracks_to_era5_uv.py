import datetime as dt
from itertools import chain
from pathlib import Path
from timeit import default_timer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import xarray as xr

from mcs_prime import PATHS, McsTracks, PixelData
from mcs_prime.util import round_times_to_nearest_second


if __name__ == '__main__':
    e5datadir = Path('/badc/ecmwf-era5/data/oper/an_ml/2019/06/01')
    stats_year_path = PATHS['statsdir'] / 'mcs_tracks_final_extc_20190101.0000_20200101.0000.nc'
    dstracks = xr.open_dataset(stats_year_path)
    round_times_to_nearest_second(dstracks)
    try:
        ds
    except NameError:
        uv = {}
        start = default_timer()
        dstracks.base_time.load()
        dstracks.meanlon.load()
        dstracks.meanlat.load()
        dstracks.movement_distance_x.load()
        dstracks.movement_distance_y.load()
        # Choose some levels.
        levels = [-71, -61, -51, -41, -31, -21, -11, -1]

        for h in range(24):
            print(h)
            e5u = xr.open_dataarray(e5datadir / f'ecmwf-era5_oper_an_ml_20190601{h:02d}00.u.nc')
            e5v = xr.open_dataarray(e5datadir / f'ecmwf-era5_oper_an_ml_20190601{h:02d}00.v.nc')

            # TODO: N.B. 30min offset between track, ERA5 - fix by interp.
            time = dt.datetime(2019, 6, 1, h, 30)
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
                time=times,
                level=e5u.level[levels],
            ),
        )
        # print(uv)
        end = default_timer()
        print(end - start)

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

