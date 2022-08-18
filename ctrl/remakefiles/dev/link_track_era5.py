import datetime as dt

import numpy as np
import pandas as pd
import xarray as xr

from remake import Remake, TaskRule

from mcs_prime import PATHS
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


class TrackERA5LinkData(TaskRule):
    rule_inputs = {}
    rule_outputs = {'track_era5_linked_data': (PATHS['outdir'] / 'track_era5_linked' / '{year}' / '{month:02d}' / '{day:02d}' /
                                               'track_era5_linked_data_{year}{month:02d}{day:02}.nc')}

    var_matrix = {('year', 'month', 'day'): [(date.year, date.month, date.day) for date in days]}

    def rule_run(self):
        e5datadir = PATHS['era5dir'] / f'data/oper/an_ml/'
        stats_year_path = PATHS['statsdir'] / f'mcs_tracks_final_extc_{self.year}0101.0000_{self.year + 1}0101.0000.nc'
        dstracks = xr.open_dataset(stats_year_path)
        round_times_to_nearest_second(dstracks)

        dstracks.base_time.load()
        dstracks.meanlon.load()
        dstracks.meanlat.load()
        dstracks.movement_distance_x.load()
        dstracks.movement_distance_y.load()

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
            # Only want levels 77-137 (lowest 60 levels), and lat limited to that of tracks,
            # and midpoint (mean) time value.
            # Why use 77? I have done testing which shows that the distribution of
            # mindiff level is bimodal, with a minimum at level 77. I am only really interested
            # in "steering level" winds below this, so find the mindiff level only in levels 77+.
            e5uv = (xr.open_mfdataset(paths).sel(latitude=slice(60, -60))
                    .sel(level=slice(77, None)).mean(dim='time').load())

            e5u = e5uv.u
            e5v = e5uv.v

            track_point_mask = dstracks.base_time == pd.Timestamp(track_time)
            track_point_lon = dstracks.meanlon.values[track_point_mask]
            track_point_lat = dstracks.meanlat.values[track_point_mask]
            # / 3.6 to convert km/h -> m/s.
            track_point_vel_x = dstracks.movement_distance_x.values[track_point_mask] / 3.6
            track_point_vel_y = dstracks.movement_distance_y.values[track_point_mask] / 3.6

            # Filter out NaNs.
            nanmask = ~np.isnan(track_point_vel_x)
            track_point_lon = track_point_lon[nanmask]
            track_point_lat = track_point_lat[nanmask]
            track_point_vel_x = track_point_vel_x[nanmask]
            track_point_vel_y = track_point_vel_y[nanmask]

            lon = xr.DataArray(track_point_lon, dims='track_point')
            lat = xr.DataArray(track_point_lat, dims='track_point')

            # N.B. no interp.
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
                # between the track point velocity and ERA5 winds. The index starts at level 77 (i.e. 0 index == level 77).
                idx = np.argmin(sqdiff, axis=0)

                N = len(track_point_lon)
                times = [track_time] * N
                ds = xr.Dataset(data_vars=dict(
                        point_time=('index', times),
                        meanlon=('index', lon.values),
                        meanlat=('index', lat.values),
                        track_point_era5_u=(['level', 'index'], track_point_era5_u.values),
                        track_point_era5_v=(['level', 'index'], track_point_era5_v.values),
                        track_point_vel_x=('index', track_point_vel_x),
                        track_point_vel_y=('index', track_point_vel_y),
                        min_diff_level=('index', e5u.level.values[idx]),
                        min_sq_diff=('index', np.min(sqdiff, axis=0))
                    ),
                    coords=dict(
                        index=np.arange(len(times)),
                        level=e5u.level.values,
                    ),
                )
                datasets.append(ds)

        ds = xr.concat(datasets, dim='index')
        ds['index'] = np.arange(len(ds.index))
        ds.to_netcdf(self.outputs['track_era5_linked_data'])

