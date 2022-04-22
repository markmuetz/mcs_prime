from pathlib import Path

import cartopy
import cartopy.geodesic
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import shapely
import shapely.ops
import xarray as xr


def round_times_to_nearest_second(dstracks):
    """Round times in dstracks.base_time to the nearest second.

    Sometimes the dstracks dataset has minor inaccuracies in the time, e.g.
    '2000-06-01T00:30:00.000013440' (13440 ns). Remove these.

    :param dstracks: xarray.Dataset to convert.
    :return: None
    """

    # N.B. I tried to do this using pure np funcions, but could not work
    # out how to convert np.int64 into a np.datetime64. Seems like it should be
    # easy.

    def remove_time_incaccuracy(t):
        return np.datetime64(int(round(t / 1e9) * 1e9), 'ns')

    vec_remove_time_incaccuracy = np.vectorize(remove_time_incaccuracy)
    tmask = ~np.isnan(dstracks.base_time.values)
    dstracks.base_time.values[tmask] = vec_remove_time_incaccuracy(
        dstracks.base_time.values[tmask].astype(int)
    )
    tmask = ~np.isnan(dstracks.start_basetime.values)
    dstracks.start_basetime.values[tmask] = vec_remove_time_incaccuracy(
        dstracks.start_basetime.values[tmask].astype(int)
    )
    tmask = ~np.isnan(dstracks.end_basetime.values)
    dstracks.end_basetime.values[tmask] = vec_remove_time_incaccuracy(
        dstracks.end_basetime.values[tmask].astype(int)
    )


class McsTracks:
    def __init__(self, dstracks, pixel_data_dir):
        self.dstracks = dstracks
        self.pixel_data_dir = pixel_data_dir
        year = 2000
        pixel_data_year_dir = pixel_data_dir / f'{year}0601.0000_{year + 1}0101.0000'
        track_pixel_paths = sorted(pixel_data_year_dir.glob('*.nc'))
        self.date_path_map = {
            dt.datetime.strptime(p.stem, 'mcstrack_%Y%m%d_%H%M'): p
            for p in track_pixel_paths
        }

    def get_track(self, track_id):
        return McsTrack(
            track_id, self.dstracks.sel(tracks=track_id), self.date_path_map
        )

    def tracks_at_time(self, datetime):
        datetime = pd.Timestamp(datetime).to_numpy()
        dstracks_at_time = self.dstracks.isel(
            tracks=(self.dstracks.base_time.values == datetime).any(axis=1)
        )
        return McsTracks(dstracks_at_time, pixel_data_dir)

    def plot(self, ax=None, display_area=True, display_pf_area=True, times='all'):
        if not ax:
            fig, ax = plt.subplots(
                subplot_kw=dict(projection=cartopy.crs.PlateCarree())
            )
            fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
            ax.coastlines()
            ax.set_title(f'{self}')
            if times != 'all' and len(times) == 1:
                dspixel = xr.open_dataset(self.date_path_map[times[0]])
                ax.contourf(dspixel.lon, dspixel.lat, dspixel.cloudnumber[0])

        for i, track_id in enumerate(self.dstracks.tracks):
            track = self.get_track(track_id.values.item())
            print(f'{track}: {i + 1}/{len(self.dstracks.tracks)}')
            track.plot(ax, display_area, display_pf_area, times)

    @property
    def start(self):
        return pd.Timestamp(self.dstracks.start_basetime.min().values)

    @property
    def end(self):
        return pd.Timestamp(self.dstracks.end_basetime.max().values)

    def __repr__(self):
        return (
            f'McsTracks[{self.start}, {self.end}, ntracks={len(self.dstracks.tracks)}]'
        )


class McsTrack:
    def __init__(self, track_id, dstrack, date_path_map):
        self.track_id = track_id
        self.dstrack = dstrack
        self.date_path_map = date_path_map

    def load(self):
        self.dstrack.load()

    def __getattr__(self, attr):
        if not attr in self.dstrack.variables:
            raise AttributeError(f"Not found: '{attr}'")
        try:
            val = getattr(self.dstrack, attr)
            return val.values[: self.duration]
        except:
            raise

    def plot(self, ax=None, display_area=True, display_pf_area=True, times='all'):
        self.load()
        if not ax:
            fig, ax = plt.subplots(
                subplot_kw=dict(projection=cartopy.crs.PlateCarree())
            )
            fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
            ax.coastlines()
            ax.set_title(f'{self}')
            if times != 'all' and len(times) == 1:
                dspixel = xr.open_dataset(self.date_path_map[times[0]])
                time_index = pd.Timestamp(times[0]).to_numpy()
                cloudnumber = self.cloudnumber[self.base_time == time_index]
                ax.contour(
                    dspixel.lon, dspixel.lat, dspixel.cloudnumber[0] == cloudnumber
                )

        ax.plot(self.meanlon, self.meanlat, 'g-')

        for i in [0, self.duration - 1]:
            ts = pd.Timestamp(self.base_time[i])
            marker = f'id:{self.track_id} - {ts.day}.{ts.hour}:{ts.minute}'
            ax.annotate(
                marker,
                (self.meanlon[i], self.meanlat[i]),
                fontsize='x-small',
            )

        if times == 'all':
            time_indices = range(self.duration)
        else:
            time_indices = []
            for time in [pd.Timestamp(t).to_numpy() for t in times]:
                time_indices.append(np.where(self.base_time == time)[0].item())

        n_points = 20
        if display_area:
            geoms = []
            for i in time_indices:
                lon = self.meanlon[i]
                lat = self.meanlat[i]
                radius = np.sqrt(self.ccs_area[i].item() / np.pi) * 1e3
                if lon < -170:
                    continue
                if np.isnan(radius):
                    continue
                circle_points = cartopy.geodesic.Geodesic().circle(
                    lon=lon, lat=lat, radius=radius, n_samples=n_points, endpoint=False
                )
                geom = shapely.geometry.Polygon(circle_points)
                geoms.append(geom)
            full_geom = shapely.ops.unary_union(geoms)
            ax.add_geometries(
                (full_geom,),
                crs=cartopy.crs.PlateCarree(),
                facecolor='none',
                edgecolor='grey',
                linewidth=2,
            )
        if display_pf_area:
            geoms = []
            for i in time_indices:
                for j in range(3):
                    lon = self.pf_lon[i, j]
                    lat = self.pf_lat[i, j]
                    radius = np.sqrt(self.pf_area[i, j].item() / np.pi) * 1e3
                    if np.isnan(radius):
                        continue
                    circle_points = cartopy.geodesic.Geodesic().circle(
                        lon=lon,
                        lat=lat,
                        radius=radius,
                        n_samples=n_points,
                        endpoint=False,
                    )
                    geom = shapely.geometry.Polygon(circle_points)
                    geoms.append(geom)
            full_geom = shapely.ops.unary_union(geoms)
            ax.add_geometries(
                (full_geom,),
                crs=cartopy.crs.PlateCarree(),
                facecolor='none',
                edgecolor='blue',
                linewidth=1,
            )

    @property
    def duration(self):
        return int(self.dstrack.track_duration.values.item())

    def __repr__(self):
        start = pd.Timestamp(self.dstrack.start_basetime.values)
        end = pd.Timestamp(self.dstrack.end_basetime.values)
        return f'McsTrack[{start}, {end}, id={self.track_id}, duration={self.duration}]'


if __name__ == '__main__':
    stats_dir = Path('/home/markmuetz/Datasets/MCS_PRIME/MCS_database/MCS_Global/stats')
    stats_year_path = stats_dir / 'mcs_tracks_final_extc_20000601.0000_20010101.0000.nc'
    try:
        dstracks
    except NameError:
        dstracks = xr.open_dataset(stats_year_path)
        round_times_to_nearest_second(dstracks)

    pixel_data_dir = Path(
        '/home/markmuetz/Datasets/MCS_PRIME/MCS_database/MCS_Global/mcstracking'
    )

    tracks = McsTracks(dstracks, pixel_data_dir)
    time = dt.datetime(2000, 6, 20, 0, 30)
    tracks.tracks_at_time(time).plot(times=[time])
    tracks.get_track(1946).plot(times=[dt.datetime(2000, 6, 20, 00, 30)])
    plt.show()
