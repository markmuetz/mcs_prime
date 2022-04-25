import gc
import datetime as dt
import math
from pathlib import Path

import cartopy
import cartopy.geodesic
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import shapely
import shapely.ops
import xarray as xr


def create_fig_ax():
    fig, ax = plt.subplots(
        subplot_kw=dict(projection=cartopy.crs.PlateCarree())
    )
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
    ax.coastlines()
    return fig, ax


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


def ticks_from_min_max_lon_lat(minlon, maxlon, minlat, maxlat, dlon=5, dlat=5):
    tminlon = (minlon + dlon) - (minlon % dlon)
    tmaxlon = maxlon - (maxlon % dlon)
    tminlat = (minlat + dlat) - (minlat % dlat)
    tmaxlat = maxlat - (maxlat % dlat)
    return np.arange(tminlon, tmaxlon + dlon, dlon), np.arange(tminlat, tmaxlat + dlat, dlat)


class PixelData:
    @classmethod
    def set_pixel_data_dir(cls, pixel_data_dir):
        track_pixel_paths = sorted(pixel_data_dir.glob('*/*.nc'))
        cls.date_path_map = {
            dt.datetime.strptime(p.stem, 'mcstrack_%Y%m%d_%H%M'): p
            for p in track_pixel_paths
        }

    @classmethod
    def get_frame(cls, time):
        return PixelFrame(time, xr.open_dataset(cls.date_path_map[time]))

    @classmethod
    def get_frames(cls, starttime, endtime):
        time = starttime
        paths = []
        times = []
        while time <= endtime:
            if time in cls.date_path_map:
                paths.append(cls.date_path_map[time])
            else:
                print(f'Warning: missing path: {path}')
            times.append(time)
            time += dt.timedelta(hours=1)
        return PixelFrames(times, xr.open_mfdataset(paths))

    @classmethod
    def anim_plot_track(cls, track, method='contourf', method_kwargs=None, zoom=False):
        times = pd.DatetimeIndex(track.base_time)
        start = pd.Timestamp(track.dstrack.start_basetime.values).to_pydatetime()
        end = pd.Timestamp(track.dstrack.end_basetime.values).to_pydatetime()

        cloudnumbers = track.cloudnumber
        frames = cls.get_frames(start, end)
        extent = frames.get_min_max_lon_lat(cloudnumbers)
        dlon = extent[1] - extent[0]
        dlat = extent[3] - extent[2]
        aspect = (dlon + 3) / (dlat * 3)
        height = 9.5

        fig = plt.figure('track animation')
        fig.set_size_inches(height * aspect, height)
        fig.subplots_adjust(left=0.1, right=0.95, bottom=0.05, top=0.95, hspace=0.1)

        precip = frames.get_data('precipitation', cloudnumbers)
        tb = frames.dspixel.tb.values
        cn = frames.get_data('cloudnumber', cloudnumbers)

        cn_swath = cn.sum(axis=0).astype(bool)
        precip_swath = precip.sum(axis=0)

        legend_elements = [
            Patch(facecolor='none', edgecolor='black', label='Tb = 241K'),
            Patch(facecolor='none', edgecolor='red', label='Cloud mask'),
            Line2D([0], [0], color='g', label='track'),
            Patch(facecolor='none', edgecolor='grey', label='Track cloud (circle)'),
            Patch(facecolor='none', edgecolor='blue', label='Track PFx3 (circle)'),
        ]

        print(precip.max())

        def precip_levels(pmax):
            pmax10 = math.ceil(pmax / 10) * 10
            return np.array([0] + list(2**np.arange(7))) / 2**6 * math.ceil(pmax10)

        frame_precip_levels = precip_levels(precip.max())
        frame_norm = mpl.colors.BoundaryNorm(boundaries=frame_precip_levels, ncolors=256)
        swath_precip_levels = precip_levels(precip_swath.max())
        swath_norm = mpl.colors.BoundaryNorm(boundaries=swath_precip_levels, ncolors=256)

        if precip.max() > frame_precip_levels[-1]:
            extend = 'max'
        else:
            extend = 'neither'

        for i, (cloudnumber, time) in enumerate(zip(cloudnumbers, times)):
            print(time)
            fig.clf()
            ax1 = fig.add_subplot(3, 1, 1, projection=cartopy.crs.PlateCarree())
            ax2 = fig.add_subplot(3, 1, 2, projection=cartopy.crs.PlateCarree())
            ax3 = fig.add_subplot(3, 1, 3)

            ax1.set_title(f'Track {track.track_id} @ {time}')
            ax1.coastlines()
            ax2.coastlines()
            track.plot(times=[time], ax=ax1)
            track.plot(times=[time], ax=ax2)

            im1 = ax1.contourf(frames.dspixel.lon, frames.dspixel.lat, precip[i],
                               cmap='Blues', norm=frame_norm, levels=frame_precip_levels, extend=extend)
            ax1.contour(frames.dspixel.lon, frames.dspixel.lat, cn[i], levels=[0.5], colors='red')
            ax1.contour(frames.dspixel.lon, frames.dspixel.lat, tb[i], levels=[241], colors='black')
            plt.colorbar(im1, ax=ax1, label='masked precip. (mm hr$^{-1}$)')

            im2 = ax2.contourf(frames.dspixel.lon, frames.dspixel.lat, precip_swath,
                               cmap='Blues', norm=swath_norm, levels=swath_precip_levels)
            ax2.contour(frames.dspixel.lon, frames.dspixel.lat, cn_swath, levels=[0.5], colors='red')

            cbar2 = plt.colorbar(im2, ax=ax2, label='masked precip. swath (mm)')
            cbar2.set_ticks(swath_precip_levels)
            cbar2.set_ticklabels([f'{v:.2f}' for v in swath_precip_levels])

            ax1.set_extent(extent)
            ax2.set_extent(extent)

            lon_ticks, lat_ticks = ticks_from_min_max_lon_lat(*extent)

            def fmt_lat_ticklabels(ticks):
                labels = []
                for t in ticks:
                    if t < 0:
                        labels.append(f'{-t:.0f} °S')
                    else:
                        labels.append(f'{t:.0f} °N')
                return labels

            def fmt_lon_ticklabels(ticks):
                labels = []
                for t in ticks:
                    if t < 0:
                        labels.append(f'{-t:.0f} °W')
                    else:
                        labels.append(f'{t:.0f} °E')
                return labels

            ax1.set_yticks(lat_ticks)
            ax1.set_yticklabels(fmt_lat_ticklabels(lat_ticks))

            ax2.set_yticks(lat_ticks)
            ax2.set_yticklabels(fmt_lat_ticklabels(lat_ticks))

            ax2.set_xticks(lon_ticks)
            ax2.set_xticklabels(fmt_lon_ticklabels(lon_ticks))

            ax1.legend(handles=legend_elements)
            for name, data in [
                ('area', track.area),
                ('PF area', np.nanmean(track.pf_area, axis=1)),
                ('PF rainrate', np.nanmean(track.pf_rainrate, axis=1)),
            ]:
                ax3.plot(data / data.max(), label=f'{name} (max={data.max():.1f})')
            ax3.set_xlim((-0.5, track.duration - 0.5))
            ax3.set_ylim((0, 1.6))
            ax3.set_yticks([0, 0.5, 1])
            ax3.axvline(x=i)
            ax3.legend()
            ax3.set_xlabel('time since start (hr)')

            figpath = Path(f'figs/anim_plot_track/track_{track.track_id}'
                           f'/track_{track.track_id}_{i:03d}.png')
            figpath.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(figpath)
            # plt.pause(0.1)
            # if input('q to quit: ') == 'q':
            #     raise Exception('quit')

    @classmethod
    def plot_track(cls, field, track, method='contourf', method_kwargs=None, ax=None, zoom=False):
        start = pd.Timestamp(track.dstrack.start_basetime.values).to_pydatetime()
        end = pd.Timestamp(track.dstrack.end_basetime.values).to_pydatetime()
        frames= cls.get_frames(start, end)

        cloudnumbers = track.cloudnumber
        frames.plot(field, method=method, method_kwargs=method_kwargs,
                    cloudnumbers=cloudnumbers, ax=ax, zoom=zoom)


class PixelFrames:
    def __init__(self, time, dspixel):
        self.time = time
        self.dspixel = dspixel

    def get_min_max_lon_lat(self, cloudnumbers):
        assert len(self.dspixel.time) == len(cloudnumbers)
        minlon = math.inf
        maxlon = -math.inf
        minlat = math.inf
        maxlat = -math.inf
        lon = self.dspixel.longitude.load().values
        lat = self.dspixel.latitude.load().values
        data_cloudnumber = self.dspixel.cloudnumber.load()
        for i, cloudnumber in enumerate(cloudnumbers):
            cloudmask = (self.dspixel.cloudnumber[i] == cloudnumber)
            minlon = min(lon[i][cloudmask].min(), minlon)
            maxlon = max(lon[i][cloudmask].max(), maxlon)
            minlat = min(lat[i][cloudmask].min(), minlat)
            maxlat = max(lat[i][cloudmask].max(), maxlat)
        return (minlon, maxlon, minlat, maxlat)

    def get_data(self, field, cloudnumbers):
        data = getattr(self.dspixel, field).values
        if len(cloudnumbers):
            data = data.copy()
            for i, cloudnumber in enumerate(cloudnumbers):
                cloudmask = (self.dspixel.cloudnumber[i] == cloudnumber)
                data[i][~cloudmask] = 0
        return data

    def plot(self, field, method='contourf', method_kwargs=None, cloudnumbers=None, ax=None, zoom=False):
        if method not in ['contour', 'contourf']:
            raise ValueError("method must be one of 'contour', 'contourf'")
        if not ax:
            fig, ax = create_fig_ax()
            ax.set_title(f'{field} @ {self.time}')
        if len(cloudnumbers) and zoom:
            extent = self.get_min_max_lon_lat(cloudnumbers)
            ax.set_extent(extent)
        if not method_kwargs:
            method_kwargs = {}

        data = getattr(self.dspixel, field).values
        if len(cloudnumbers):
            data = data.copy()
            for i, cloudnumber in enumerate(cloudnumbers):
                cloudmask = (self.dspixel.cloudnumber[i] == cloudnumber)
                data[i][~cloudmask] = 0
        mean_data = data.mean(axis=0)
        getattr(ax, method)(self.dspixel.lon, self.dspixel.lat, mean_data, **method_kwargs)


class PixelFrame:
    def __init__(self, time, dspixel):
        self.time = time
        self.dspixel = dspixel

    def plot(self, field, method='contourf', method_kwargs=None, cloudnumber=None, ax=None, zoom=False):
        if method not in ['contour', 'contourf']:
            raise ValueError("method must be one of 'contour', 'contourf'")
        if not ax:
            fig, ax = create_fig_ax()
            ax.set_title(f'{field} @ {self.time}')
        if cloudnumber:
            cloudmask = (self.dspixel.cloudnumber == cloudnumber)[0]
            print(cloudmask.sum())
            if zoom:
                minlon = self.dspixel.longitude.values[cloudmask].min()
                maxlon = self.dspixel.longitude.values[cloudmask].max()
                minlat = self.dspixel.latitude.values[cloudmask].min()
                maxlat = self.dspixel.latitude.values[cloudmask].max()
                ax.set_extent([minlon, maxlon, minlat, maxlat])
        if not method_kwargs:
            method_kwargs = {}

        data = getattr(self.dspixel, field)[0].values
        if cloudnumber:
            data = data.copy()
            data[~cloudmask] = 0
        getattr(ax, method)(self.dspixel.lon, self.dspixel.lat, data, **method_kwargs)


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
                frame = PixelData.get_frame(times[0])
                frame.plot('cloudnumber', ax=ax)

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
                time_index = pd.Timestamp(times[0]).to_numpy()
                cloudnumber = self.cloudnumber[self.base_time == time_index]
                frame = PixelData.get_frame(times[0])
                frame.plot('precipitation', 'contour', cloudnumber=cloudnumber, ax=ax, zoom=True)

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
                ax.plot(lon, lat, 'go')
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
    # plt.ion()
    stats_dir = Path('/home/markmuetz/Datasets/MCS_PRIME/MCS_database/MCS_Global/stats')
    # 2019.
    stats_year_path = stats_dir / 'mcs_tracks_final_extc_20190101.0000_20200101.0000.nc'
    try:
        dstracks
    except NameError:
        dstracks = xr.open_dataset(stats_year_path)
        round_times_to_nearest_second(dstracks)

    pixel_data_dir = Path(
        '/home/markmuetz/Datasets/MCS_PRIME/MCS_database/MCS_Global/mcstracking'
    )

    tracks = McsTracks(dstracks, pixel_data_dir)
    PixelData.set_pixel_data_dir(pixel_data_dir)
    time = dt.datetime(2019, 6, 21, 6, 30)
    selected_tracks = tracks.tracks_at_time(time)
    # track = tracks.get_track(15378)

    track = tracks.get_track(15477)
    PixelData.anim_plot_track(track, zoom=True)

    # gc.collect()

    # track = tracks.get_track(15463)
    # PixelData.anim_plot_track(track, zoom=True)

    # for track_id in selected_tracks.dstracks.tracks.values:
    #     print('='*3, track_id, '='*3)
    #     track = tracks.get_track(track_id)
    #     print(track)
    #     PixelData.anim_plot_track(track, zoom=True)
    #     gc.collect()
    # plt.show()
