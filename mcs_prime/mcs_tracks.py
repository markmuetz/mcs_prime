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

from .util import round_times_to_nearest_second


def create_fig_ax():
    fig, ax = plt.subplots(subplot_kw=dict(projection=cartopy.crs.PlateCarree()))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
    ax.coastlines()
    return fig, ax


def ticks_from_min_max_lon_lat(minlon, maxlon, minlat, maxlat, dlon=5, dlat=5):
    tminlon = (minlon + dlon) - (minlon % dlon)
    tmaxlon = maxlon - (maxlon % dlon)
    tminlat = (minlat + dlat) - (minlat % dlat)
    tmaxlat = maxlat - (maxlat % dlat)
    return np.arange(tminlon, tmaxlon + dlon, dlon), np.arange(tminlat, tmaxlat + dlat, dlat)


class PixelData:
    def __init__(self, pixel_data_dir):
        track_pixel_paths = sorted(pixel_data_dir.glob('*/*.nc'))
        self.date_path_map = {dt.datetime.strptime(p.stem, 'mcstrack_%Y%m%d_%H%M'): p for p in track_pixel_paths}

    def get_frame(self, time):
        return PixelFrame(time, xr.open_dataset(self.date_path_map[time]))

    def get_frames(self, starttime, endtime):
        time = starttime
        paths = []
        times = []
        while time <= endtime:
            if time in self.date_path_map:
                paths.append(self.date_path_map[time])
            else:
                print(f'Warning: missing path: {path}')
            times.append(time)
            time += dt.timedelta(hours=1)
        return PixelFrames(times, xr.open_mfdataset(paths))

    @classmethod
    def plot_track(self, field, track, method='contourf', method_kwargs=None, ax=None, zoom=False):
        start = pd.Timestamp(track.dstrack.start_basetime.values).to_pydatetime()
        end = pd.Timestamp(track.dstrack.end_basetime.values).to_pydatetime()
        frames = self.get_frames(start, end)

        cloudnumbers = track.cloudnumber
        frames.plot(
            field,
            method=method,
            method_kwargs=method_kwargs,
            cloudnumbers=cloudnumbers,
            ax=ax,
            zoom=zoom,
        )


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
            cloudmask = self.dspixel.cloudnumber[i] == cloudnumber
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
                cloudmask = self.dspixel.cloudnumber[i] == cloudnumber
                data[i][~cloudmask] = 0
        return data

    def plot(
        self,
        field,
        method='contourf',
        method_kwargs=None,
        cloudnumbers=None,
        ax=None,
        zoom=False,
    ):
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
                cloudmask = self.dspixel.cloudnumber[i] == cloudnumber
                data[i][~cloudmask] = 0
        mean_data = data.mean(axis=0)
        getattr(ax, method)(self.dspixel.lon, self.dspixel.lat, mean_data, **method_kwargs)


class PixelFrame:
    def __init__(self, time, dspixel):
        self.time = time
        self.dspixel = dspixel

    def plot(
        self,
        field,
        method='contourf',
        method_kwargs=None,
        cloudnumber=None,
        ax=None,
        zoom=False,
    ):
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
    @classmethod
    def load(cls, stats_path, pixeldir):
        dstracks = xr.open_dataset(stats_path)
        round_times_to_nearest_second(dstracks)
        pixel_data = PixelData(pixeldir)
        return cls(dstracks, pixel_data)

    def __init__(self, dstracks, pixel_data=None):
        self.dstracks = dstracks
        self.pixel_data = pixel_data

    def get_track(self, track_id):
        return McsTrack(track_id, self.dstracks.sel(tracks=track_id), self.pixel_data)

    def tracks_at_time(self, datetime):
        datetime = pd.Timestamp(datetime).to_numpy()
        dstracks_at_time = self.dstracks.isel(tracks=(self.dstracks.base_time.values == datetime).any(axis=1))
        return McsTracks(dstracks_at_time)

    def plot(self, ax=None, display_area=True, display_pf_area=True, times='all'):
        if not ax:
            fig, ax = plt.subplots(subplot_kw=dict(projection=cartopy.crs.PlateCarree()))
            fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
            ax.coastlines()
            ax.set_title(f'{self}')
            if times != 'all' and len(times) == 1:
                frame = self.pixel_data.get_frame(times[0])
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
        return f'McsTracks[{self.start}, {self.end}, ntracks={len(self.dstracks.tracks)}]'


class McsTrack:
    def __init__(self, track_id, dstrack, pixel_data):
        self.track_id = track_id
        self.dstrack = dstrack
        self.pixel_data = pixel_data

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
            fig, ax = plt.subplots(subplot_kw=dict(projection=cartopy.crs.PlateCarree()))
            fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
            ax.coastlines()
            ax.set_title(f'{self}')
            if times != 'all' and len(times) == 1:
                time_index = pd.Timestamp(times[0]).to_numpy()
                cloudnumber = self.cloudnumber[self.base_time == time_index]
                frame = self.pixel_data.get_frame(times[0])
                frame.plot(
                    'precipitation',
                    'contour',
                    cloudnumber=cloudnumber,
                    ax=ax,
                    zoom=True,
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

    def animate(self, method='contourf', method_kwargs=None, zoom=False, savefigs=False, figdir=None):
        user_input = ''
        times = pd.DatetimeIndex(self.base_time)
        start = pd.Timestamp(self.dstrack.start_basetime.values).to_pydatetime()
        end = pd.Timestamp(self.dstrack.end_basetime.values).to_pydatetime()

        cloudnumbers = self.cloudnumber
        frames = self.pixel_data.get_frames(start, end)
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

        def precip_levels(pmax):
            pmax10 = math.ceil(pmax / 10) * 10
            return np.array([0] + list(2 ** np.arange(7))) / 2**6 * math.ceil(pmax10)

        frame_precip_levels = precip_levels(precip.max())
        frame_norm = mpl.colors.BoundaryNorm(boundaries=frame_precip_levels, ncolors=256)
        swath_precip_levels = precip_levels(precip_swath.max())
        swath_norm = mpl.colors.BoundaryNorm(boundaries=swath_precip_levels, ncolors=256)

        figpaths = []
        for i, (cloudnumber, time) in enumerate(zip(cloudnumbers, times)):
            print(time)
            fig.clf()
            ax1 = fig.add_subplot(3, 1, 1, projection=cartopy.crs.PlateCarree())
            ax2 = fig.add_subplot(3, 1, 2, projection=cartopy.crs.PlateCarree())
            ax3 = fig.add_subplot(3, 1, 3)

            ax1.set_title(f'Track {self.track_id} @ {time}')
            ax2.set_title(f'Track {self.track_id} swath')
            ax1.coastlines()
            ax2.coastlines()
            self.plot(times=[time], ax=ax1)
            self.plot(times=[time], ax=ax2)

            self._anim_individual_frame(ax1, frames, precip[i], cn[i], tb[i], frame_norm, frame_precip_levels)
            self._anim_swath(ax2, frames, precip_swath, cn_swath, swath_norm, swath_precip_levels)
            self._anim_timeseries(ax3, i)

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

            lon_ticks, lat_ticks = ticks_from_min_max_lon_lat(*extent)

            ax1.set_yticks(lat_ticks)
            ax1.set_yticklabels(fmt_lat_ticklabels(lat_ticks))

            ax2.set_yticks(lat_ticks)
            ax2.set_yticklabels(fmt_lat_ticklabels(lat_ticks))

            ax2.set_xticks(lon_ticks)
            ax2.set_xticklabels(fmt_lon_ticklabels(lon_ticks))

            ax1.legend(handles=legend_elements)

            ax1.set_extent(extent)
            ax2.set_extent(extent)

            if savefigs:
                figpath = Path(figdir / f'anim_track/track_{self.track_id}/track_{self.track_id}_{i:03d}.png')
                figpath.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(figpath)
                figpaths.append(figpath)
            else:
                plt.pause(1)
                if user_input != 'c':
                    user_input = input('c for continue, q to quit, <enter> for step: ')
                    if user_input == 'q':
                        raise Exception('quit')
                    elif user_input == 'c':
                        print('<ctrl-c> to stop')
        return figpaths

    @staticmethod
    def _anim_individual_frame(ax, frames, precip_frame, cn_frame, tb_frame, frame_norm, frame_precip_levels):
        im1 = ax.contourf(
            frames.dspixel.lon,
            frames.dspixel.lat,
            precip_frame,
            cmap='Blues',
            norm=frame_norm,
            levels=frame_precip_levels,
        )
        ax.contour(
            frames.dspixel.lon,
            frames.dspixel.lat,
            cn_frame,
            levels=[0.5],
            colors='red',
        )
        ax.contour(
            frames.dspixel.lon,
            frames.dspixel.lat,
            tb_frame,
            levels=[241],
            colors='black',
        )
        plt.colorbar(im1, ax=ax, label='masked precip. (mm hr$^{-1}$)')

    @staticmethod
    def _anim_swath(ax, frames, precip_swath, cn_swath, swath_norm, swath_precip_levels):
        im2 = ax.contourf(
            frames.dspixel.lon,
            frames.dspixel.lat,
            precip_swath,
            cmap='Blues',
            norm=swath_norm,
            levels=swath_precip_levels,
        )
        ax.contour(
            frames.dspixel.lon,
            frames.dspixel.lat,
            cn_swath,
            levels=[0.5],
            colors='red',
        )

        cbar2 = plt.colorbar(im2, ax=ax, label='masked precip. swath (mm)')
        cbar2.set_ticks(swath_precip_levels)
        cbar2.set_ticklabels([f'{v:.2f}' for v in swath_precip_levels])

    def _anim_timeseries(self, ax, i):
        for name, data in [
            ('area', self.area),
            ('PF area', np.nanmean(self.pf_area, axis=1)),
            ('PF rainrate', np.nanmean(self.pf_rainrate, axis=1)),
        ]:
            ax.plot(data / np.nanmax(data), label=f'{name} (max={np.nanmax(data):.1f})')
        ax.set_xlim((-0.5, self.duration - 0.5))
        ax.set_ylim((0, 1.6))
        ax.set_yticks([0, 0.5, 1])
        ax.axvline(x=i)
        ax.legend(loc='upper right')
        ax.set_xlabel('time since start (hr)')

    @property
    def duration(self):
        return int(self.dstrack.track_duration.values.item())

    def __repr__(self):
        start = pd.Timestamp(self.dstrack.start_basetime.values)
        end = pd.Timestamp(self.dstrack.end_basetime.values)
        return f'McsTrack[{start}, {end}, id={self.track_id}, duration={self.duration}]'
