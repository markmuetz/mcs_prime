import gc
import datetime as dt
import math
from pathlib import Path

import cartopy
import cartopy.geodesic
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import shapely
import shapely.ops
import xarray as xr

from .util import round_times_to_nearest_second
from .mcs_prime_config import status_dict


def round_away_from_zero_to_sigfig(val, nsf):
    sign = 1 if val >= 0 else -1
    factor = 10 ** (nsf - math.ceil(math.log10(sign * val)))
    newval = math.ceil(sign * val * factor) / factor
    return sign * newval


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
        if len(cloudnumbers) == 1:
            cloudmask = self.dspixel.cloudnumber[0] == cloudnumbers[0]
            minlon = min(lon[cloudmask].min(), minlon)
            maxlon = max(lon[cloudmask].max(), maxlon)
            minlat = min(lat[cloudmask].min(), minlat)
            maxlat = max(lat[cloudmask].max(), maxlat)
        else:
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
    def mfload(cls, stats_paths, pixeldir, round_times=True):
        dstracks = xr.open_mfdataset(
            stats_paths,
            concat_dim='tracks',
            combine='nested',
            mask_and_scale=False,
        )
        dstracks['tracks'] = np.arange(0, dstracks.dims['tracks'], 1, dtype=int)
        if round_times:
            dstracks.base_time.load()
            dstracks.start_basetime.load()
            dstracks.end_basetime.load()
            round_times_to_nearest_second(dstracks)
        pixel_data = PixelData(pixeldir)
        return cls(dstracks, pixel_data)

    @classmethod
    def load(cls, stats_path, pixeldir, round_times=True):
        dstracks = xr.open_dataset(stats_path)
        if round_times:
            round_times_to_nearest_second(dstracks)
        pixel_data = PixelData(pixeldir)
        return cls(dstracks, pixel_data)

    def __init__(self, dstracks, pixel_data=None):
        self.dstracks = dstracks
        self.pixel_data = pixel_data

    def get_track(self, track_id):
        return McsTrack(track_id, self.dstracks.sel(tracks=track_id), self.pixel_data)

    def get_lon_lat_for_local_solar_time(self, t0, t1):
        base_time = pd.DatetimeIndex(self.dstracks.base_time.values.flatten())
        meanlon = self.dstracks.meanlon.values.flatten()
        meanlat = self.dstracks.meanlat.values.flatten()

        lst_offset_hours = meanlon / 180 * 12
        time_hours = (base_time.hour + base_time.minute / 60).values
        lst = time_hours + lst_offset_hours
        lst = lst % 24
        lst_mask = (lst > t0) & (lst < t1)
        nan_mask = ~np.isnan(base_time)
        mask = lst_mask & nan_mask
        return meanlon[mask], meanlat[mask]

    def plot_diurnal_cycle(self, dhours=6, mode='mean'):
        sizes = {1: (4, 6), 2: (4, 3), 3: (4, 2), 4: (2, 3), 6: (2, 2), 24: (1, 1)}
        size = sizes[dhours]
        fig, axes = plt.subplots(
            size[0], size[1], subplot_kw=dict(projection=cartopy.crs.PlateCarree()), sharex=True, sharey=True
        )
        if dhours == 24:
            axes = np.array([axes])
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
        xmin = -180
        xmax = 180
        ymin = -60
        ymax = 60
        dx = 4
        dy = 4
        bx = np.arange(xmin, xmax + dx, dx)
        by = np.arange(ymin, ymax + dy, dy)
        bxmid = (bx[1:] + bx[:-1]) / 2
        bymid = (by[1:] + by[:-1]) / 2

        nan_mask = ~np.isnan(self.dstracks.meanlon.values)
        meanlon = self.dstracks.meanlon.values[nan_mask]
        meanlat = self.dstracks.meanlat.values[nan_mask]

        if mode in ['anomaly', 'anomaly_frac']:
            mean_hist, _, _ = np.histogram2d(meanlon, meanlat, bins=(bx, by), density=True)
            diff_hists = {}

        hists = {}

        base_time = pd.DatetimeIndex(self.dstracks.base_time.values[nan_mask])

        lst_offset_hours = meanlon / 180 * 12
        time_hours = (base_time.hour + base_time.minute / 60).values
        lst = time_hours + lst_offset_hours
        lst = lst % 24

        for i, ax in enumerate(axes.flatten()):
            hour = i * dhours
            ax.coastlines()
            ax.set_title(f'LST: {hour} - {hour + dhours}')

            lst_mask = (lst > hour) & (lst < hour + dhours)
            lon = meanlon[lst_mask]
            lat = meanlat[lst_mask]

            hist, _, _ = np.histogram2d(lon, lat, bins=(bx, by), density=True)
            hists[i] = hist
            if mode == 'anomaly':
                diff_hists[i] = hist - mean_hist
            elif mode == 'anomaly_frac':
                diff_hists[i] = hist / mean_hist

        if mode == 'anomaly':
            diff_hist_min = min([np.nanmin(diff_hists[i]) for i, ax in enumerate(axes.flatten())])
            diff_hist_max = max([np.nanmax(diff_hists[i]) for i, ax in enumerate(axes.flatten())])

            abs_max = round_away_from_zero_to_sigfig(max(abs(diff_hist_min), abs(diff_hist_max)), 2)

            levels = np.array([-1, -5e-1, -2e-1, -1e-1, -5e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1]) * abs_max
            cmap = plt.get_cmap('RdBu_r').copy()
        elif mode == 'anomaly_frac':
            levels = np.array([0, 1 / 2, 2 / 3, 4 / 5, 0.9, 0.95, 1.05, 1.1, 5 / 4, 3 / 2, 2, 10])
            cmap = plt.get_cmap('RdBu_r').copy()
        else:
            cmap = plt.get_cmap('Spectral_r').copy()
            levels = np.array([0, 2, 3, 5, 8, 10, 15, 20, 25, 30, 35, 40, 50, 60, 80, 90, 100]) / 100 * hist.max()
        norm = colors.BoundaryNorm(boundaries=levels, ncolors=256)

        for i, ax in enumerate(axes.flatten()):
            if mode in ['anomaly', 'anomaly_frac']:
                hist = diff_hists[i]
            else:
                hist = hists[i]
            extent = (xmin, xmax, ymin, ymax)
            im = ax.imshow(hist.T, origin='lower', extent=extent, cmap=cmap, norm=norm)

        plt.colorbar(im, ax=axes, label=f'MCS density {mode}')

    def tracks_at_time(self, datetime):
        datetime = pd.Timestamp(datetime).to_numpy()
        dstracks_at_time = self.dstracks.isel(tracks=(self.dstracks.base_time.values == datetime).any(axis=1))
        return McsTracks(dstracks_at_time, self.pixel_data)

    def land_sea_both_tracks(self, land_ratio=0.9, sea_ratio=0.1):
        if land_ratio < sea_ratio:
            # If this is the case, some tracks will be double counted.
            raise ValueError('land_ratio must be greater than sea_ratio')
        track_mean_pf_landfrac = self.dstracks.pf_landfrac.mean(dim='times').values
        land_mask = track_mean_pf_landfrac > land_ratio
        sea_mask = track_mean_pf_landfrac <= sea_ratio
        both_mask = (track_mean_pf_landfrac <= land_ratio) & (track_mean_pf_landfrac > sea_ratio)

        land_tracks = McsTracks(self.dstracks.isel(tracks=land_mask), self.pixel_data)
        sea_tracks = McsTracks(self.dstracks.isel(tracks=sea_mask), self.pixel_data)
        if both_mask.sum() != 0:
            both_tracks = McsTracks(self.dstracks.isel(tracks=both_mask), self.pixel_data)
        else:
            both_tracks = None
        return land_tracks, sea_tracks, both_tracks

    def plot(self, ax=None, display_area=True, display_pf_area=True, times='all', colour='g', linestyle='-'):
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
            track.plot(ax, display_area, display_pf_area, times, colour, linestyle)

    @property
    def start(self):
        return pd.Timestamp(self.dstracks.start_basetime.values.min())

    @property
    def end(self):
        return pd.Timestamp(self.dstracks.end_basetime.values.max())

    def __repr__(self):
        return f'McsTracks[{self.start}, {self.end}, ntracks={len(self.dstracks.tracks)}]'

    def _repr_html_(self):
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

    def plot(self, ax=None, display_area=True, display_pf_area=True, times='all', colour='g', linestyle='-', use_status_for_marker=False):
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

        ax.plot(self.meanlon, self.meanlat, color=colour, linestyle=linestyle)

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


        for i in range(self.duration):
            lon = self.meanlon[i]
            lat = self.meanlat[i]
            if use_status_for_marker:
                marker = f'{int(self.track_status[i])}'
                ax.annotate(
                    marker,
                    (lon, lat),
                    fontsize='small',
                )
            elif times != 'all':
                marker = 'o'
                ax.plot(lon, lat, color=colour, linestyle='None', marker=marker)

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
            try:
                full_geom = shapely.ops.unary_union(geoms)
                ax.add_geometries(
                    (full_geom,),
                    crs=cartopy.crs.PlateCarree(),
                    facecolor='none',
                    edgecolor='grey',
                    linewidth=2,
                )
            except ValueError as ve:
                print(f'Warning: {ve}')
                # This can happen, perhaps if MCS geom stradles -180?
                if ve.args[0] != 'No Shapely geometry can be created from null value':
                    raise
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
            try:
                full_geom = shapely.ops.unary_union(geoms)
                ax.add_geometries(
                    (full_geom,),
                    crs=cartopy.crs.PlateCarree(),
                    facecolor='none',
                    edgecolor='blue',
                    linewidth=1,
                )
            except ValueError as ve:
                print(f'Warning: {ve}')
                # This can happen, perhaps if MCS geom stradles -180?
                if ve.args[0] != 'No Shapely geometry can be created from null value':
                    raise

    def animate(self, method='contourf', method_kwargs=None, zoom='swath', savefigs=False, figdir=None):
        user_input = ''
        times = pd.DatetimeIndex(self.base_time)
        start = pd.Timestamp(self.dstrack.start_basetime.values).to_pydatetime()
        end = pd.Timestamp(self.dstrack.end_basetime.values).to_pydatetime()

        cloudnumbers = self.cloudnumber
        frames = self.pixel_data.get_frames(start, end)
        swath_extent = frames.get_min_max_lon_lat(cloudnumbers)
        dlon = swath_extent[1] - swath_extent[0]
        dlat = swath_extent[3] - swath_extent[2]
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
        i = 0
        while i < len(cloudnumbers) and i >= 0:
            cloudnumber = cloudnumbers[i]
            time = times[i]
            print(time)
            fig.clf()
            ax1 = fig.add_subplot(3, 1, 1, projection=cartopy.crs.PlateCarree())
            ax2 = fig.add_subplot(3, 1, 2, projection=cartopy.crs.PlateCarree())
            ax3 = fig.add_subplot(3, 1, 3)

            status = int(self.track_status[i])
            ax1.set_title(f'Track {self.track_id} @ {time} ({status=})')
            ax2.set_title(f'Track {self.track_id} swath')
            ax1.coastlines()
            ax2.coastlines()
            self.plot(times=[time], ax=ax1, use_status_for_marker=True)
            self.plot(times=[time], ax=ax2, use_status_for_marker=True)

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

            if zoom == 'swath':
                extent = swath_extent
            elif zoom == 'current':
                frames = self.pixel_data.get_frames(time, time)
                extent = frames.get_min_max_lon_lat([cloudnumber])
            lon_ticks, lat_ticks = ticks_from_min_max_lon_lat(*extent)

            ax1.set_yticks(lat_ticks)
            ax1.set_yticklabels(fmt_lat_ticklabels(lat_ticks))

            ax2.set_yticks(lat_ticks)
            ax2.set_yticklabels(fmt_lat_ticklabels(lat_ticks))

            ax2.set_xticks(lon_ticks)
            ax2.set_xticklabels(fmt_lon_ticklabels(lon_ticks))

            ax1.set_extent(extent)
            ax2.set_extent(extent)

            ax1.legend(handles=legend_elements)


            if savefigs:
                figpath = Path(figdir / f'anim_track/track_{self.track_id}/track_{self.track_id}_{i:03d}.png')
                figpath.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(figpath)
                figpaths.append(figpath)
            else:
                plt.pause(1)
                print(status, status_dict[status])
                if user_input != 'c':
                    user_input = input('c for continue, q to quit, p for prev, <enter> for step: ')
                    if user_input == 'q':
                        raise Exception('quit')
                    elif user_input == 'p':
                        i -= 1
                    elif user_input == 'c':
                        print('<ctrl-c> to stop')
                        i += 1
                    else:
                        i += 1
                else:
                    i += 1
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
