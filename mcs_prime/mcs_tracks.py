import datetime as dt
import math

import cartopy
import cartopy.geodesic
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely.geometry
import shapely.ops
import xarray as xr

from .mcs_prime_config_util import PATHS, round_times_to_nearest_second


def _create_fig_ax():
    fig, ax = plt.subplots(subplot_kw=dict(projection=cartopy.crs.PlateCarree()))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
    ax.coastlines()
    return fig, ax


class PixelData:
    def __init__(self, pixel_data_dir):
        track_pixel_paths = sorted(pixel_data_dir.glob("**/*.nc"))
        self.date_path_map = {
            dt.datetime.strptime(p.stem, "mcstrack_%Y%m%d_%H%M"): p
            for p in track_pixel_paths
        }
        if len(self.date_path_map) == 0:
            raise Exception(f"No pixel data found in: {pixel_data_dir}")

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
                print(f"Warning: missing path for time: {time}")
            times.append(time)
            time += dt.timedelta(hours=1)
        return PixelFrames(times, xr.open_mfdataset(paths))


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
        return minlon, maxlon, minlat, maxlat

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
        method="contourf",
        method_kwargs=None,
        cloudnumbers=None,
        ax=None,
        zoom=False,
    ):
        if method not in ["contour", "contourf"]:
            raise ValueError("method must be one of 'contour', 'contourf'")
        if not ax:
            fig, ax = _create_fig_ax()
            ax.set_title(f"{field} @ {self.time}")
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
        getattr(ax, method)(
            self.dspixel.lon, self.dspixel.lat, mean_data, **method_kwargs
        )


class PixelFrame:
    def __init__(self, time, dspixel):
        self.time = time
        self.dspixel = dspixel

    def plot(
        self,
        field,
        method="contourf",
        method_kwargs=None,
        cloudnumber=None,
        ax=None,
        zoom=False,
    ):
        if method not in ["contour", "contourf"]:
            raise ValueError("method must be one of 'contour', 'contourf'")
        if not ax:
            fig, ax = _create_fig_ax()
            ax.set_title(f"{field} @ {self.time}")
        data = getattr(self.dspixel, field)[0].values
        if cloudnumber:
            cloudmask = (self.dspixel.cloudnumber == cloudnumber)[0]
            print(cloudmask.sum())
            if zoom:
                minlon = self.dspixel.longitude.values[cloudmask].min()
                maxlon = self.dspixel.longitude.values[cloudmask].max()
                minlat = self.dspixel.latitude.values[cloudmask].min()
                maxlat = self.dspixel.latitude.values[cloudmask].max()
                ax.set_extent([minlon, maxlon, minlat, maxlat])
            data = data.copy()
            data[~cloudmask] = 0

        if not method_kwargs:
            method_kwargs = {}
        getattr(ax, method)(self.dspixel.lon, self.dspixel.lat, data, **method_kwargs)


class McsTracks:
    @classmethod
    def mfopen(cls, stats_paths=None, pixeldir=None, round_times=True):
        if stats_paths is None:
            stats_paths = sorted(
                PATHS["statsdir"].glob(
                    "mcs_tracks_final_extc_????????.0000_????????.0000.nc"
                )
            )
        if pixeldir is None:
            pixeldir = PATHS["pixeldir"]
        dstracks = xr.open_mfdataset(
            stats_paths,
            concat_dim="tracks",
            combine="nested",
            mask_and_scale=False,
        )
        dstracks["tracks"] = np.arange(0, dstracks.dims["tracks"], 1, dtype=int)
        if round_times:
            round_times_to_nearest_second(
                dstracks, ["base_time", "start_basetime", "end_basetime"]
            )
        pixel_data = PixelData(pixeldir)
        return cls(dstracks, pixel_data)

    @classmethod
    def open(cls, stats_path, pixeldir=None, round_times=True):
        if pixeldir is None:
            pixeldir = PATHS["pixeldir"]
        dstracks = xr.open_dataset(stats_path)
        if round_times:
            round_times_to_nearest_second(
                dstracks, ["base_time", "start_basetime", "end_basetime"]
            )
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

    def tracks_at_time(self, datetime):
        datetime = pd.Timestamp(datetime).to_numpy()
        dstracks_at_time = self.dstracks.isel(
            tracks=(self.dstracks.base_time.values == datetime).any(axis=1)
        )
        return McsTracks(dstracks_at_time, self.pixel_data)

    def land_sea_both_tracks(self, land_ratio=0.9, sea_ratio=0.1):
        if land_ratio < sea_ratio:
            # If this is the case, some tracks will be double counted.
            raise ValueError("land_ratio must be greater than sea_ratio")
        track_mean_pf_landfrac = self.dstracks.pf_landfrac.mean(dim="times").values
        land_mask = track_mean_pf_landfrac > land_ratio
        sea_mask = track_mean_pf_landfrac <= sea_ratio
        both_mask = (track_mean_pf_landfrac <= land_ratio) & (
            track_mean_pf_landfrac > sea_ratio
        )

        land_tracks = McsTracks(self.dstracks.isel(tracks=land_mask), self.pixel_data)
        sea_tracks = McsTracks(self.dstracks.isel(tracks=sea_mask), self.pixel_data)
        if both_mask.sum() != 0:
            both_tracks = McsTracks(
                self.dstracks.isel(tracks=both_mask), self.pixel_data
            )
        else:
            both_tracks = None
        return land_tracks, sea_tracks, both_tracks

    def plot(
        self,
        ax=None,
        display_area=True,
        display_pf_area=True,
        times="all",
        colour="g",
        linestyle="-",
    ):
        if not ax:
            fig, ax = _create_fig_ax()
            ax.set_title(f"{self}")
        if times != "all" and len(times) == 1 and self.pixel_data is not None:
            frame = self.pixel_data.get_frame(times[0])
            frame.plot("cloudnumber", ax=ax)

        for i, track_id in enumerate(self.dstracks.tracks):
            track = self.get_track(track_id.values.item())
            print(f"{track}: {i + 1}/{len(self.dstracks.tracks)}")
            track.plot(ax, display_area, display_pf_area, times, colour, linestyle)

    @property
    def start(self):
        return pd.Timestamp(self.dstracks.start_basetime.values.min())

    @property
    def end(self):
        return pd.Timestamp(self.dstracks.end_basetime.values.max())

    def __repr__(self):
        return (
            f"McsTracks[{self.start}, {self.end}, ntracks={len(self.dstracks.tracks)}]"
        )

    def _repr_html_(self):
        return (
            f"McsTracks[{self.start}, {self.end}, ntracks={len(self.dstracks.tracks)}]"
        )


class McsTrack:
    def __init__(self, track_id, dstrack, pixel_data):
        self.track_id = track_id
        self.dstrack = dstrack
        self.pixel_data = pixel_data

    def load(self):
        self.dstrack.load()

    def __getattr__(self, attr):
        if attr not in self.dstrack.variables:
            raise AttributeError(f"Not found: '{attr}'")
        val = getattr(self.dstrack, attr)
        return val.values[: self.duration]

    def plot(
        self,
        ax=None,
        display_area=True,
        display_pf_area=True,
        times="all",
        colour="g",
        linestyle="-",
        use_status_for_marker=False,
    ):
        self.load()
        if not ax:
            fig, ax = _create_fig_ax()
            ax.set_title(f"{self}")
        if times != "all" and len(times) == 1 and self.pixel_data is not None:
            time_index = pd.Timestamp(times[0]).to_numpy()
            cloudnumber = self.cloudnumber[self.base_time == time_index]
            frame = self.pixel_data.get_frame(times[0])
            frame.plot(
                "precipitation",
                "contour",
                cloudnumber=cloudnumber,
                ax=ax,
                zoom=True,
            )

        ax.plot(self.meanlon, self.meanlat, color=colour, linestyle=linestyle)

        for i in [0, self.duration - 1]:
            ts = pd.Timestamp(self.base_time[i])
            marker = f"id:{self.track_id} - {ts.day}.{ts.hour}:{ts.minute}"
            ax.annotate(
                marker,
                (self.meanlon[i], self.meanlat[i]),
                fontsize="x-small",
            )

        if times == "all":
            time_indices = range(self.duration)
        else:
            time_indices = []
            for time in [pd.Timestamp(t).to_numpy() for t in times]:
                time_indices.append(np.where(self.base_time == time)[0].item())

        for i in range(self.duration):
            lon = self.meanlon[i]
            lat = self.meanlat[i]
            if use_status_for_marker:
                marker = f"{int(self.track_status[i])}"
                ax.annotate(
                    marker,
                    (lon, lat),
                    fontsize="small",
                )
            elif times != "all":
                marker = "o"
                ax.plot(lon, lat, color=colour, linestyle="None", marker=marker)

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
                    facecolor="none",
                    edgecolor="grey",
                    linewidth=2,
                )
            except ValueError as ve:
                print(f"Warning: {ve}")
                # This can happen, perhaps if MCS geom stradles -180?
                if ve.args[0] != "No Shapely geometry can be created from null value":
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
                    facecolor="none",
                    edgecolor="blue",
                    linewidth=1,
                )
            except ValueError as ve:
                print(f"Warning: {ve}")
                # This can happen, perhaps if MCS geom stradles -180?
                if ve.args[0] != "No Shapely geometry can be created from null value":
                    raise

    @property
    def duration(self):
        return int(self.dstrack.track_duration.values.item())

    def __repr__(self):
        start = pd.Timestamp(self.dstrack.start_basetime.values)
        end = pd.Timestamp(self.dstrack.end_basetime.values)
        return f"McsTrack[{start}, {end}, id={self.track_id}, duration={self.duration}]"
