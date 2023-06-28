import cartopy.crs as ccrs
import cartopy.geodesic
from IPython.display import clear_output
import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np
import pandas as pd
import shapely.geometry
import shapely.ops
import xarray as xr

from remake.util import format_path as fmtp

from mcs_prime import McsTracks
from mcs_prime.mcs_prime_config_util import PATHS, lon_360_to_180, gen_region_masks
import mcs_prime.mcs_prime_config_util as cu

e5datadir = PATHS['era5dir'] / 'data/oper/an_sfc/{year}/{month:02d}/{day:02d}'


def create_geoms_from_mask(lon, lat, mask):
    assert len(lon) == mask.shape[1]
    assert len(lat) == mask.shape[0]
    dlat = lat[1] - lat[0]
    dlon = lon[1] - lon[0]

    lat_bounds = np.linspace(lat[0] - dlat / 2, lat[-1] + dlat / 2, len(lat) + 1)
    lon_bounds = np.linspace(lon[0] - dlon / 2, lon[-1] + dlon / 2, len(lon) + 1) % 360

    geoms = []
    for yidx, xidx in zip(*np.where(mask)):
        if abs(lon_bounds[xidx] - 180) < 0.5:
            continue
        geoms.append(shapely.geometry.box(
            lon_360_to_180(lon_bounds[xidx]),
            lat_bounds[yidx],
            lon_360_to_180(lon_bounds[xidx + 1]),
            lat_bounds[yidx + 1]
        ))
    full_geom = shapely.ops.unary_union(geoms)
    return full_geom.geoms


class McsMaskPlotterData:
    def __init__(self, times, e5vars):
        self.times = times
        self.e5vars = e5vars
        self.e5times = pd.DatetimeIndex(
            [t - pd.Timedelta(minutes=30) for t in self.times] +
            [self.times[-1] + pd.Timedelta(minutes=30)]
        )
        assert (self.times[0].year == self.times.year).all()
        year = self.times[0].year

        self.tracks_inputs = {'tracks': cu.fmt_mcs_stats_path(year)}
        self.e5inputs = {
            f'era5_{t}_{var}': fmtp(cu.FMT_PATH_ERA5_SFC, year=t.year, month=t.month, day=t.day, hour=t.hour, var=var)
            for t in self.e5times
            for var in e5vars
        }
        self.pixel_on_e5_inputs = {
            f'pixel_on_e5_{t}': fmtp(cu.FMT_PATH_PIXEL_ON_ERA5, year=t.year, month=t.month, day=t.day, hour=t.hour)
            for t in self.times
        }

        class Object: pass
        self.logger = Object()
        self.logger.info = print
        self.logger.debug = print

    def load(self):
        self.tracks = McsTracks.open(self.tracks_inputs['tracks'], None)
        self.pixel_on_e5 = xr.open_mfdataset(self.pixel_on_e5_inputs.values()).load()

        mcm, msm, ccm, csm, em = gen_region_masks(self.logger, self.pixel_on_e5, self.tracks)
        self.mcm = mcm
        self.msm = msm
        self.ccm = ccm
        self.csm = csm
        self.em = em

        self.e5data = (xr.open_mfdataset(self.e5inputs.values()).sel(latitude=slice(60, -60))
                       .interp(time=self.times)).load()

        self.all_geoms_at_time = {}
        for i, time in enumerate(self.times):
            self.all_geoms_at_time[time] = [
                create_geoms_from_mask(self.pixel_on_e5.longitude, self.pixel_on_e5.latitude, m[i])
                for m in [self.mcm, self.msm, self.ccm, self.csm]
            ]


class McsMaskPlotter:
    def __init__(self, plotter_data):
        self.plotter_data = plotter_data

    def animate_track(self, fig, ax, track_id):
        track = self.plotter_data.tracks.dstracks.isel(tracks=track_id)
        duration = track.track_duration.values.item()
        plot_args_kwargs = []
        track_times = pd.DatetimeIndex(track.base_time.values[:duration])
        for time, lat, lon in zip(track_times, track.meanlat.values[:duration], track.meanlon.values[:duration]):
            extent = (lon - 10, lon + 10, lat - 10, lat + 10)
            if time in self.plotter_data.times:
                plot_args_kwargs.append(((time, 'tcwv', extent), {}))
            else:
                print(f'Skipping {time}')
        return self.animate(fig, ax, plot_args_kwargs)

    def animate(self, fig, ax, plot_args_kwargs):
        def anim_frame(i):
            print(i)
            args, kwargs = plot_args_kwargs[i]
            ax.clear()
            self.plot(ax, *args, **kwargs)
            clear_output(wait=True)

        anim = matplotlib.animation.FuncAnimation(fig, anim_frame, frames=len(plot_args_kwargs), interval=500)
        return anim

    def plot(self, ax=None, time=None, var='tcwv', extent=None,
             show_field=True, show_colourbar=True, show_radii=True, show_mcs_masks=True, show_precip=True,
             grid_x=[], grid_y=[]):
        if ax is None:
            fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
            fig.set_size_inches(10, 7.2)
        if time is None:
            time = self.plotter_data.times[0]
        if extent is None:
            extent = (-180, 180, -60, 60)
        lonmin, lonmax, latmin, latmax = extent

        idx = np.where(self.plotter_data.times == time)[0][0]

        pixel_on_e5 = self.plotter_data.pixel_on_e5.sel(time=time)
        e5data = self.plotter_data.e5data.sel(time=time)

        all_geoms = self.plotter_data.all_geoms_at_time[time]

        lonmin_idx = np.argmin(np.abs(lon_360_to_180(e5data.longitude.values) - lonmin))
        lonmax_idx = np.argmin(np.abs(lon_360_to_180(e5data.longitude.values) - lonmax))
        latmin_idx = np.argmin(np.abs(e5data.latitude.values - latmin))
        latmax_idx = np.argmin(np.abs(e5data.latitude.values - latmax))
        # print(lonmin_idx, lonmax_idx, latmin_idx, latmax_idx)

        s = (slice(latmax_idx, latmin_idx), slice(lonmin_idx, lonmax_idx))
        var_data = e5data[var][s]

        # im = ax.contourf(var_data.longitude, var_data.latitude, var_data, levels=np.linspace(0, 70, 15))
        # lonmin, lonmax = lon_360_to_180(var_data.longitude.values[[0, -1]])
        # latmin, latmax = var_data.latitude.values[[-1, 0]]
        # print(lonmin, lonmax, latmin, latmax)

        lat = pixel_on_e5.latitude.values
        lon = pixel_on_e5.longitude.values
        dlat = lat[1] - lat[0]
        dlon = lon[1] - lon[0]
        if show_precip:
            ax.contour(pixel_on_e5.longitude, pixel_on_e5.latitude, pixel_on_e5.precipitation, levels=[2, 5, 10], colors=['purple', 'purple', 'purple'], zorder=5)
        if show_field:
            im = ax.imshow(var_data, extent=(lonmin - dlon / 2, lonmax + dlon / 2, latmin + dlat / 2, latmax - dlat / 2))

        if show_mcs_masks:
            for geoms, kwargs1, kwargs2 in zip(
                all_geoms,
                [
                    dict(facecolor='red', alpha=0.3),
                    dict(facecolor='red', alpha=0),
                    dict(facecolor='blue', alpha=0.3),
                    dict(facecolor='blue', alpha=0),
                ],
                [
                    dict(edgecolor='red', linestyle='-', hatch='//'),
                    dict(edgecolor='red', linestyle='--', hatch=r'\\'),
                    dict(edgecolor='blue', linestyle='-', hatch='//'),
                    dict(edgecolor='blue', linestyle='--', hatch=r'\\'),
                ]
            ):
                # geoms = [g for g in geoms if not filter_geom(g)]
                # geoms = create_geoms_from_mask(pixel_on_e5.longitude, pixel_on_e5.latitude, m[0])
                ax.add_geometries(
                    geoms,
                    crs=cartopy.crs.PlateCarree(),
                    linewidth=0,
                    **kwargs1,
                )
                ax.add_geometries(
                    geoms,
                    crs=cartopy.crs.PlateCarree(),
                    facecolor='none',
                    linewidth=2,
                    **kwargs2,
                )

        ax.coastlines()
        if grid_x and grid_y:
            ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                         xlocs=grid_x, ylocs=grid_y)

        if show_colourbar:
            if var == 'tcwv':
                plt.colorbar(im, orientation='horizontal', label='TCWV (mm)')
            elif var == 'cape':
                plt.colorbar(im, orientation='horizontal', label='CAPE (J kg$^{-1}$)')
        # ax.contour(e5data.longitude, e5data.latitude, e5data.tcwv)

        ax.set_xlim((lonmin, lonmax))
        ax.set_ylim((latmin, latmax))

        if show_radii:
            mask = self.plotter_data.tracks.dstracks.base_time == time
            lats = self.plotter_data.tracks.dstracks.meanlat.values[mask]
            lons = self.plotter_data.tracks.dstracks.meanlon.values[mask]
            track_ids = self.plotter_data.tracks.dstracks.tracks.values[mask.any(axis=1)]

            ax.scatter(lons, lats, marker='o', color='g', zorder=10)

            geoms = []
            for lat, lon, track_id in zip(lats, lons, track_ids):
                if not (lat > latmin and lat < latmax and lon > lonmin and lon < lonmax):
                    continue
                ax.text(lon + 0.1, lat + 0.1, f'{track_id}', color='g')
                for radius in [100, 200]:
                    circle_points = cartopy.geodesic.Geodesic().circle(
                        lon=lon, lat=lat, radius=radius * 1e3, n_samples=100, endpoint=False
                    )
                    geom = shapely.geometry.Polygon(circle_points)
                    geoms.append(geom)

            ax.add_geometries(
                geoms,
                crs=cartopy.crs.PlateCarree(),
                facecolor="none",
                edgecolor="g",
                linewidth=2,
            )
