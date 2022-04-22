"""Experimental code for accessing V2 global tracking data"""
from collections import Counter
import datetime as dt
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

# From dspf.trackresult.attrs['values'].split(';')
# ['-1 = default (not MCS)',
#  ' 0 = cp from previous file dissipated',
#  ' 1 = cp track from previous file continues',
#  ' 2 = cp is the larger fragment of a merger',
#  ' 3 = cp is the larger fragment of a split',
#  ' 10 = start of a new track',
#  ' 12 = cp is the smaller fragment of a merger',
#  ' 13 = cp is the smaller fragment of a split']
# TODO: These are out-of-date -- applied to v1
TRACK_RESULT = {
    -1: 'default (not MCS)',
    0: 'cloud from previous file dissipated',
    1: 'cloud track from previous file continues',
    2: 'cloud is the larger fragment of a merger',
    3: 'cloud is the larger fragment of a split',
    10: 'start of a new track',
    12: 'cloud is the smaller fragment of a merger',
    13: 'cloud is the smaller fragment of a split'
}
TRACK_RESULT_COLOUR_MARKER = {
    -1: ('m', 'X'),
    0: ('k', '+'),
    1: ('k', 'o'),
    2: ('r', '*'),
    3: ('r', '^'),
    10: ('k', 'p'),
    12: ('b', '*'),
    13: ('b', '^'),
}

def round_times_to_nearest_second(dspf):
    """Round times in dspf.base_time to the nearest second.

    Sometimes the dspf dataset has minor inaccuracies in the time, e.g.
    '2000-06-01T00:30:00.000013440' (13440 ns). Remove these.

    :param dspf: xarray.Dataset to convert.
    :return: None
    """

    # N.B. I tried to do this using pure np funcions, but could not work
    # out how to convert np.int64 into a np.datetime64. Seems like it should be
    # easy.

    def remove_time_incaccuracy(t):
        return np.datetime64(int(round(t / 1e9) * 1e9), 'ns')

    vec_remove_time_incaccuracy = np.vectorize(remove_time_incaccuracy)
    tmask = ~np.isnan(dspf.base_time.values)
    dspf.base_time.values[tmask] = vec_remove_time_incaccuracy(dspf.base_time.values[tmask].astype(int))


def tracks_for_days(dspf, date, ndays=1):
    """Return tracks for date+ given number of days"""

    date = np.datetime64(date)
    endday = np.datetime64(date) + np.timedelta64(ndays, 'D')
    track_start_time = dspf.base_time.values[:, 0]
    return dspf.isel(tracks=(track_start_time > date) & (track_start_time < endday))


def display_track_stats(dspf):
    """Display useful stats about tracks"""
    num_tracks = len(dspf.tracks)
    print('Tracks:', num_tracks)
    num_track_points = int(dspf.track_duration.sum().compute().values.item())
    print('Track points:', num_track_points)

    print('=' * 10, 'trackresults', '=' * 10)
    trackresult_stats = Counter(dspf.track_status.values.flatten())
    for k, v in TRACK_RESULT.items():
        print(f'{v:<41}: {trackresult_stats[k]}')
    assert num_track_points == sum(v for k, v in trackresult_stats.items() if k != -999)

    print('=' * 10, 'starttrackresults', '=' * 10)
    starttrackresult_stats = Counter(dspf.starttrackresult.values)
    for k, v in TRACK_RESULT.items():
        print(f'{v:<41}: {starttrackresult_stats[k]}')
    assert num_tracks == sum(v for k, v in starttrackresult_stats.items())

    print('=' * 10, 'endtrackresults', '=' * 10)
    endtrackresult_stats = Counter(dspf.endtrackresult.values)
    for k, v in TRACK_RESULT.items():
        print(f'{v:<41}: {endtrackresult_stats[k]}')
    assert num_tracks == sum(v for k, v in endtrackresult_stats.items())

def check_cloudnumbers(dspf):
    date_cn = []
    dspf.base_time.load()
    dspf.cloudnumber.load()
    for track_index, track_id in enumerate(dspf.tracks):
        track = dspf.sel(tracks=track_id)
        track_duration = int(track.track_duration.values.item())
        print(f'{track_index + 1}/{len(dspf.tracks)}', track.base_time.values[0], track_duration)
        for i in range(track_duration):
            date_cn.append((track.base_time.values[i], track.cloudnumber.values[i]))
    assert len(date_cn) == len(set(date_cn))


def plot_tracks_at_time(dspf, dspixel, time, cloudnumber=None, ax=None,
                        display_track=True, display_area=True, display_pf_area=True, display_trackresult=False):
    if not ax:
        fig, ax = plt.subplots(subplot_kw=dict(projection=cartopy.crs.PlateCarree()))
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)

    title = f'{len(dspf.tracks)} tracks: {time}'
    ax.set_title(title)

    if cloudnumber:
        ax.contourf(dspixel.lon, dspixel.lat, dspixel.cloudnumber[0] == cloudnumber)
    else:
        ax.contourf(dspixel.lon, dspixel.lat, dspixel.cloudnumber[0])
    # ax.contour(dspixel.lon, dspixel.lat, dspixel.precipitation[0])

    for track_index, track_id in enumerate(dspf.tracks):
        track = dspf.sel(tracks=track_id)
        time_index = np.where(track.base_time.values == time)[0].item()

        if cloudnumber and track.cloudnumber[time_index] != cloudnumber:
            continue

        track_duration = int(track.track_duration.values.item())
        print(f'{track_index + 1}/{len(dspf.tracks)}', track.base_time.values[0], track_duration)

        legend_elements = [Line2D([0], [0], color='g', label='track')]

        if display_track:
            ax.plot(track.meanlon[:track_duration], track.meanlat[:track_duration], 'g-')
            colour_marker = TRACK_RESULT_COLOUR_MARKER.get(track.start_status.values.item(), ('r', 'x'))
            if colour_marker:
                ax.plot(track.meanlon[0], track.meanlat[0], ''.join(colour_marker))

            colour_marker = TRACK_RESULT_COLOUR_MARKER.get(track.end_status.values.item(), ('r', 'x'))
            if colour_marker:
                ax.plot(track.meanlon[track_duration - 1], track.meanlat[track_duration - 1], ''.join(colour_marker))

        ts = pd.Timestamp(track.base_time[time_index].values)
        marker = f'{ts.day}-{ts.hour}:{ts.minute}'
        ax.annotate(marker, (track.meanlon[time_index].values, track.meanlat[time_index].values), fontsize='x-small')

        n_points = 20
        if display_area:
            legend_elements.append(
                Patch(facecolor='none', edgecolor='grey', label='CCS')
            )
            geoms = []
            lon = track.meanlon[time_index]
            lat = track.meanlat[time_index]
            radius = np.sqrt(track.ccs_area[time_index].values.item() / np.pi) * 1e3
            if lon < -170:
                continue
            if np.isnan(radius):
                continue
            circle_points = cartopy.geodesic.Geodesic().circle(lon=lon, lat=lat,
                                                               radius=radius, n_samples=n_points,
                                                               endpoint=False)
            geom = shapely.geometry.Polygon(circle_points)
            geoms.append(geom)
            full_geom = shapely.ops.unary_union(geoms)
            ax.add_geometries((full_geom,), crs=cartopy.crs.PlateCarree(),
                              facecolor='none', edgecolor='grey', linewidth=2)
        if display_pf_area:
            legend_elements.append(
                Patch(facecolor='none', edgecolor='b', label='PF')
            )
            geoms = []
            for j in range(3):
                lon = track.pf_lon[time_index, j]
                lat = track.pf_lat[time_index, j]
                radius = np.sqrt(track.pf_area[time_index, j].values.item() / np.pi) * 1e3
                if np.isnan(radius):
                    continue
                circle_points = cartopy.geodesic.Geodesic().circle(lon=lon, lat=lat,
                                                                   radius=radius,
                                                                   n_samples=n_points,
                                                                   endpoint=False)
                geom = shapely.geometry.Polygon(circle_points)
                geoms.append(geom)
            full_geom = shapely.ops.unary_union(geoms)
            ax.add_geometries((full_geom,), crs=cartopy.crs.PlateCarree(),
                              facecolor='none', edgecolor='blue', linewidth=1)

        if display_trackresult:
            # legend_elements.extend([
            #     Line2D([0], [0], marker=cm[1], color='none', label=TRACK_RESULT[k],
            #            markerfacecolor=cm[0], markeredgecolor=cm[0])
            #     for k, cm in TRACK_RESULT_COLOUR_MARKER.items()
            #     if cm
            # ])
            track_status = int(track.track_status.values[time_index])
            colour_marker = ('b', f'{track_status}')
            if colour_marker:
                ax.plot(track.meanlon[time_index], track.meanlat[time_index], ''.join(colour_marker))
        if legend_elements:
            ax.legend(handles=legend_elements)

    ax.coastlines()
    plt.show()


def plot_tracks(dspf, ax=None, display_area=True, display_pf_area=True, display_trackresult=True):
    if not ax:
        fig, ax = plt.subplots(subplot_kw=dict(projection=cartopy.crs.PlateCarree()))
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
    # display_track_stats(dspf)
    print()
    # This makes a huge speed difference.
    dspf.base_time.load()
    dspf.meanlon.load()
    dspf.meanlat.load()
    dspf.track_status.load()
    dspf.ccs_area.load()
    dspf.pf_area.load()
    dspf.pf_lon.load()
    dspf.pf_lat.load()

    start_track = dspf.isel(tracks=0)
    end_track = dspf.isel(tracks=-1)
    # How are you meant to format datetime64s? Not like this...
    start_time = f'{start_track.base_time.values[0]}'[:16]
    end_time = f'{end_track.base_time.values[int(end_track.track_duration.values.item()) - 1]}'[:16]
    title = f'{len(dspf.tracks)} tracks: {start_time} - {end_time}'
    ax.set_title(title)

    for track_index, track_id in enumerate(dspf.tracks):
        track = dspf.sel(tracks=track_id)
        track_duration = int(track.track_duration.values.item())
        print(f'{track_index + 1}/{len(dspf.tracks)}', track.base_time.values[0], track_duration)

        legend_elements = [Line2D([0], [0], color='g', label='track')]

        ax.plot(track.meanlon[:track_duration], track.meanlat[:track_duration], 'g-')
        colour_marker = TRACK_RESULT_COLOUR_MARKER.get(track.start_status.values.item(), ('r', 'x'))
        if colour_marker:
            ax.plot(track.meanlon[0], track.meanlat[0], ''.join(colour_marker))

        colour_marker = TRACK_RESULT_COLOUR_MARKER.get(track.end_status.values.item(), ('r', 'x'))
        if colour_marker:
            ax.plot(track.meanlon[track_duration - 1], track.meanlat[track_duration - 1], ''.join(colour_marker))

        for i in [0, track_duration -1]:
            ts = pd.Timestamp(track.base_time[i].values)
            marker = f'{ts.day}.{ts.hour}'
            ax.annotate(marker, (track.meanlon[i].values, track.meanlat[i].values), fontsize='x-small')

        n_points = 20
        if display_area:
            legend_elements.append(
                Patch(facecolor='none', edgecolor='grey', label='CCS')
            )
            geoms = []
            for i in range(track_duration):
                lon = track.meanlon[i]
                lat = track.meanlat[i]
                radius = np.sqrt(track.ccs_area[i].values.item() / np.pi) * 1e3
                if lon < -170:
                    continue
                if np.isnan(radius):
                    continue
                circle_points = cartopy.geodesic.Geodesic().circle(lon=lon, lat=lat,
                                                                   radius=radius, n_samples=n_points,
                                                                   endpoint=False)
                geom = shapely.geometry.Polygon(circle_points)
                geoms.append(geom)
            full_geom = shapely.ops.unary_union(geoms)
            ax.add_geometries((full_geom,), crs=cartopy.crs.PlateCarree(),
                              facecolor='none', edgecolor='grey', linewidth=2)
        if display_pf_area:
            legend_elements.append(
                Patch(facecolor='none', edgecolor='b', label='PF')
            )
            geoms = []
            for i in range(track_duration):
                for j in range(3):
                    lon = track.pf_lon[i, j]
                    lat = track.pf_lat[i, j]
                    radius = np.sqrt(track.pf_area[i, j].values.item() / np.pi) * 1e3
                    if np.isnan(radius):
                        continue
                    circle_points = cartopy.geodesic.Geodesic().circle(lon=lon, lat=lat,
                                                                       radius=radius,
                                                                       n_samples=n_points,
                                                                       endpoint=False)
                    geom = shapely.geometry.Polygon(circle_points)
                    geoms.append(geom)
            full_geom = shapely.ops.unary_union(geoms)
            ax.add_geometries((full_geom,), crs=cartopy.crs.PlateCarree(),
                              facecolor='none', edgecolor='blue', linewidth=1)

        if display_trackresult:
            # legend_elements.extend([
            #     Line2D([0], [0], marker=cm[1], color='none', label=TRACK_RESULT[k],
            #            markerfacecolor=cm[0], markeredgecolor=cm[0])
            #     for k, cm in TRACK_RESULT_COLOUR_MARKER.items()
            #     if cm
            # ])
            for i in range(track_duration):
                track_status = int(track.track_status.values[i])
                colour_marker = ('b', f'{track_status}')
                if colour_marker:
                    ax.plot(track.meanlon[i], track.meanlat[i], ''.join(colour_marker))
        if legend_elements:
            ax.legend(handles=legend_elements)

    ax.coastlines()
    plt.show()


if __name__ == '__main__':
    plt.ion()

    stats_dir = Path('/home/markmuetz/Datasets/MCS_PRIME/MCS_database/MCS_Global/stats')
    stats_year_path = stats_dir / 'mcs_tracks_final_extc_20000601.0000_20010101.0000.nc'
    dspf = xr.open_dataset(stats_year_path)
    round_times_to_nearest_second(dspf)

    tracking_dir = Path('/home/markmuetz/Datasets/MCS_PRIME/MCS_database/MCS_Global/mcstracking')
    year = 2000
    tracking_year_dir = tracking_dir / f'{year}0601.0000_{year + 1}0101.0000'
    track_pixel_paths = sorted(tracking_year_dir.glob('*.nc'))
    date_path_map = {dt.datetime.strptime(p.stem, 'mcstrack_%Y%m%d_%H%M'): p
                     for p in track_pixel_paths}
    # Get pairs of frames which are separated by one hour (one timestep).
    ts = pd.DatetimeIndex(date_path_map.keys())
    tds = (ts[1:] - ts[:-1])
    pairs_mask = (tds.days == 0) & (tds.seconds == 3600)
    pairs = ts[1:-1][pairs_mask[:-1] | pairs_mask[1:]].values.reshape(-1, 2)

    for i in range(len(pairs)):
        for j in range(2):
            dspf_at_time = dspf.isel(tracks=(dspf.base_time.values == pairs[i, j]).any(axis=1))
            dspixel = xr.open_dataset(date_path_map[pd.Timestamp(pairs[i, j]).to_pydatetime()])
            plot_tracks_at_time(dspf_at_time, dspixel, pairs[i, j])
        plt.pause(0.1)
        if input('q to exit: ') == 'q':
            raise Exception()

