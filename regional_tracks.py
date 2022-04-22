from collections import Counter
import datetime as dt

import cartopy
import cartopy.geodesic
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import shapely
import shapely.ops

# From dspf.trackresult.attrs['values'].split(';')
# ['-1 = default (not MCS)',
#  ' 0 = cp from previous file dissipated',
#  ' 1 = cp track from previous file continues',
#  ' 2 = cp is the larger fragment of a merger',
#  ' 3 = cp is the larger fragment of a split',
#  ' 10 = start of a new track',
#  ' 12 = cp is the smaller fragment of a merger',
#  ' 13 = cp is the smaller fragment of a split']
TRACK_RESULT = {
    -1: 'default (not MCS)',
    0: 'cloud from previous file dissipated',
    1: 'cloud track from previous file continues',
    2: 'cloud is the larger fragment of a merger',
    3: 'cloud is the larger fragment of a split',
    10: 'start of a new track',
    12: 'cloud is the smaller fragment of a merger',
    13: 'cloud is the smaller fragment of a split',
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


def tracks_for_days(dspf, date, ndays=1):
    date = np.datetime64(date)
    endday = np.datetime64(date) + np.timedelta64(ndays, 'D')
    track_start_time = dspf.base_time.values[:, 0]
    return dspf.isel(tracks=(track_start_time > date) & (track_start_time < endday))


def display_track_stats(dspf):
    num_tracks = len(dspf.tracks)
    print('Tracks:', num_tracks)
    num_track_points = int(dspf.length.sum().compute().values.item())
    print('Track points:', num_track_points)

    print('=' * 10, 'trackresults', '=' * 10)
    trackresult_stats = Counter(dspf.trackresult.values.flatten())
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
        length = int(track.length.values.item())
        print(
            f'{track_index + 1}/{len(dspf.tracks)}', track.base_time.values[0], length
        )
        for i in range(length):
            date_cn.append((track.base_time.values[i], track.cloudnumber.values[i]))
    assert len(date_cn) == len(set(date_cn))


def plot_tracks(
    dspf, display_area=True, display_pf_area=True, display_trackresult=True
):
    fig, ax = plt.subplots(subplot_kw=dict(projection=cartopy.crs.PlateCarree()))
    display_track_stats(dspf)
    print()
    # This makes a huge speed difference.
    dspf.base_time.load()
    dspf.meanlon.load()
    dspf.meanlat.load()
    dspf.trackresult.load()
    dspf.ccs_area.load()
    dspf.pf_area.load()
    dspf.pf_lon.load()
    dspf.pf_lat.load()

    start_track = dspf.isel(tracks=0)
    end_track = dspf.isel(tracks=-1)
    # How are you meant to format datetime64s? Not like this...
    start_time = f'{start_track.base_time.values[0]}'[:16]
    end_time = f'{end_track.base_time.values[int(end_track.length.values.item()) - 1]}'[
        :16
    ]
    title = f'{len(dspf.tracks)} tracks: {start_time} - {end_time}'
    ax.set_title(title)

    for track_index, track_id in enumerate(dspf.tracks):
        track = dspf.sel(tracks=track_id)
        length = int(track.length.values.item())
        print(
            f'{track_index + 1}/{len(dspf.tracks)}', track.base_time.values[0], length
        )

        legend_elements = [Line2D([0], [0], color='g', label='track')]

        ax.plot(track.meanlon[:length], track.meanlat[:length], 'g-')
        colour_marker = TRACK_RESULT_COLOUR_MARKER[track.starttrackresult.values.item()]
        if colour_marker:
            ax.plot(track.meanlon[0], track.meanlat[0], ''.join(colour_marker))

        colour_marker = TRACK_RESULT_COLOUR_MARKER[track.endtrackresult.values.item()]
        if colour_marker:
            ax.plot(
                track.meanlon[length - 1],
                track.meanlat[length - 1],
                ''.join(colour_marker),
            )

        for i in [0, length - 1]:
            ts = pd.Timestamp(track.base_time[i].values)
            marker = f'{ts.day}.{ts.hour}'
            ax.annotate(
                marker,
                (track.meanlon[i].values, track.meanlat[i].values),
                fontsize='x-small',
            )

        n_points = 20
        if display_area:
            legend_elements.append(
                Patch(facecolor='none', edgecolor='grey', label='CCS')
            )
            geoms = []
            for i in range(length):
                lon = track.meanlon[i]
                lat = track.meanlat[i]
                radius = np.sqrt(track.ccs_area[i].values.item() / np.pi) * 1e3
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
            legend_elements.append(Patch(facecolor='none', edgecolor='b', label='PF'))
            geoms = []
            for i in range(length):
                for j in range(3):
                    lon = track.pf_lon[i, j]
                    lat = track.pf_lat[i, j]
                    radius = np.sqrt(track.pf_area[i, j].values.item() / np.pi) * 1e3
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

        if display_trackresult:
            legend_elements.extend(
                [
                    Line2D(
                        [0],
                        [0],
                        marker=cm[1],
                        color='none',
                        label=TRACK_RESULT[k],
                        markerfacecolor=cm[0],
                        markeredgecolor=cm[0],
                    )
                    for k, cm in TRACK_RESULT_COLOUR_MARKER.items()
                    if cm
                ]
            )
            for i in range(length):
                trackresult = int(track.trackresult.values[i])
                colour_marker = TRACK_RESULT_COLOUR_MARKER[trackresult]
                if colour_marker:
                    ax.plot(track.meanlon[i], track.meanlat[i], ''.join(colour_marker))
        if legend_elements:
            ax.legend(handles=legend_elements)

    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
    ax.coastlines()
    plt.show()
