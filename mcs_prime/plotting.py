from pathlib import Path

import cartopy
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from .mcs_prime_config import status_dict


def round_away_from_zero_to_sigfig(val, nsf):
    sign = 1 if val >= 0 else -1
    factor = 10 ** (nsf - math.ceil(math.log10(sign * val)))
    newval = math.ceil(sign * val * factor) / factor
    return sign * newval


def ticks_from_min_max_lon_lat(minlon, maxlon, minlat, maxlat, dlon=5, dlat=5):
    tminlon = (minlon + dlon) - (minlon % dlon)
    tmaxlon = maxlon - (maxlon % dlon)
    tminlat = (minlat + dlat) - (minlat % dlat)
    tmaxlat = maxlat - (maxlat % dlat)
    return np.arange(tminlon, tmaxlon + dlon, dlon), np.arange(
        tminlat, tmaxlat + dlat, dlat
    )


def plot_diurnal_cycle(tracks, dhours=6, mode="mean"):
    sizes = {1: (4, 6), 2: (4, 3), 3: (4, 2), 4: (2, 3), 6: (2, 2), 24: (1, 1)}
    size = sizes[dhours]
    fig, axes = plt.subplots(
        size[0],
        size[1],
        subplot_kw=dict(projection=cartopy.crs.PlateCarree()),
        sharex=True,
        sharey=True,
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

    nan_mask = ~np.isnan(tracks.dstracks.meanlon.values)
    meanlon = tracks.dstracks.meanlon.values[nan_mask]
    meanlat = tracks.dstracks.meanlat.values[nan_mask]

    if mode in ["anomaly", "anomaly_frac"]:
        mean_hist, _, _ = np.histogram2d(meanlon, meanlat, bins=(bx, by), density=True)
        diff_hists = {}

    hists = {}

    base_time = pd.DatetimeIndex(tracks.dstracks.base_time.values[nan_mask])

    lst_offset_hours = meanlon / 180 * 12
    time_hours = (base_time.hour + base_time.minute / 60).values
    lst = time_hours + lst_offset_hours
    lst = lst % 24

    for i, ax in enumerate(axes.flatten()):
        hour = i * dhours
        ax.coastlines()
        ax.set_title(f"LST: {hour} - {hour + dhours}")

        lst_mask = (lst > hour) & (lst < hour + dhours)
        lon = meanlon[lst_mask]
        lat = meanlat[lst_mask]

        hist, _, _ = np.histogram2d(lon, lat, bins=(bx, by), density=True)
        hists[i] = hist
        if mode == "anomaly":
            diff_hists[i] = hist - mean_hist
        elif mode == "anomaly_frac":
            diff_hists[i] = hist / mean_hist

    if mode == "anomaly":
        diff_hist_min = min(
            [np.nanmin(diff_hists[i]) for i, ax in enumerate(axes.flatten())]
        )
        diff_hist_max = max(
            [np.nanmax(diff_hists[i]) for i, ax in enumerate(axes.flatten())]
        )

        abs_max = round_away_from_zero_to_sigfig(
            max(abs(diff_hist_min), abs(diff_hist_max)), 2
        )

        levels = (
            np.array([-1, -5e-1, -2e-1, -1e-1, -5e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1])
            * abs_max
        )
        cmap = plt.get_cmap("RdBu_r").copy()
    elif mode == "anomaly_frac":
        levels = np.array(
            [0, 1 / 2, 2 / 3, 4 / 5, 0.9, 0.95, 1.05, 1.1, 5 / 4, 3 / 2, 2, 10]
        )
        cmap = plt.get_cmap("RdBu_r").copy()
    else:
        cmap = plt.get_cmap("Spectral_r").copy()
        levels = (
            np.array([0, 2, 3, 5, 8, 10, 15, 20, 25, 30, 35, 40, 50, 60, 80, 90, 100])
            / 100
            * hist.max()
        )
    norm = colors.BoundaryNorm(boundaries=levels, ncolors=256)

    for i, ax in enumerate(axes.flatten()):
        if mode in ["anomaly", "anomaly_frac"]:
            hist = diff_hists[i]
        else:
            hist = hists[i]
        extent = (xmin, xmax, ymin, ymax)
        im = ax.imshow(hist.T, origin="lower", extent=extent, cmap=cmap, norm=norm)

    plt.colorbar(im, ax=axes, label=f"MCS density {mode}")


def animate_track(
    track,
    zoom="swath",
    savefigs=False,
    figdir=None,
):
    user_input = ""
    times = pd.DatetimeIndex(track.base_time)
    start = pd.Timestamp(track.dstrack.start_basetime.values).to_pydatetime()
    end = pd.Timestamp(track.dstrack.end_basetime.values).to_pydatetime()

    cloudnumbers = track.cloudnumber
    frames = track.pixel_data.get_frames(start, end)
    swath_extent = frames.get_min_max_lon_lat(cloudnumbers)
    dlon = swath_extent[1] - swath_extent[0]
    dlat = swath_extent[3] - swath_extent[2]
    aspect = (dlon + 3) / (dlat * 3)
    height = 9.5

    fig = plt.figure("track animation")
    fig.set_size_inches(height * aspect, height)
    fig.subplots_adjust(left=0.1, right=0.95, bottom=0.05, top=0.95, hspace=0.1)

    precip = frames.get_data("precipitation", cloudnumbers)
    tb = frames.dspixel.tb.values
    cn = frames.get_data("cloudnumber", cloudnumbers)

    cn_swath = cn.sum(axis=0).astype(bool)
    precip_swath = precip.sum(axis=0)

    legend_elements = [
        Patch(facecolor="none", edgecolor="black", label="Tb = 241K"),
        Patch(facecolor="none", edgecolor="red", label="Cloud mask"),
        Line2D([0], [0], color="g", label="track"),
        Patch(facecolor="none", edgecolor="grey", label="Track cloud (circle)"),
        Patch(facecolor="none", edgecolor="blue", label="Track PFx3 (circle)"),
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
    while 0 <= i < len(cloudnumbers):
        cloudnumber = cloudnumbers[i]
        time = times[i]
        print(time)
        fig.clf()
        ax1 = fig.add_subplot(3, 1, 1, projection=cartopy.crs.PlateCarree())
        ax2 = fig.add_subplot(3, 1, 2, projection=cartopy.crs.PlateCarree())
        ax3 = fig.add_subplot(3, 1, 3)

        status = int(track.track_status[i])
        ax1.set_title(f"Track {track.track_id} @ {time} ({status=})")
        ax2.set_title(f"Track {track.track_id} swath")
        ax1.coastlines()
        ax2.coastlines()
        track.plot(times=[time], ax=ax1, use_status_for_marker=True)
        track.plot(times=[time], ax=ax2, use_status_for_marker=True)

        _anim_individual_frame(
            ax1, frames, precip[i], cn[i], tb[i], frame_norm, frame_precip_levels
        )
        _anim_swath(
            ax2, frames, precip_swath, cn_swath, swath_norm, swath_precip_levels
        )
        _anim_timeseries(track, ax3, i)

        def fmt_lat_ticklabels(ticks):
            labels = []
            for t in ticks:
                if t < 0:
                    labels.append(f"{-t:.0f} °S")
                else:
                    labels.append(f"{t:.0f} °N")
            return labels

        def fmt_lon_ticklabels(ticks):
            labels = []
            for t in ticks:
                if t < 0:
                    labels.append(f"{-t:.0f} °W")
                else:
                    labels.append(f"{t:.0f} °E")
            return labels

        if zoom == "swath":
            extent = swath_extent
        elif zoom == "current":
            frames = track.pixel_data.get_frames(time, time)
            extent = frames.get_min_max_lon_lat([cloudnumber])
        else:
            raise Exception("zoom must be: swath or current")
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
            figpath = Path(
                figdir
                / f"anim_track/track_{track.track_id}/track_{track.track_id}_{i:03d}.png"
            )
            figpath.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(figpath)
            figpaths.append(figpath)
        else:
            plt.pause(1)
            print(status, status_dict[status])
            if user_input != "c":
                user_input = input(
                    "c for continue, q to quit, p for prev, <enter> for step: "
                )
                if user_input == "q":
                    raise Exception("quit")
                elif user_input == "p":
                    i -= 1
                elif user_input == "c":
                    print("<ctrl-c> to stop")
                    i += 1
                else:
                    i += 1
            else:
                i += 1
    return figpaths


def _anim_individual_frame(
    ax, frames, precip_frame, cn_frame, tb_frame, frame_norm, frame_precip_levels
):
    im1 = ax.contourf(
        frames.dspixel.lon,
        frames.dspixel.lat,
        precip_frame,
        cmap="Blues",
        norm=frame_norm,
        levels=frame_precip_levels,
    )
    ax.contour(
        frames.dspixel.lon,
        frames.dspixel.lat,
        cn_frame,
        levels=[0.5],
        colors="red",
    )
    ax.contour(
        frames.dspixel.lon,
        frames.dspixel.lat,
        tb_frame,
        levels=[241],
        colors="black",
    )
    plt.colorbar(im1, ax=ax, label="masked precip. (mm hr$^{-1}$)")


def _anim_swath(ax, frames, precip_swath, cn_swath, swath_norm, swath_precip_levels):
    im2 = ax.contourf(
        frames.dspixel.lon,
        frames.dspixel.lat,
        precip_swath,
        cmap="Blues",
        norm=swath_norm,
        levels=swath_precip_levels,
    )
    ax.contour(
        frames.dspixel.lon,
        frames.dspixel.lat,
        cn_swath,
        levels=[0.5],
        colors="red",
    )

    cbar2 = plt.colorbar(im2, ax=ax, label="masked precip. swath (mm)")
    cbar2.set_ticks(swath_precip_levels)
    cbar2.set_ticklabels([f"{v:.2f}" for v in swath_precip_levels])


def _anim_timeseries(track, ax, i):
    for name, data in [
        ("area", track.area),
        ("PF area", np.nanmean(track.pf_area, axis=1)),
        ("PF rainrate", np.nanmean(track.pf_rainrate, axis=1)),
    ]:
        ax.plot(data / np.nanmax(data), label=f"{name} (max={np.nanmax(data):.1f})")
    ax.set_xlim((-0.5, track.duration - 0.5))
    ax.set_ylim((0, 1.6))
    ax.set_yticks([0, 0.5, 1])
    ax.axvline(x=i)
    ax.legend(loc="upper right")
    ax.set_xlabel("time since start (hr)")
