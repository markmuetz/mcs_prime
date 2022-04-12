# coding: utf-8
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import xarray as xr

BASEDIR = '/home/markmuetz/Datasets/MCS_PRIME/MCS_database/Feng2020JGR_data/data'

def plot_heatmap_for_reg_wrong_calc(ax, dspf):
    # This calculates the wrong thing. Suppose you have a stationary MCS, this will
    # count its contribution multiple times in the same place - not what you want.
    # Need to calc on per track basis.
    meanlon = dspf.meanlon.values.flatten()
    meanlon = meanlon[~np.isnan(meanlon)]
    meanlat = dspf.meanlat.values.flatten()
    meanlat = meanlat[~np.isnan(meanlat)]

    xmin = np.floor(meanlon.min())
    xmax = np.ceil(meanlon.max())
    ymin = np.floor(meanlat.min())
    ymax = np.floor(meanlat.max())
    bx = np.arange(xmin, xmax + 0.5, 0.5)
    by = np.arange(ymin, ymax + 0.5, 0.5)
    bxmid = (bx[1:] + bx[:-1]) / 2
    bymid = (by[1:] + by[:-1]) / 2

    hist, _, _ = np.histogram2d(meanlon, meanlat, bins=(bx, by))
    cmap = plt.get_cmap('Spectral_r').copy()
    cmap.set_over('magenta')
    cmap.set_under('white')
    levels = np.array([2, 3, 5, 8, 10, 15, 20, 25, 30, 35, 40, 50, 60, 80, 100]) / 4
    norm = colors.BoundaryNorm(boundaries=levels, ncolors=256)

    im = ax.contourf(bxmid, bymid, hist.T / 6,
                     cmap=cmap, extend='both',
                     levels=levels)
    return im


def update_heatmap_for_reg(hist, dspf, bx, by):
    dspf.tracks.load()
    dspf.meanlon.load()
    dspf.meanlat.load()
    for track_id in range(len(dspf.tracks)):
        print(track_id)
        track = dspf.sel(tracks=track_id)
        length = int(track.length.values.item())
        track_hist, _, _ = np.histogram2d(track.meanlon[:length], track.meanlat[:length], bins=(bx, by))
        hist += track_hist.T.astype(bool)



if __name__ == '__main__':
    dspfs = {}
    for reg in ['nam', 'spac', 'apac']:
        dspfs[reg] = xr.open_mfdataset(f'{BASEDIR}/{reg}/*.nc', concat_dim='tracks', combine='nested', mask_and_scale=False)
        dspfs[reg]['tracks'] = np.arange(0, dspfs[reg].dims['tracks'], 1, dtype=int)

    bx = np.arange(-180, 181, 1)
    by = np.arange(-60, 61, 1)
    bxmid = (bx[1:] + bx[:-1]) / 2
    bymid = (by[1:] + by[:-1]) / 2
    hist = np.zeros((120, 360))
    for reg in ['nam', 'spac', 'apac']:
        update_heatmap_for_reg(hist, dspfs[reg], bx, by)

    fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.PlateCarree()))
    ax.coastlines()

    cmap = plt.get_cmap('Spectral_r').copy()
    cmap.set_over('magenta')
    cmap.set_under('white')
    levels = (np.array([2, 3, 5, 8, 10, 15, 20, 25, 30, 35, 40, 50, 60, 80, 90]) / 100 * hist.max())
    norm = colors.BoundaryNorm(boundaries=levels, ncolors=256)

    im = ax.contourf(bxmid, bymid, hist,
                     cmap=cmap, extend='both',
                     levels=levels)
    plt.colorbar(im, orientation='horizontal')
    plt.show()

