# coding: utf-8
import datetime as dt

import cartopy.crs as ccrs
import matplotlib.pyplot as plt

from mcs_prime import PATHS
from mcs_prime import McsTracks


tracks = McsTracks.load(PATHS['statsdir'] / 'mcs_tracks_final_extc_20190101.0000_20200101.0000.nc', PATHS['pixeldir'])
tracks_at_time = tracks.tracks_at_time(dt.datetime(2019, 6, 21, 0, 30))
land_tracks, sea_tracks, both_tracks = tracks_at_time.land_sea_both_tracks()

fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.PlateCarree()))
ax.coastlines()

land_tracks.plot(ax=ax, colour='g', linestyle='-')
sea_tracks.plot(ax=ax, colour='b', linestyle='-')
both_tracks.plot(ax=ax, colour='r', linestyle='-')
plt.show()
