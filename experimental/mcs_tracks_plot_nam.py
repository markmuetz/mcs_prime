# coding: utf-8
import cartopy
import cartopy.geodesic
import matplotlib.pyplot as plt
import numpy as np
import shapely
import xarray as xr

n_points = 100

ds = xr.open_dataset('/home/markmuetz/Datasets/MCS_PRIME/MCS_database/Feng2020JGR_data/data/nam/robust_mcs_tracks_extc_20140101_20141231.nc')
fig, ax = plt.subplots(subplot_kw=dict(projection=cartopy.crs.PlateCarree()))

for track_id in range(len(ds.tracks))[:10]:
    print(track_id)
    track = ds.sel(tracks=track_id)
    length = int(track.length.data.item())
    for i in range(length):
        for j in range(3):
            lon = track.pf_lon[i, j]
            lat = track.pf_lat[i, j]
            radius = np.sqrt(track.pf_area[i, j].values.item() / np.pi) * 1e3
            if np.isnan(radius):
                continue
            print('  ', i, radius)
            circle_points = cartopy.geodesic.Geodesic().circle(lon=lon, lat=lat, radius=radius, n_samples=n_points, endpoint=False)
            geom = shapely.geometry.Polygon(circle_points)
            ax.add_geometries((geom,), crs=cartopy.crs.PlateCarree(), facecolor='red', edgecolor='none', linewidth=0)
    # ax.plot(track.pf_lon[:length, 0], track.pf_lat[:length, 0])

ax.coastlines()
plt.show()

