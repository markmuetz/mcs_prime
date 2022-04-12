# coding: utf-8
import cartopy
import cartopy.geodesic
import matplotlib.pyplot as plt
import numpy as np
import shapely
import xarray as xr

n_points = 100

# ds = xr.open_dataset('/home/markmuetz/Datasets/MCS_PRIME/MCS_database/Feng2020JGR_data/data/nam/robust_mcs_tracks_extc_20140101_20141231.nc')
dspf = xr.open_mfdataset('/home/markmuetz/Datasets/MCS_PRIME/MCS_database/Feng2020JGR_data/data/nam/robust_mcs_tracks_extc_*.nc', concat_dim='tracks', combine='nested', mask_and_scale=False)
dspf['tracks'] = np.arange(0, dspf.dims['tracks'], 1, dtype=int)

fig, ax = plt.subplots(subplot_kw=dict(projection=cartopy.crs.PlateCarree()))
dspf.tracks.load()
dspf.meanlon.load()
dspf.meanlat.load()

if False:
    # Plots PFs.
    for track_id in range(len(dspf.tracks))[:10]:
        print(track_id)
        track = dspf.sel(tracks=track_id)
        length = int(track.length.values.item())
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

if True:
    # Plots MCSs.
    print(len(dspf.tracks))
    for track_id in range(len(dspf.tracks)):
        print(track_id)
        track = dspf.sel(tracks=track_id)
        length = int(track.length.values.item())
        # for i in range(length):
        #     lon = track.meanlon[i]
        #     lat = track.meanlat[i]
        #     radius = np.sqrt(track.ccs_area[i].values.item() / np.pi) * 1e3
        #     if lon < -170:
        #         continue
        #     if np.isnan(radius):
        #         continue
        #     circle_points = cartopy.geodesic.Geodesic().circle(lon=lon, lat=lat, radius=radius, n_samples=n_points, endpoint=False)
        #     geom = shapely.geometry.Polygon(circle_points)
        #     ax.add_geometries((geom,), crs=cartopy.crs.PlateCarree(), facecolor='red', edgecolor='none', linewidth=0)
        ax.plot(track.meanlon[:length], track.meanlat[:length])


ax.coastlines()
plt.show()

