# coding: utf-8
from mcs_prime import PATHS
dstracks = xr.open_dataset(PATHS['tracksdir'] / 'mcs_tracks_final_extc_20190101.0000_20200101.0000.nc')
import xarray as xr
dstracks = xr.open_dataset(PATHS['tracksdir'] / 'mcs_tracks_final_extc_20190101.0000_20200101.0000.nc')
dstracks = xr.open_dataset(PATHS['statsdir'] / 'mcs_tracks_final_extc_20190101.0000_20200101.0000.nc')
dstracks
dstracks = xr.open_dataset(PATHS['statsdir'] / 'mcs_tracks_final_extc_*.0000.nc')
dstracks = xr.open_mfdataset(PATHS['statsdir'] / 'mcs_tracks_final_extc_*.0000.nc')
PATHS['statsdir'].glob('*.nc')
sorted(PATHS['statsdir'].glob('*.nc'))
dstracks = xr.open_mfdataset(sorted(PATHS['statsdir'].glob('mcs_tracks_final_extc_*.0000.nc')))
dstracks
dstracks.area
dstracks.base_time
dstracks.base_time.values[:, 0]
dstracks.tracks = np.arange(len(dstracks.tracks))
import numpy as np
dstracks.tracks = np.arange(len(dstracks.tracks))
dstracks['tracks'] = np.arange(len(dstracks.tracks))
dstracks.tracks
dstracks.area.max(axis=1)
dstracks.area.max(axis=1).values
max_areas = dstracks.area.max(axis=1).values
max_areas
get_ipython().run_line_magic('pinfo', 'np.percentile')
np.percentile(max_areas, [25, 50, 75])
dstracks.track_duration
dstracks.track_duration.values
durations = dstracks.track_duration.values
import matplotlib.pyplot as plt
get_ipython().run_line_magic('pinfo', 'plt.hist2d')
get_ipython().run_line_magic('pinfo', 'plt.hist2d')
plt.hist2d
get_ipython().run_line_magic('pinfo', 'plt.hist2d')
plt.hist2d(durations, max_areas, (30, 30))
plt.show()
plt.hist2d(durations, max_areas, (30, 30))
plt.show()
plt.hist2d(durations, max_areas, (np.linspace(0, 50, 30), np.linspace(0, 0.5e6, 30)))
plt.show()
plt.hist2d(durations, max_areas, (np.linspace(0, 50, 50), np.linspace(0, 0.5e6, 30)))
plt.show()
dstracks.pf_rainrate
dstracks.pf_rainrate.max(axis=(1, 2))
max_rainrates = dstracks.pf_rainrate.max(axis=(1, 2)).values
plt.hist2d(durations, max_rainrates, (np.linspace(0, 50, 50), 50))
plt.show()
plt.hist2d(max_areas, max_rainrates, (np.linspace(0, 0.5e6, 50), 50))
plt.show()
dstracks.tb
dstracks.bt
dstracks
dstracks.variables
[v for v in dstracks.variables]
dstracks.corecold_mintb
dstracks.corecold_mintb.min(axis=1)
min_tbs = dstracks.corecold_mintb.min(axis=1)
plt.hist2d(min_tbs, max_rainrates, (50, 50))
plt.show()
import cartopy
cartopy.feature.LAND
land = cartopy.feature.LAND
land
land.intersecting_geometries
land.intersecting_geometries()
land.geometries
land.geometries()
geoms = list(land.geometries())
geoms
from shapely.geometry import Point
import shapely
get_ipython().run_line_magic('pinfo', 'shapely.ops.unary_union')
shapely.ops.unary_union(geoms)
land_geom = shapely.ops.unary_union(geoms)
Point(-1.3, 51.7)
Point(-1.3, 51.7).intersect(land_geom)
Point(-1.3, 51.7).intersects(land_geom)
Point(-10, 49).intersects(land_geom)
lons = np.linspace(-180, 180, 360)
lats = np.linspace(-90, 90, 180)
L = np.zeros((360, 180))
for i, lon in lons, 
for i, lon in enumerate(lons):
    for j, lat in enumerate(lats):
        L[i, j] = Point(lon, lat).intersects(land_geom)
        
for i, lon in enumerate(lons):
    print(i, lon)
    for j, lat in enumerate(lats):
        L[i, j] = Point(lon, lat).intersects(land_geom)
        
L
L.shape
plt.contourf(lons, lats, L)
plt.contourf(lons, lats, L.T)
plt.show()
360 * 180
dstracks.track_duration.sum()
dstracks.track_duration.sum().values
dsL = xr.Dataset(L, dims={'lat': lats, 'lon': lons})
dsL = xr.Dataset(L, coords={'lat': lats, 'lon': lons})
dsL = xr.Dataset(L, dim={'lat': lats, 'lon': lons})
dsL = xr.DataArray(data_vars=L, coords={'lat': lats, 'lon': lons})
dsL = xr.DataArray(L, coords={'lat': lats, 'lon': lons})
dsL = xr.DataArray(L, coords={'lon': lons, 'lat': lats})
dsL
dsL.sel(lon=-65, lat=22, method='nearest')
dsL.sel(lon=[-65, -64], lat=[22, 23], method='nearest')
track = dstracks.isel(321)
track = dstracks.isel(dim=321)
track = dstracks.isel(tracks=3212)
track
dsL.interp(lon=track.meanlon, lat=meanlat)
dsL.interp(lon=track.meanlon, lat=track.meanlat)
from mcs_prime import McsTracks
tracks = McsTracks.load(PATHS['statsdir'] / 'mcs_tracks_final_extc_20190101.0000_20200101.0000.nc',
                        PATHS['pixeldir'])
tracks.get_track(15244)
track = tracks.get_track(15244)
dsL.interp(lon=track.meanlon, lat=track.meanlat)
dsL.interp(lon=track.dstrack.meanlon[:track.duration], lat=track.dstrack.meanlat[:track.duration])
track.plot()
plt.show()
tracks_at_time = tracks.tracks_at_time(dt.datetime(2019, 6, 21, 0, 30))
import datetime as dt
tracks_at_time = tracks.tracks_at_time(dt.datetime(2019, 6, 21, 0, 30))
tracks_at_time
tracks_at_time.plot()
plt.show()
track = tracks.get_track(15416)
dsL.interp(lon=track.dstrack.meanlon[:track.duration], lat=track.dstrack.meanlat[:track.duration])
tracks_at_time.plot()
plt.show()
tracks
dstracks = xr.open_mfdataset(sorted(PATHS['statsdir'].glob('mcs_tracks_final_extc_*.0000.nc')))
dstracks
tracks.dstracks
dstracks
sorted(PATHS['statsdir'].glob('mcs_tracks_final_extc_*.0000.nc'))
dstracks = xr.open_mfdataset(sorted(PATHS['statsdir'].glob('mcs_tracks_final_extc_*.0000.nc')))
dstracks = xr.open_mfdataset(sorted(PATHS['statsdir'].glob('mcs_tracks_final_extc_*.0000.nc')), concat_dim='tracks', combine='nested', mask_and_scale=False)
dstracks
dstracks['tracks'] = np.arange(len(dstracks.tracks))
dstracks.base_time.values[:, 0]
tracks
