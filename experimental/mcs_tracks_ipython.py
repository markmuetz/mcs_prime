# coding: utf-8
import xarray as xr

xr.open_dataset('robust_mcs_tracks_extc_20140101_20141231.nc')
ds = xr.open_dataset('robust_mcs_tracks_extc_20140101_20141231.nc')
ds.times
ds.tracks
ds
ds.variables
ds.variables
ds.variables[0]
ds.variables.keys()
list(ds.variables.keys())
ds
ds.coords
ds.coords.tracks
ds.tracks
ds.sel(track=0)
ds.sel(tracks=0)
ds.sel(tracks=1)
ds.sel(tracks=2)
ds.sel(tracks=3)
ds.sel(tracks=4)
ds.sel(tracks=4).base_time
ds.sel(tracks=4).length
list(ds.variables.keys())
ds.sel(tracks=4).pf_lon
ds.sel(tracks=4).pf_lon.shape
ds.sel(tracks=4).pf_lon[:, 0]
for track_id in range(len(ds.tracks)):
    track = ds.sel(tracks=track_id)
    plt.plot(track.pf_lon[: track.length, 0], track.pf_lat[: track.length, 0])

import matplotlib.pyplot as plt

for track_id in range(len(ds.tracks)):
    track = ds.sel(tracks=track_id)
    plt.plot(track.pf_lon[: track.length, 0], track.pf_lat[: track.length, 0])

track
track.length
track.length.data
track.length.data[0]
track.length.data.value
track.length.data.item
track.length.data.item()
for track_id in range(len(ds.tracks)):
    track = ds.sel(tracks=track_id)
    length = int(track.length.data.item())
    plt.plot(track.pf_lon[:length, 0], track.pf_lat[:length, 0])

plt.show()
import cartopy.crs as ccrs

fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.PlateCarree()))
for track_id in range(len(ds.tracks)):
    print(track_id)
    track = ds.sel(tracks=track_id)
    length = int(track.length.data.item())
    ax.plot(track.pf_lon[:length, 0], track.pf_lat[:length, 0])

ax.coastlines()
plt.show()
list(ds.variables.keys())
track.tracks
tracks.movement_storm_x
track.movement_storm_x
track = ds.sel(tracks=22)
track.movement_storm_x
ds.movement_storm_x
ds.movement_storm_x.attributes
ds.movement_storm_x.attrs
print(ds.movement_storm_x)
ds.variables.keys()
ds.meridional_propagation_speed
list(ds.variables.keys())
ds.uspeed
ds.uspeed.max()
ds.uspeed.min()
ds.vspeed.min()
ds.vspeed.max()
ds.mergecloudnumber
3065 * 200 * 100
ds.mergecloudnumber.values
ds.splitcloudnumber.values
