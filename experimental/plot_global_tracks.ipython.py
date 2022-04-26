# coding: utf-8
get_ipython().run_line_magic('run', 'tracks.py')
import xarray as xr

dspf = xr.open_dataset(stats_year_path)
stats_year_path = stats_dir / 'mcs_tracks_final_extc_20000601.0000_20010101.0000.nc'
stats_dir = Path('/home/markmuetz/Datasets/MCS_PRIME/MCS_database/MCS_Global/stats')
from pathlib import Path

stats_dir = Path('/home/markmuetz/Datasets/MCS_PRIME/MCS_database/MCS_Global/stats')
stats_year_path = stats_dir / 'mcs_tracks_final_extc_20000601.0000_20010101.0000.nc'
dspf = xr.open_dataset(stats_year_path)
tracking_dir = Path('/home/markmuetz/Datasets/MCS_PRIME/MCS_database/MCS_Global/mcstracking')
tracking_year_dir = tracking_dir / f'{year}0601.0000_{year + 1}0101.0000'
year = 2000
tracking_year_dir = tracking_dir / f'{year}0601.0000_{year + 1}0101.0000'
track_pixel_paths = sorted(tracking_year_dir.glob('*.nc'))
import datetime as dt

date_path_map = {dt.datetime.strptime(p.stem, 'mcstrack_%Y%m%d_%H%M'): p for p in track_pixel_paths}
ts = pd.DatetimeIndex(date_path_map.keys())
ts
pairs_mask = (tds.days == 0) & (tds.seconds == 3600)
tds = ts[1:] - ts[:-1]
pairs_mask = (tds.days == 0) & (tds.seconds == 3600)
pairs = ts[1:-1][pairs_mask[:-1] | pairs_mask[1:]].values.reshape(-1, 2)
pairs
pairs[0]
pairs[0, 0]
dspf.base_time.values == pairs[0, 0]
(dspf.base_time.values == pairs[0, 0]).sum()
dspf.values
dspf.base_time.values
round_times_to_nearest_second(dspf)
(dspf.base_time.values == pairs[0, 0]).sum()
(dspf.base_time.values == pairs[0, 1]).sum()
(dspf.base_time.values == pairs[1, 0]).sum()
(dspf.base_time.values == pairs[0, 0]).any(axis=1)
dspf[(dspf.base_time.values == pairs[0, 0]).any(axis=1)]
dspf.tracks[(dspf.base_time.values == pairs[0, 0]).any(axis=1)]
dspf.isel(tracks=(dspf.base_time.values == pairs[0, 0]).any(axis=1))
dspf_at_time = dspf.isel(tracks=(dspf.base_time.values == pairs[0, 0]).any(axis=1))
plot_tracks(dspf_at_time, display_trackresult=False, display_area=False, display_pf_area=False)
plot_tracks(dspf_at_time, display_trackresult=False, display_area=True, display_pf_area=False)
plot_tracks(dspf_at_time, display_trackresult=False, display_area=True, display_pf_area=True)
pairs[0, 0]
pd.DateTime(pairs[0, 0])
pd.Datetime(pairs[0, 0])
get_ipython().run_line_magic('pinfo', 'pd.Timestamp')
pd.Timestamp(pairs[0, 0])
pd.Timestamp(pairs[0, 0]).to_pydatetime()
date_path_map[pd.Timestamp(pairs[0, 0]).to_pydatetime()]
dspixel = xr.open_dataset(date_path_map[pd.Timestamp(pairs[0, 0]).to_pydatetime()])
fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.PlateCarree()))
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.PlateCarree()))
ax.contourf(dspixel.lat, dspixel.lon, dspixel.cloudnumber[0])
ax.contourf(dspixel.lon, dspixel.lat, dspixel.cloudnumber[0])
plt.show()
plt.ion()
fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.PlateCarree()))
ax.contourf(dspixel.lon, dspixel.lat, dspixel.cloudnumber[0])
ax.coastlines()
plot_tracks(dspf_at_time, display_trackresult=False, display_area=True, display_pf_area=True)
get_ipython().run_line_magic('run', 'tracks.py')
fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.PlateCarree()))
ax.contourf(dspixel.lon, dspixel.lat, dspixel.cloudnumber[0])
plot_tracks(
    dspf_at_time,
    ax=ax,
    display_trackresult=False,
    display_area=True,
    display_pf_area=True,
)
get_ipython().run_line_magic('run', 'tracks.py')
plt.clf()
fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.PlateCarree()))
ax.contourf(dspixel.lon, dspixel.lat, dspixel.cloudnumber[0])
plot_tracks(
    dspf_at_time,
    ax=ax,
    display_trackresult=False,
    display_area=True,
    display_pf_area=True,
)
fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
dspf_at_time.isel(0)
dspf_at_time.isel(tracks=0)
track = dspf_at_time.isel(tracks=0)
track.base_time
track.base_time.values == pairs[0, 0]
np.where(track.base_time.values == pairs[0, 0])
np.where(track.base_time.values == pairs[0, 0])[0]
np.where(track.base_time.values == pairs[0, 0])[0].item
np.where(track.base_time.values == pairs[0, 0])[0].item()
get_ipython().run_line_magic('run', 'tracks.py')
plot_time(dspf, dspixel, ax=ax)
plot_time(dspf, dspixel, pairs[0, 0], ax=ax)
get_ipython().run_line_magic('debug', '')
plot_time(dspf_at_time, dspixel, pairs[0, 0], ax=ax)
get_ipython().run_line_magic('run', 'tracks.py')
plot_time(dspf_at_time, dspixel, pairs[0, 0], ax=ax)
plt.show()
fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.PlateCarree()))
plot_time(dspf_at_time, dspixel, pairs[0, 0])
dspixel
plot_time(dspf_at_time, dspixel, pairs[0, 0])
dspixel = xr.open_dataset(date_path_map[pd.Timestamp(pairs[0, 1]).to_pydatetime()])
plot_time(dspf_at_time, dspixel, pairs[0, 1])
dspf_at_time = dspf.isel(tracks=(dspf.base_time.values == pairs[0, 1]).any(axis=1))
plot_time(dspf_at_time, dspixel, pairs[0, 1])
get_ipython().run_line_magic('run', 'tracks.py')
track = dspf_at_time.isel(tracks=0)
time_index = np.where(track.base_time.values == time)[0].item()
time_index = np.where(track.base_time.values == pairs[0, 1])[0].item()
time_index
cloudnumber = track.cloudnumber[time_index]
cloudnumber
cloudnumber = track.cloudnumber[time_index].item
cloudnumber
cloudnumber = track.cloudnumber[time_index].values.item
cloudnumber
cloudnumber = track.cloudnumber[time_index].values.item()
cloudnumber
cloudnumber = int(track.cloudnumber[time_index].values.item())
plot_tracks_at_time(dspf_at_time, dspixel, pairs[0, 1], cloudnumber=cloudnumber)
dspixel
get_ipython().run_line_magic('run', 'tracks.py')
plot_tracks_at_time(dspf_at_time, dspixel, pairs[0, 1], cloudnumber=cloudnumber)
