# coding: utf-8
from pathlib import Path

stats_dir = Path('/home/markmuetz/Datasets/MCS_PRIME/MCS_database/MCS_Global/stats')
tracking_dir = Path('/home/markmuetz/Datasets/MCS_PRIME/MCS_database/MCS_Global/mcstracking')
year = 2000
tracking_year_dir = tracking_dir / f'{year}0601.0000_{year + 1}0101.0000'
tracking_year_dir
tracking_year_dir.exists()
track_pixel_paths = tracking_year_dir.glob('*.nc')
track_pixel_paths
track_pixel_paths = sorted(tracking_year_dir.glob('*.nc'))
track_pixel_paths
[p.stem for p in track_pixel_paths]
import datetime as dt

get_ipython().run_line_magic('pinfo', 'dt.datetime.strptime')
dt.datetime.strptime('mcstrack_20001229_2230', 'mcstrack_%Y%m%d_%H%M')
date_path_map = {dt.datetime.strptime(p.stem, 'mcstrack_%Y%m%d_%H%M'): p for p in track_pixel_paths}
date_path_map
date_path_map.keys()
list(date_path_map.keys())
get_ipython().run_line_magic('ls', '')
stats_year_path = stats_dir / 'mcs_tracks_final_extc_20000601.0000_20010101.0000.nc'
stats_year_path
stats_year_path.exists()
stats = xr.open_dataset(stats_year_path)
import x
import xarray as xr

stats = xr.open_dataset(stats_year_path)
stats
import pandas as pd

ts = pd.DatetimeIndex(date_path_map.keys())
ts
ts[1:] - ts[:-1]
(ts[1:] - ts[:-1])
(ts[1:] - ts[:-1]).hour
(ts[1:] - ts[:-1]).hours
(ts[1:] - ts[:-1]).minute
tds = ts[1:] - ts[:-1]
tds.seconds
tds.seconds == 3600
ts[:-1][tds.seconds == 3600]
ts[:-1][tds.seconds == 3600].date
ts
ts[:-1][(tds.days == 0) & (tds.seconds == 3600)]
ts
ts[:40]
(tds.days == 0) & (tds.seconds == 3600)
ts[4:6]
ts[3:7]
ts[:-1][(tds.days == 0) & (tds.seconds == 3600)]
ts[:40]
stats
ts
ts[:40]
ts[20:30]
ts[23:30]
ts[23:25]
pair = ts[23:25]
pair[0]
type(pair[0])
stats.base_time
stats.base_time == pair[0]
(stats.base_time == pair[0]).sum()
pair
pair.values
pair.values[0]
stats.base_time == pair.values[0]
(stats.base_time == pair.values[0]).sum()
(stats.base_time == pair.values[1]).sum()
(tds.days == 0) & (tds.seconds == 3600)
pairs = (tds.days == 0) & (tds.seconds == 3600)
ts[1:-1][pairs[:-1] | pairs[1:]]
ts[1:-1][pairs[:-1] | pairs[1:]].values
pair_values = ts[1:-1][pairs[:-1] | pairs[1:]].values
pair_values
pair_values.reshape(-1, 2)
pairs_reshaped = pair_values.reshape(-1, 2)
pairs_reshaped
(stats.base_time == pairs_reshaped[0, 0]).sum().item()
(stats.base_time == pairs_reshaped[1, 0]).sum().item()
(stats.base_time == pairs_reshaped[1, 1]).sum().item()
(stats.base_time == pairs_reshaped[2, 0]).sum().item()
(stats.base_time == pairs_reshaped[3, 0]).sum().item()
(stats.base_time == pairs_reshaped[3, 0]).count().item()
(stats.base_time == pairs_reshaped[3, 0]).count()
(stats.base_time == pairs_reshaped[3, 0]).sum()
stats
stats.base_time.attrs
stats.base_time.shape
get_ipython().run_line_magic('run', 'tracks.py')
plot_tracks(tracks_for_days(stats, '2000-06-20', 1), display_trackresult=True)
stats
get_ipython().run_line_magic('run', 'tracks.py')
plot_tracks(tracks_for_days(stats, '2000-06-20', 1), display_trackresult=True)
plot_tracks(tracks_for_days(stats, '2000-06-20', 1), display_trackresult=False)
get_ipython().run_line_magic('run', 'tracks.py')
plot_tracks(tracks_for_days(stats, '2000-06-20', 1), display_trackresult=True)
get_ipython().run_line_magic('run', 'tracks.py')
plot_tracks(tracks_for_days(stats, '2000-06-20', 1), display_trackresult=True)
get_ipython().run_line_magic('run', 'tracks.py')
plot_tracks(tracks_for_days(stats, '2000-06-20', 1), display_trackresult=True)
plot_tracks(tracks_for_days(stats, '2000-06-20', 1), display_trackresult=False)
get_ipython().run_line_magic('run', 'tracks.py')
plot_tracks(tracks_for_days(stats, '2000-06-20', 1), display_trackresult=False)
plot_tracks(
    tracks_for_days(stats, '2000-06-20', 1),
    display_trackresult=False,
    display_area=False,
    display_pfarea=False,
)
plot_tracks(
    tracks_for_days(stats, '2000-06-20', 1),
    display_trackresult=False,
    display_area=False,
    display_pf_area=False,
)
plot_tracks(
    tracks_for_days(stats, '2000-06-19', 3),
    display_trackresult=False,
    display_area=False,
    display_pf_area=False,
)
plot_tracks(
    tracks_for_days(stats, '2000-06-19', 2),
    display_trackresult=False,
    display_area=False,
    display_pf_area=False,
)
plt.ion()
plot_tracks(
    tracks_for_days(stats, '2000-06-19', 2),
    display_trackresult=False,
    display_area=False,
    display_pf_area=False,
)
ts
ts.month
ts.month == 6
(ts.month == 6) & (ts.day in [19, 20])
(ts.month == 6) & ([19, 20] in ts.day)
(ts.month == 6) & (ts.day.is_in([19, 20]))
(ts.month == 6) & (ts.day.isin([19, 20]))
ts[(ts.month == 6) & (ts.day.isin([19, 20]))]
ts[(ts.month == 6) & (ts.day.isin([19, 20]))].values
ts[(ts.month == 6) & (ts.day.isin([19, 20]))].values[0]
(stats.base_time == ts[(ts.month == 6) & (ts.day.isin([19, 20]))].values[0]).sum()
stats.base_time
s = tracks_for_days('2000-06-19', 2)
s = tracks_for_days(stats, '2000-06-19', 2)
s
stats.basetime
stats.base_time
stats.base_time.sel
get_ipython().run_line_magic('pinfo', 'stats.base_time.sel')
stats.base_time.sel(pair.values[0])
stats.sel(base_time=pair.values[0])
stats.base_time
s.base_time
s.base_time - pairs[0]
s.base_time.values - pairs[0]
pairs[0]
s.base_time.values - pair[0]
pair
s.base_time.values - pair.values[0]
(s.base_time.values - pair.values[0]) < 1
(s.base_time.values - pair.values[0]).astype(int) < 1
np.abs((s.base_time.values - pair.values[0]).astype(int)) < 1e6
(s.base_time.values - pair.values[0]).astype(int)
np.abs((s.base_time.values - pair.values[0]).astype(int))
np.abs((s.base_time.values - pair.values[0]).astype(float))
np.abs((s.base_time.values - pair.values[0]).astype(float)) < 1e6
(np.abs((s.base_time.values - pair.values[0]).astype(float)) < 1e6).sum()
(np.abs((s.base_time.values - pair.values[1]).astype(float)) < 1e6).sum()
stats.base_time
stats.base_time[0]
stats.base_time[0, 0]
stats.base_time[0, 0].values
v = stats.base_time[0, 0].values
v.astype
v.astype(dt.datetime)
v
v.astype(int)
v.astype(int) % 1000000000
99 % 100
np.datetime64(959819400000000000)
get_ipython().run_line_magic('pinfo', 'np.datetime64')
np.datetime64(959819400000000000, 'us')
np.datetime64(959819400000000000, 'ns')
int(365 / 10)
int(364 / 10)
int(360 / 10)
int(359 / 10)
rount(359 / 10)
round(359 / 10)
type(round(359 / 10))
get_ipython().run_line_magic('pinfo', 'rount')
get_ipython().run_line_magic('pinfo', 'round')
np.datetime64(round(959819400000000000 / 1e9), 's')
v
np.datetime64(round(v.astype(long) / 1e9), 's')
np.datetime64(round(v.astype(int) / 1e9), 's')
stats.base_time[0, 0]
stats.base_time[0, 0] = np.datetime64(round(v.astype(int) / 1e9), 's')
stats.base_time[0, 0]
stats.base_time.values
stats.base_time.values.astype(int)
stats.base_time = np.datetime64(round(stats.base_time.values.astype(int) / 1e9), 's')
np.round
np.round(21)
stats.base_time = np.datetime64(np.round(stats.base_time.values.astype(int) / 1e9), 's')
np.round(stats.base_time.values.astype(int) / 1e9)
np.round(stats.base_time.values.astype(int) / int(1e9))
np.round(stats.base_time.values.astype(int) / 1000000000)
np.round(stats.base_time.values.astype(int) / 1000000000).astype(int)
np.datetime64(np.round(stats.base_time.values.astype(int) / 1e9), 's')
np.datetime64(np.round(stats.base_time.values.astype(int) / 1e9).astype(int), 's')
get_ipython().run_line_magic('pinfo', 'np.round')
np.round(1 / 21)
np.around(1 / 21)
get_ipython().run_line_magic('pinfo', 'np.around')
np.datetime64(np.round(stats.base_time.values.astype(int) / 1e9).astype(np.int64), 's')
np.round(stats.base_time.values.astype(int) / 1e9).astype(np.int64)
np.round(stats.base_time.values.astype(np.int64) / 1e9).astype(np.int64)
v
stats.base_time = np.datetime64(round(stats.base_time.values.astype(int) / 1e9), 's')
round(stats.base_time.values.astype(int) / 1e9)
round(stats.base_time.values[0, 0].astype(int) / 1e9)
np.round(stats.base_time.values[0, 0].astype(int) / 1e9)
np.round(stats.base_time.values[0, 0].astype(int) / 1e9).astype(int)
np.datetime64(np.round(stats.base_time.values[0, 0].astype(int) / 1e9).astype(int), 's')
np.datetime64(round(stats.base_time.values[0, 0].astype(int) / 1e9).astype(int), 's')
np.datetime64(round(stats.base_time.values[0, 0].astype(int) / 1e9), 's')
np.round(stats.base_time.values[0, 0].astype(int) / 1e9).astype(int)
type(np.round(stats.base_time.values[0, 0].astype(int) / 1e9).astype(int))
type(round(stats.base_time.values[0, 0].astype(int) / 1e9).astype(int))
type(round(stats.base_time.values[0, 0].astype(int) / 1e9))
type(np.round(stats.base_time.values[0, 0].astype(int) / 1e9).astype(int))
type(np.round(stats.base_time.values[0, :3].astype(int) / 1e9).astype(int))
np.round(stats.base_time.values[0, :3].astype(int) / 1e9).astype(int)
np.round(stats.base_time.values[0, :3].astype(int) / 1e9).astype(int)[0]
type(np.round(stats.base_time.values[0, :3].astype(int) / 1e9).astype(int)[0])
np.round(stats.base_time.values[0, :3].astype(int) / 1e9).astype(int)
np.datetime64(np.round(stats.base_time.values[0, :3].astype(int) / 1e9).astype(int), 's')
np.datetime64(np.round(stats.base_time.values[0, :3].astype(int) / 1e9).astype(int)[0], 's')
get_ipython().run_line_magic('pinfo', 'np.int_')
np.int_(1)
type(np.int_(1))
type(np.int(1))
np.datetime64(np.round(stats.base_time.values[0, :3].astype(int) / 1e9).astype(str)[0], 's')
np.datetime64(
    np.round(stats.base_time.values[0, :3].astype(int) / 1e9).astype(int).astype(str)[0],
    's',
)
np.int64
np.int64(v)
np.datetime64(np.int64(v))
np.datetime64(int(np.int64(v)))
np.datetime64(int(np.int64(v)), 'ns')
np.datetime64(np.int64(v), 'ns')
np.datetime64
get_ipython().run_line_magic('pinfo', 'np.datetime64')
v
v.tolist()
type(v.tolist())
pair
pair.values
pair.values.tolist()
paris
pairs_reshaped
pairs_reshaped.to_list()
pairs_reshaped.tolist()
p = stats.base_time[:2]
p
p = stats.base_time[0, :2].values
p
p = stats.base_time[1, :2].values
p
p.astype(int)
p.astype(int).dtype
np.datetime64(p.astype(int), 'ns')
np.datetime64(map(p.astype(int), int), 'ns')
p.astype(int)
p.astype(int).apply
get_ipython().run_line_magic('pinfo', 'np.apply_over_axes')
np.apply_over_axes(p.astype(int), int)
np.apply
get_ipython().run_line_magic('pinfo', 'np.apply_along_axis')
np.vectorize(int)
vec_int = np.vectorize(int)
p
vec_int(p.astype(int))
np.datetime64(vec_int(p.astype(int)))
vec_int(p.astype(int))
vec_int(p.astype(int)).dtype


def c(val):
    return np.datetime64(int(val))


vec_c = np.vectorize(c)
vec_c(p.astype(int)).dtype


def c(val):
    return np.datetime64(int(val), 'ns')


vec_c(p.astype(int)).dtype
vec_c = np.vectorize(c)
vec_c(p.astype(int)).dtype
vec_c(p.astype(int))


def c(val):
    return np.datetime64(int(round(val / 1e9)), 's')


vec_c = np.vectorize(c)
vec_c(p.astype(int))


def remove_time_incaccuracy(val):
    return np.datetime64(int(round(val / 1e9) * 1e9), 'ns')


vec_rm = np.vectorize(remove_time_incaccuracy)
vec_rm(p.astype(int))
vec_rm(stats.base_time.values)
vec_rm(stats.base_time.values.astype(int))
stats.base_time.values.astype(int)
stats.base_time.values
stats.base_time.values[0, -1]
np.isnan(stats.base_time.values[0, -1])
np.isnan(stats.base_time.values)
tmask = ~np.isnan(stats.base_time.values)
vec_rm(stats.base_time.values.astype(int)[tmask])
stats.base_time[tmask] = vec_rm(stats.base_time.values.astype(int)[tmask])
stats.base_time.values[tmask] = vec_rm(stats.base_time.values.astype(int)[tmask])
stats.base_time
