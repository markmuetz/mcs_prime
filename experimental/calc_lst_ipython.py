# coding: utf-8
get_ipython().run_line_magic('run', 'plot_land_sea_tracks.py')
dt.date
dt.time
dt.time(2000, 1, 1)
dt.date(2000, 1, 1)
dt.date(2000, 1, 1) + dt.time(12, 0)
dt.datetime.combine(dt.date(2000, 1, 1), dt.time(12, 0))
tracks.dstracks.base_time
import pandas as pd
pd.DatetimeIndex(tracks.dstracks.base_time)
base_time = pd.DatetimeIndex(tracks.dstracks.base_time)
base_time.time
base_time = pd.DatetimeIndex(tracks.dstracks.base_time.flatten())
base_time = pd.DatetimeIndex(tracks.dstracks.base_time.values.flatten())
base_time
base_time.time
meanlat = tracks.dstracks.meanlat.values.flatten()
meanlon = tracks.dstracks.meanlon.values.flatten()
meanlon
np.cos(meanlon * np.pi / 180) 
import numpy as np
np.cos(meanlon * np.pi / 180)
np.cos(meanlon * np.pi / 180) * 12
get_ipython().run_line_magic('pinfo', 'pd.Timedelta')
get_ipython().run_line_magic('pinfo', 'pd.TimedeltaIndex')
lst_offset_hours = np.cos(meanlon * np.pi / 180) * 12
get_ipython().run_line_magic('pinfo', 'pd.TimedeltaIndex')
pd.TimedeltaIndex(lst_offset_hours, 'h')
base_time.time
base_time.time + pd.TimedeltaIndex(lst_offset_hours, 'h')
base_time + pd.TimedeltaIndex(lst_offset_hours, 'h')
(base_time + pd.TimedeltaIndex(lst_offset_hours, 'h')).time
lst = (base_time + pd.TimedeltaIndex(lst_offset_hours, 'h')).time
lst
len(lst)
np.zeros(len(lst))
np.zeros(len(lst)).astype(dt.time)
h0 = dt.time(0, 0)
h1 = dt.time(1, 0)
h0arr = np.array([h0] * len(lst))
h1arr = np.array([h1] * len(lst))
(lst > h0) & (lst <= h1)
np.isnan(lst)
lst
lst.astype(float)
lst
lst
lst[-1]
type(lst[-1])
lst.astype(bool)
lst.astype(int)
pd.TimedeltaIndex(lst)
pd.TimedeltaIndex(dt.time(0) - lst)
lst
type(lst[-1])
type(lst[0])
pd._libs.tslibs.nattype
pd._libs.tslibs.nattype.NaT
lst == pd._libs.tslibs.nattype.NaT
lst[-1]
lst[-1] == pd._libs.tslibs.nattype.NaT
isinstance(lst[-1], pd._libs.tslibs.nattype.NaT)
isinstance(lst[-1], pd._libs.tslibs.nattype.NaTType)
isinstance(lst, pd._libs.tslibs.nattype.NaTType)
lst.dtype
lst[0].dtype
base_time
base_time.hour
base_time.hour + base_time.minute / 60
(base_time.hour + base_time.minute / 60).values
time_hours = (base_time.hour + base_time.minute / 60).values
lst = time_hours + lst_offset_hours
lst
lst % 24
lst % 24
lst = lst % 24
lst
(lst > 0) & (lst < 1)
lst_mask = ~np.isnan(lst)
(lst[lst_mask] > 0) & (lst[lst_mask] < 1)
((lst[lst_mask] > 0) & (lst[lst_mask] < 1)).sum()
((lst[lst_mask] > 0) & (lst[lst_mask] <= 1)).sum()
for i in range(24):
    print(((lst[lst_mask] > i) & (lst[lst_mask] <= i + 1)).sum())
    
base_time
np.isnan(base_time)
(lst > 0) & (lst < 1)
