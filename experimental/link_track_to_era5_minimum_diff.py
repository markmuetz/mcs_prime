#!/usr/bin/env python
# coding: utf-8

# # Link tracks to ERA5 by finding the minimum in the difference between track vel, ERA5 wind at different levels
# 
# Only choose lowest 100 levels.

# In[1]:


import datetime as dt
from itertools import chain
from pathlib import Path
from timeit import default_timer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import xarray as xr

from mcs_prime import PATHS, McsTracks, PixelData
from mcs_prime.util import round_times_to_nearest_second


# In[2]:


e5datadir = PATHS['era5dir'] / 'data/oper/an_ml/2019/06/01'
stats_year_path = PATHS['statsdir'] / 'mcs_tracks_final_extc_20190101.0000_20200101.0000.nc'
dstracks = xr.open_dataset(stats_year_path)
round_times_to_nearest_second(dstracks)


# In[3]:


dstracks.base_time.load()
dstracks.meanlon.load()
dstracks.meanlat.load()
dstracks.movement_distance_x.load()
dstracks.movement_distance_y.load()

h = 12
track_time = dt.datetime(2019, 6, 1, h, 30)  # track data every hour on the half hour.
e5time = dt.datetime(2019, 6, 1, h, 0)  # ERA5 data every hour on the hour.
# Cannot interp track data - get ERA5 before and after and interp using e.g. ...mean(dim=time).

paths = [e5datadir / f'ecmwf-era5_oper_an_ml_{t.year}{t.month:02d}{t.day:02d}{t.hour:02d}00.{var}.nc'
         for var in ['u', 'v']
         for t in [e5time, e5time + dt.timedelta(hours=1)]]
# Only want levels 38-137 (lowest 100 levels), and lat limited to that of tracks, and midpoint time value.
e5uv = xr.open_mfdataset(paths).sel(latitude=slice(60, -60)).sel(level=slice(38, None)).mean(dim='time').load()

e5u = e5uv.u
e5v = e5uv.v


# In[4]:


track_point_mask = dstracks.base_time == pd.Timestamp(track_time)
track_point_lon = dstracks.meanlon.values[track_point_mask]
track_point_lat = dstracks.meanlat.values[track_point_mask]

track_point_vel_x = dstracks.movement_distance_x.values[track_point_mask] / 3.6 # km/h -> m/s.
track_point_vel_y = dstracks.movement_distance_y.values[track_point_mask] / 3.6

# Filter out NaNs.
nanmask = ~np.isnan(track_point_vel_x)
track_point_lon = track_point_lon[nanmask]
track_point_lat = track_point_lat[nanmask]
track_point_vel_x = track_point_vel_x[nanmask]
track_point_vel_y = track_point_vel_y[nanmask]


# In[5]:


# This is the same way you select values along a transect.
# But here I just want at unconnected points.
lon = xr.DataArray(track_point_lon, dims='track_point')
lat = xr.DataArray(track_point_lat, dims='track_point')

# N.B. no interp., mean over time does interpolation around half hour.
track_point_era5_u = e5u.sel(longitude=lon, latitude=lat, method='nearest').values
track_point_era5_v = e5v.sel(longitude=lon, latitude=lat, method='nearest').values

e5u.close()
e5v.close()
e5uv.close()


# In[6]:


track_point_era5_u


# In[7]:


# Can now calculate squared diff with judicious use of array broadcasting.
sqdiff = ((track_point_era5_u - track_point_vel_x[None, :])**2 + 
          (track_point_era5_v - track_point_vel_y[None, :])**2)


# In[8]:


np.isnan(sqdiff).any()


# In[9]:


idx = np.argmin(sqdiff, axis=0)
idx


# What does the above show? It is the level index of the minimum squared difference between the track point velocity and ERA5 winds. The index starts at level 38 (i.e. 0 index == level 38).

# In[10]:


index = np.arange(len(track_point_lon))

ds = xr.Dataset(data_vars=dict(
    level=('index', e5u.level.values[idx]),
    min_sq_diff=('index', np.min(sqdiff, axis=0))
    ),
    coords=dict(index=index),
)


# In[11]:


ds


# In[ ]:




