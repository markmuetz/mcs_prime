# coding: utf-8
from mcs_prime import PATHS
import xarray as xr
paths = sorted(PATHS['pixeldir'].glob('*/mcstrack_2019060*.nc'))
paths = sorted(PATHS['pixeldir'].glob('*/mcstrack_20190601*.nc'))
len(paths)
dspixel = xr.open_mfdataset(paths)
dspixel.precipitation.load()
dspixel.precipitation.
dspixel.precipitation.mean()
dspixel.precipitation.mean(dim='time')
dspixel.cloudnumber
dspixel.cloudnumber.mean(dim='time')
dspixel.cloudnumber.mean(dim='time').compute()
dspixel.cloudnumber.mean(dim='time').load()
dspixel.cloudnumber.load()
dspixel.cloudnumber.max()
dspixel.cloudnumber.min()()
dspixel.cloudnumber.min()
import numpy as np
~np.isnan(dspixel.cloudnumber)
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.PlateCarree()))
mcs_precip = dspixel.precipitation * ~np.isnan(dspixel.cloudnumber)
ax.contourf(dspixel.lon, dspixel.lat, mcs_precip)
ax.contourf(dspixel.lon, dspixel.lat, mcs_precip.mean(dim='time'))
plt.show()
mcs_precip.mean(dim='time') / dspixel.precipitation.mean(dim='time')
(mcs_precip.mean(dim='time') / dspixel.precipitation.mean(dim='time')).max()
fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.PlateCarree()))
ax.contourf(dspixel.lon, dspixel.lat, mcs_precip.mean(dim='time') / dspixel.precipitation.mean(dim='time'))
ax.coastlines()
plt.show()
fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.PlateCarree()))
ax.contourf(dspixel.lon, dspixel.lat, mcs_precip.mean(dim='time') / dspixel.precipitation.mean(dim='time'))
plt.colorbar(_)
ax.coastlines()
plt.show()
