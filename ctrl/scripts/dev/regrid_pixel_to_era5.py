# coding: utf-8
import datetime as dt
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe

from load_2019 import tracks
from mcs_prime import PATHS

figdir = PATHS['figdir'] / 'tmp/regrid_pixel_to_era5'
figdir.mkdir(exist_ok=True, parents=True)

time = dt.datetime(2019, 1, 5, 0, 30)
pdtime = pd.Timestamp(time)

ts = tracks.tracks_at_time(time)
frame = ts.pixel_data.get_frame(time)
tmask = (ts.dstracks.base_time == pdtime).values
cns = ts.dstracks.cloudnumber.values[tmask]
cns.sort()

year = 2019
month = 1
day = 5
e5datadir = Path(f'/badc/ecmwf-era5/data/oper/an_ml/{year}/{month:02d}/{day:02d}')
e5u = xr.open_dataarray(e5datadir / f'ecmwf-era5_oper_an_ml_{year}{month:02d}{day:02d}0000.u.nc')
e5u = e5u.sel(latitude=slice(60, -60)).sel(level=137).isel(time=0)
pixel_precip = frame.dspixel.precipitation.isel(time=0)

regridder_path = figdir / 'bilinear_1200x3600_481x1440_peri.nc'
if regridder_path.exists():
    print('loading regridder')
    regridder = xe.Regridder(pixel_precip, e5u, 'bilinear', periodic=True,
                             reuse_weights=True, weights=regridder_path)
else:
    # N.B. this is reasonably challenging.
    # * lat direction is different,
    # * ERA5 lon: 0--360, pixel lon: -180--180.
    print('generating regridder')
    regridder = xe.Regridder(pixel_precip, e5u, 'bilinear', periodic=True)
    regridder.to_netcdf(regridder_path)
print(regridder)

mask_path = figdir / 'cloudnumber_ERA5_grid.nc'
if mask_path.exists():
    print('loading da_mask')
    da_mask = xr.open_dataarray(mask_path)
else:
    print('generating da_mask')
    mask_regridded = []
    for i in cns:
        print(i)
        mask_regridded.append(regridder((frame.dspixel.cloudnumber == i).astype(float)))

    da_mask_regridded = xr.concat(mask_regridded, pd.Index(cns, name='cn'))
    da_mask = ((da_mask_regridded > 0.5).astype(int) * da_mask_regridded.cn).sum(dim='cn')
    da_mask.to_netcdf(mask_path)


# (da_mask >= 1).plot()

# fig, (ax0, ax1) = plt.subplots(2, 1, subplot_kw=dict(projection=ccrs.PlateCarree()),
#                                sharex=True, sharey=True)
#
native_pixel_path = figdir / 'cloudnumber_native_grid.nc'
frame.dspixel.cloudnumber.isin(cns).to_netcdf(native_pixel_path)
# (da_mask >= 1)[::-1].plot(ax=ax1)

