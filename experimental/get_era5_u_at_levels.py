from pathlib import Path
import xarray as xr

datadir = Path('/badc/ecmwf-era5/data/oper/an_ml/2019/12/01')

e5u = xr.open_dataarray(datadir / 'ecmwf-era5_oper_an_ml_201912012300.u.nc')

# This is the same way you select values along a transect. But here I just want at unconnected points.
lon = xr.DataArray([58.2, 123], dims='index')
lat = xr.DataArray([60, 61], dims='index')
# Choose 3 levels near the bottom of domain.
print(e5u.isel(time=0).isel(level=[-21, -11, -1]).sel(longitude=lon, latitude=lat, method='nearest'))

