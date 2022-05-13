# coding: utf-8
import xarray as xr
e5u = xr.open_dataarray('ecmwf-era5_oper_an_ml_202012012300.u.nc')
e5u = xr.open_dataarray('ecmwf-era5_oper_an_ml_201912012300.u.nc')
e5u
get_ipython().run_line_magic('ls', '')
e5lnsp = xr.open_dataarray('ecmwf-era5_oper_an_ml_201912012300.lnsp.nc')
e5lnsp
e5z = xr.open_dataarray('ecmwf-era5_oper_an_ml_201912012300.z.nc')
e5z
xr.open_dataarray('ecmwf-era5_oper_an_ml_201912012300.o3.nc')
xr.open_dataarray('ecmwf-era5_oper_an_ml_201912012300.q.nc')
xr.open_dataarray('ecmwf-era5_oper_an_ml_201912012300.t.nc')
get_ipython().run_line_magic('ls', '')
xr.open_dataarray('ecmwf-era5_oper_an_ml_201912012300.vo.nc')
e5u
e5u.sel(latitude=60, longitude=58.2)
e5u.sel(latitude=60, longitude=58.2, method='nearest')
e5u.isel(level=[-20, -10, 0]).sel(latitude=60, longitude=58.2, method='nearest')
e5u.isel(level=[-20, -10, 0]).sel(latitude=[60, 61], longitude=[58.2, 123], method='nearest')
e5u.isel(level=[-20, -10, 0]).sel(latitude=[60, 61], longitude=[58.2, 123], method='nearest').values
e5u.isel(level=[-20, -10, 0]).sel(latitude=[60, 61], longitude=[58.2, 123], method='nearest').values.shape
e5u.isel(level=-1).sel(latitude=[60, 61], longitude=[58.2, 123], method='nearest').values.shape
lat = xr.DataArray([60, 61], dims='index')
lon = xr.DataArray([58.2, 123], dims='index')
e5u.isel(level=-1).sel(latitude=lat, longitude=lon, method='nearest').values.shape
e5u.isel(level=[-21, -11, -1]).sel(latitude=lat, longitude=lon, method='nearest').values.shape
e5u.isel(level=[-21, -11, -1]).sel(latitude=lat, longitude=lon, method='nearest')
e5u.isel(time=0).isel(level=[-21, -11, -1]).sel(latitude=lat, longitude=lon, method='nearest')
get_ipython().run_line_magic('ls', '')
