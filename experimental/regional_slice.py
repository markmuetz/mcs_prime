import datetime as dt

import xarray as xr

from mcs_prime import PATHS, McsTracks, PixelData

year = 2019
month = 6
day = 21
e5datadir = PATHS['era5dir'] / f'data/oper/an_sfc/{year}/{month:02d}/{day:02d}'

h = 6

e5time = dt.datetime(year, month, day, h, 0)
paths = [e5datadir / (f'ecmwf-era5_oper_an_sfc_{t.year}{t.month:02d}{t.day:02d}'
                      f'{t.hour:02d}00.{var}.nc')
         for var in ['cape', 'tcwv']
         for t in [e5time, e5time + dt.timedelta(hours=1)]]

# Similar to Li 2023 region but slightly larger.
lat_slice = slice(55, 0)
lon_slice = slice(95, 165)


e5data = (xr.open_mfdataset(paths).sel(latitude=lat_slice, longitude=lon_slice)
          .interp(time=e5time + dt.timedelta(minutes=30)).load())


e5data

outdir = PATHS['outdir'] / 'regional_ERA5_data' / f'{e5time.year}' / f'{e5time.month:02d}' / f'{e5time.day:02d}'
filename = f'ecmwf-era5_oper_an_sfc_{e5time.year}{e5time.month:02d}{e5time.day:02d}{e5time.hour:02d}30.east_asia.nc'
outpath = outdir / filename
outpath

outpath.parent.mkdir(exist_ok=True, parents=True)
e5data.to_netcdf(outpath)
