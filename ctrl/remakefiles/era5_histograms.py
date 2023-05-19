import calendar
import datetime as dt
from functools import partial
from itertools import product
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe


from remake import Remake, TaskRule
from remake.util import format_path as fmtp

from mcs_prime import PATHS, McsTracks, PixelData
from mcs_prime.era5_calc import ERA5Calc

# A couple of the jobs ran out of mem. A few of the GenERA5PixelData jobs
# take longer than 4hr, plus the short-serial-4hr queue can only have
# 400 active jobs on it, whereas short-serial can have 2000.
# Increase mem and runtime.
# slurm_config = {'queue': 'short-serial', 'mem': 64000, 'max_runtime': '20:00:00'}
slurm_config = {'account': 'short4hr', 'queue': 'short-serial-4hr', 'mem': 64000}
era5_histograms = Remake(config=dict(slurm=slurm_config, content_checks=False))

# years = list(range(2000, 2021))
years = [2020]
months = range(1, 13)

# For testing - 5 days.
DATES = pd.date_range(f'{years[0]}-01-01', f'{years[0]}-01-05')
# DATES = pd.date_range(f'{years[0]}-01-01', f'{years[-1]}-12-31')
DATE_KEYS = [(y, m, d) for y, m, d in zip(DATES.year, DATES.month, DATES.day)]


class GenRegridder(TaskRule):
    """Generate a xesmf regridder for regridding/coarsening from MCS dataset to ERA5 grid."""
    rule_inputs = {'cape': (PATHS['era5dir'] / 'data/oper/an_sfc/'
                            '2020/01/01'
                            '/ecmwf-era5_oper_an_sfc_'
                            '202001010000.cape.nc'),
                   'pixel': (PATHS['pixeldir'] /
                             '20200101.0000_20210101.0000/2020/01/01' /
                             'mcstrack_20200101_0030.nc')}
    rule_outputs = {'regridder': (PATHS['outdir'] / 'conditional_era5_histograms' /
                                  'regridder' /
                                  'bilinear_1200x3600_481x1440_peri.nc')}

    def rule_run(self):
        e5cape = xr.open_dataarray(self.inputs['cape'])
        dspixel = xr.open_dataset(self.inputs['pixel'])

        e5cape = e5cape.sel(latitude=slice(60, -60)).isel(time=0)
        pixel_precip = dspixel.precipitation.isel(time=0)
        # Note, quite a challenging regrid:
        # * lat direction is different,
        # * ERA5 lon: 0--360, pixel lon: -180--180.
        # xesmf seems to manage without any issue though.
        # Uses bilinear and periodic regridding.
        regridder = xe.Regridder(pixel_precip, e5cape, 'bilinear', periodic=True)
        regridder.to_netcdf(self.outputs['regridder'])


class GenERA5Shear(TaskRule):
    @staticmethod
    def rule_inputs(year, month, day):
        start = dt.datetime(year, month, day)
        # Note there are 25 of these.
        e5times = pd.date_range(start, start + dt.timedelta(hours=24), freq='H')
        e5inputs = {f'era5_{t}_{var}': (PATHS['era5dir'] /
                                        f'data/oper/an_ml/{t.year}/{t.month:02d}/{t.day:02d}' /
                                        (f'ecmwf-era5_oper_an_ml_{t.year}{t.month:02d}{t.day:02d}'
                                         f'{t.hour:02d}00.{var}.nc'))
                    for t in e5times
                    for var in ['u', 'v']}
        return e5inputs

    rule_outputs = {'shear': (PATHS['outdir'] / 'era5_processed' /
                              '{year}' /
                              'daily_shear_{year}_{month:02d}_{day:02d}.nc')}
    var_matrix = {
        ('year', 'month', 'day'): DATE_KEYS,
    }
    def rule_run(self):
        # https://confluence.ecmwf.int/display/UDOC/L137+model+level+definitions
        levels = [
            136,  # 1010hPa/30m
            111,  # 804hPa/1911m
            101,  # 610hPa/4074m
        ]
        start = dt.datetime(self.year, self.month, self.day)
        e5times = pd.date_range(start, start + dt.timedelta(hours=24), freq='H')
        e5inputs = {(t, v): self.inputs[f'era5_{t}_{v}']
                    for t in e5times
                    for v in ['u', 'v']}
        e5data = xr.open_mfdataset(e5inputs.values()).sel(latitude=slice(60, -60), level=levels).load()
        u = e5data.u.values
        v = e5data.v.values
        # Calc LLS (1010-804), Low-to-mid shear (804-610), MLS (1010-610) in one line.
        # LLS = shear[:, :, 0]
        # Low-to-Mid shear = shear[:, :, 1]
        # MLS = shear[:, :, 2]
        # axis 1 is the level axis.
        shear = np.sqrt((np.roll(u, -1, axis=1) - u)**2 + (np.roll(v, -1, axis=1) - v)**2)

        dsout = xr.Dataset(
            coords=dict(
                time=e5times,
                shear_level=[0, 1, 2],
                latitude=e5data.latitude,
                longitude=e5data.longitude,
            ),
            data_vars={
                'shear': (('time', 'shear_level', 'latitude', 'longitude'), shear),
            },
            attrs={
                'shear_level 0': 'LLS: shear between surf and 800 hPa (ERA5 136-111)',
                'shear_level 1': 'Low-to-mid: shear between 800 and 600 hPa (ERA5 111-101)',
                'shear_level 2': 'MLS: shear between surf and 600 hPa (ERA5 136-101)',
            }
        )
        print(dsout)
        dsout.to_netcdf(self.outputs['shear'])


def calc_mf_u(rho, q, u):
    """Calculates the x-component of density-weighted moisture flux on a c-grid"""
    def calc_mid(var):
        return (var + np.roll(var, -1, axis=2)) / 2
    return calc_mid(rho * q * u)


def calc_mf_v(rho, q, v):
    """Calculates the y-component of density-weighted moisture flux on a c-grid"""
    def calc_mid(var):
        s1 = (slice(None), slice(None, -1), slice(None))
        s2 = (slice(None), slice(1, None), slice(None))
        return (var[s1] + var[s2]) / 2
    return calc_mid(rho * q * v)


def calc_div_mf(rho, q, u, v, dx, dy):
    """Calculates the divergence of the moisture flux

    Switches back to original grid, but loses latitudinal extremes.
    Keeps longitudinal extremes due to biperiodic domain.
    """
    mf_u = calc_mf_u(rho, q, u)
    mf_v = calc_mf_v(rho, q, v)
    dqu_dx = (mf_u - np.roll(mf_u, 1, axis=2)) / dx[None, :, None]
    # Note, these indices are not the wrong way round!
    # latitude decreases with increasing index, hence I want the opposite
    # to what you would expect.
    dqv_dy = (mf_v[:, :-1, :] - mf_v[:, 1:, :] ) / dy

    return dqu_dx[:, 1:-1] + dqv_dy


class GenERA5VIMoistureFluxDiv(TaskRule):
    @staticmethod
    def rule_inputs(year, month, day):
        start = dt.datetime(year, month, day)
        # Note there are 225 these.
        e5times = pd.date_range(start, start + dt.timedelta(hours=24), freq='H')
        e5inputs = {f'era5_{t}_{var}': (PATHS['era5dir'] /
                                        f'data/oper/an_ml/{t.year}/{t.month:02d}/{t.day:02d}' /
                                        (f'ecmwf-era5_oper_an_ml_{t.year}{t.month:02d}{t.day:02d}'
                                         f'{t.hour:02d}00.{var}.nc'))
                    for var in ['u', 'v', 't', 'q', 'lnsp']
                    for t in e5times}
        e5inputs['model_levels'] = PATHS['datadir'] / 'ERA5/ERA5_L137_model_levels_table.csv'
        return e5inputs

    rule_outputs = {'vi_moisture_flux_div': (PATHS['outdir'] / 'era5_processed' /
                                             '{year}' /
                                             'daily_moisture_flux_div_{year}_{month:02d}_{day:02d}.nc')}
    var_matrix = {
        ('year', 'month', 'day'): DATE_KEYS,
    }

    depends_on = [ERA5Calc, calc_mf_u, calc_mf_v, calc_div_mf]

    def rule_run(self):
        start = dt.datetime(self.year, self.month, self.day)
        # Note there are 25 of these.
        e5times = pd.date_range(start, start + dt.timedelta(hours=24), freq='H')

        e5calc = ERA5Calc(self.inputs['model_levels'])

        vi_div_mf = []
        for i, time in enumerate(e5times):
            # Running out of memory doing full 4D calc: break into 24x 3D calc.
            print(time)
            e5inputs = {(time, v): self.inputs[f'era5_{time}_{v}']
                        for v in ['u', 'v', 't', 'q', 'lnsp']}
            # Only load levels where appreciable q, to save memory.
            # q -> 0 by approx. level 70. Go higher (level 60=~100hPa) to be safe.
            with xr.open_mfdataset(e5inputs.values()) as ds:
                e5data = ds.isel(time=0).sel(latitude=slice(60.25, -60.25), level=slice(60, 137)).load()
            print(e5data)
            u, v = e5data.u.values, e5data.v.values
            q = e5data.q.values
            T = e5data.t.values
            lnsp = e5data.lnsp.values

            nlev = 137 - 60 + 1
            p = e5calc.calc_pressure(lnsp)[-nlev:]
            Tv = e5calc.calc_Tv(T, q)
            print(p.mean(axis=(1, 2)))
            print('p', p.shape)
            print('Tv', Tv.shape)
            rho = e5calc.calc_rho(p, Tv)

            # Calc dx/dy.
            dx_deg = e5data.longitude.values[1] - e5data.longitude.values[0]
            dy_deg = e5data.latitude.values[0] - e5data.latitude.values[1]  # N.B. want positive so swap indices.
            Re = 6371e3  # Radius of Earth in m.

            dy = dy_deg / 360 * 2 * np.pi * Re  # km
            dx = np.cos(e5data.latitude.values * np.pi / 180) * dx_deg / 360 * 2 * np.pi * Re  # km

            div_mf = calc_div_mf(rho, q, u, v, dx, dy)
            # TODO: Should be pressure weighted I think.
            vi_div_mf.append(div_mf.sum(axis=0))

        dsout = xr.Dataset(
            coords=dict(
                time=e5times,
                latitude=e5data.latitude[1:-1],
                longitude=e5data.longitude,
            ),
            data_vars={
                'vertically_integrated_moisture_flux_div': (('time', 'latitude', 'longitude'),
                                                            np.array(vi_div_mf))
            },
        )
        print(dsout)
        dsout.to_netcdf(self.outputs['vi_moisture_flux_div'])


class GenERA5PixelData(TaskRule):
    # Disabled because it takes a long time just to check all the files.
    enabled = False
    """Generate ERA5 pixel data by regridding native MCS dataset masks/cloudnumbers

    The MCS dataset pixel-level data has a field cloudnumber, which maps onto the
    corresponding cloudnumber field in the tracks data. Here, for each time (on
    the half-hour, to match MCS dataset), I regrid this data to the ERA5 grid.
    It is done this way round because the pixel-level data is on the IMERG grid, which
    is finer than ERA5 (0.1deg vs 0.25deg).
    This is non-trivial, because they use different longitude endpoints. The regridding/
    coarsening is done by xesmf, which handled this and periodic longitude.

    The scheme for this is:
    * load pixel to ERA5 regridder.
    * for each hour for which there is data, do regridding.
    * load pixel-level cloudnumber/Tb.
    * for each cloudnumber, regrid the mask for that cloudnumber, keeping all values
      greater than 0.5.
    * combine into one integer field.
    * also regrid Tb field.
    * save compressed int16 results.
    """
    @staticmethod
    def rule_inputs(year, month, day):
        # Just need one to recreate regridder.
        e5inputs = {'cape': (PATHS['era5dir'] / 'data/oper/an_sfc/'
                             '2020/01/01'
                             '/ecmwf-era5_oper_an_sfc_'
                             '202001010000.cape.nc')}

        if year == 2000:
            start_date = '20000601'
        else:
            start_date = f'{year}0101'
        # Not all files exist! Do not include those that don't.
        pixel_paths = [(PATHS['pixeldir'] /
                        f'{start_date}.0000_{year + 1}0101.0000'
                        f'/{year}/{month:02d}/{day:02d}' /
                        f'mcstrack_{year}{month:02d}{day:02d}_{h:02d}30.nc')
                       for h in range(24)]
        pixel_inputs = {f'pixel_{h}': p
                        for h, p in zip(range(24), pixel_paths)
                        if p.exists()}
        inputs = {**e5inputs, **pixel_inputs}

        inputs['regridder'] = GenRegridder.rule_outputs['regridder']
        return inputs

    rule_outputs = {'e5pixel': (PATHS['outdir'] / 'era5_pixel' /
                                '{year}' / '{month:02d}' / '{day:02d}' /
                                'era5_MCS_pixel_{year}{month:02d}{day:02d}.nc')}

    var_matrix = {('year', 'month', 'day'): DATE_KEYS}

    def rule_run(self):
        e5cape = xr.open_dataarray(self.inputs['cape']).sel(latitude=slice(60, -60))
        # Note, this is a subset of times based on which times exist.
        hours = [h for h in range(24) if f'pixel_{h}' in self.inputs]
        pixel_inputs = {h: self.inputs[f'pixel_{h}']
                        for h in hours}

        times = [dt.datetime(self.year, self.month, self.day, h, 30) for h in hours]
        data = np.zeros((len(hours), len(e5cape.latitude), len(e5cape.longitude)))
        dsout = xr.Dataset(
            coords=dict(
                time=times,
                latitude=e5cape.latitude,
                longitude=e5cape.longitude,
            ),
            data_vars={
                'cloudnumber': (('time', 'latitude', 'longitude'), data.copy()),
                'tb': (('time', 'latitude', 'longitude'), data.copy()),
            }
        )
        regridder = None
        for i, h in enumerate(hours):
            # Load the pixel data.
            dspixel = xr.open_dataset(pixel_inputs[h])
            pixel_precip = dspixel.precipitation.isel(time=0).load()

            if regridder is None:
                # Reload the regridder.
                regridder = xe.Regridder(pixel_precip, e5cape, 'bilinear', periodic=True,
                                         reuse_weights=True, weights=self.inputs['regridder'])

            # Perform the regrid for each cloudnumber.
            mask_regridded = []
            max_cn = int(dspixel.cloudnumber.values[~np.isnan(dspixel.cloudnumber.values)].max())
            cns = range(1, max_cn + 1)
            # The idea here is to stack masks together, one for earch cn, then sum along
            # the stacked dim to get a cn field on ERA5 grid.
            for cn in cns:
                print(cn)
                mask_regridded.append(regridder((dspixel.cloudnumber == cn).astype(float)))

            da_mask_regridded = xr.concat(mask_regridded, pd.Index(cns, name='cn'))
            # Note, da_mask_regridded will have shape (len(cns), len(lat), len(lon))
            # and the `* da_mask_regridded.cn` will multiply each 1/0 bool mask by its
            # corresponding # cn, turning 1s into its cloudnumber. Summing along the cn index
            # turns this into a (len(lat), len(lon)) field.
            # Note, a regridded value belongs to a cn if it is over 0.5, so that masks should not
            # overlap.
            e5cn = ((da_mask_regridded > 0.5).astype(int) * da_mask_regridded.cn).sum(dim='cn')
            mcs_mask = e5cn.values[0] > 0.5

            e5tb = regridder(dspixel.tb)

            dsout.cloudnumber[i] = e5cn[0]
            dsout.tb[i] = e5tb[0]
            # N.B. I have checked that the max cloudnumber (1453) in the tracks dataset is < 2**16
            # assert cns.max() < 2**16 - 1

        # Save as compressed int16 field for lots of compression!
        comp = dict(dtype='int16', zlib=True, complevel=4)
        encoding = {var: comp for var in dsout.data_vars}
        dsout.to_netcdf(self.outputs['e5pixel'], encoding=encoding)


ERA5VARS = ['cape', 'tcwv']
ERA5_PROCESSED_VARS = ['shear']
LS_REGIONS = ['all', 'land', 'ocean']


class Era5MeanField(TaskRule):
    def rule_inputs(year, month):
        start = pd.Timestamp(year, month, 1)
        end = start + pd.DateOffset(months=1)
        e5times = pd.date_range(start, end, freq='H')
        proc_pixel_times = pd.date_range(start, end - pd.Timedelta(days=1), freq='D')

        e5inputs = {f'era5_{t}_{var}': (PATHS['era5dir'] /
                                        f'data/oper/an_sfc/{t.year}/{t.month:02d}/{t.day:02d}' /
                                        (f'ecmwf-era5_oper_an_sfc_{t.year}{t.month:02d}{t.day:02d}'
                                         f'{t.hour:02d}00.{var}.nc'))
                    for t in e5times
                    for var in ERA5VARS}
        e5proc_shear = {
            f'era5p_shear_{t}': fmtp(GenERA5Shear.rule_outputs['shear'],
                                     year=t.year,
                                     month=t.month,
                                     day=t.day)
            for t in proc_pixel_times
        }
        e5proc_vimfd = {
            f'era5p_vimfd_{t}': fmtp(GenERA5VIMoistureFluxDiv.rule_outputs['vi_moisture_flux_div'],
                                     year=t.year,
                                     month=t.month,
                                     day=t.day)
            for t in proc_pixel_times
        }
        inputs = {
            **e5inputs,
            **e5proc_shear,
            **e5proc_vimfd,
        }
        return inputs

    rule_outputs = {'meanfield': (PATHS['outdir'] / 'conditional_era5_histograms' /
                                  '{year}' /
                                  'era5_mean_field_{year}_{month:02d}.nc')}

    var_matrix = {
        'year': years,
        'month': months,
    }

    def load_data(self):
        start = pd.Timestamp(self.year, self.month, 1)
        end = start + pd.DateOffset(months=1)
        e5times = pd.date_range(start, end - pd.Timedelta(hours=1), freq='H')
        proc_pixel_times = pd.date_range(start, end - pd.Timedelta(days=1), freq='D')

        e5paths = [self.inputs[f'era5_{t}_{v}']
                   for t in e5times
                   for v in ERA5VARS]
        e5proc_shear_paths = [self.inputs[f'era5p_shear_{t}']
                              for t in proc_pixel_times]
        e5proc_vimfd_paths = [self.inputs[f'era5p_vimfd_{t}']
                              for t in proc_pixel_times]

        def open_25hr_data(paths):
            """These datasets have 25 hours of data in them - discard last hour for all.

            Originally this was so I could load one day and have all the info to interp to half-hourly
            times.
            """
            datasets = [xr.open_dataset(p) for p in paths]
            return xr.concat(
                [ds.isel(time=slice(24)) for ds in datasets],
                dim='time'
            )

        self.logger.debug('Open ERA5')
        e5ds = xr.open_mfdataset(e5paths).sel(latitude=slice(60, -60))
        self.logger.debug('Open proc shear')
        e5shear = open_25hr_data(e5proc_shear_paths)
        self.logger.debug('Open proc VIMFD')
        e5vimfd = open_25hr_data(e5proc_vimfd_paths)

        return xr.merge([e5ds.load(), e5shear.load(), e5vimfd.load()])

    def rule_run(self):
        self.logger.info('Load data')
        ds = self.load_data()
        dsout = ds.mean(dim='time').load()
        dsout = dsout.expand_dims({'time': 1})
        dsout = dsout.assign_coords({'time': [pd.Timestamp(ds.time.mean().item())]})
        dsout.attrs['ntimes'] = len(ds.time)
        print(dsout)

        comp = dict(zlib=True, complevel=4)
        encoding = {var: comp for var in dsout.data_vars}
        dsout.to_netcdf(self.outputs['meanfield'], encoding=encoding)


def conditional_inputs(year, month):
    start = pd.Timestamp(year, month, 1)
    end = start + pd.DateOffset(months=1)
    e5times = pd.date_range(start, end, freq='H')
    proc_pixel_times = pd.date_range(start, end - pd.Timedelta(days=1), freq='D')

    e5pixel_inputs = {
        f'e5pixel_{t}': fmtp(GenERA5PixelData.rule_outputs['e5pixel'],
                             year=t.year,
                             month=t.month,
                             day=t.day)
        for t in proc_pixel_times
    }
    if start.year == 2000:
        start_date = '20000601'
    else:
        start_date = f'{start.year}0101'
    e5pixel_inputs['tracks'] = (PATHS['statsdir'] /
                                f'mcs_tracks_final_extc_{start_date}.0000_{start.year + 1}0101.0000.nc')


    e5inputs = {f'era5_{t}_{var}': (PATHS['era5dir'] /
                                    f'data/oper/an_sfc/{t.year}/{t.month:02d}/{t.day:02d}' /
                                    (f'ecmwf-era5_oper_an_sfc_{t.year}{t.month:02d}{t.day:02d}'
                                     f'{t.hour:02d}00.{var}.nc'))
                for t in e5times
                for var in ERA5VARS}
    e5proc_shear = {
        f'era5p_shear_{t}': fmtp(GenERA5Shear.rule_outputs['shear'],
                                 year=t.year,
                                 month=t.month,
                                 day=t.day)
        for t in proc_pixel_times
    }
    e5proc_vimfd = {
        f'era5p_vimfd_{t}': fmtp(GenERA5VIMoistureFluxDiv.rule_outputs['vi_moisture_flux_div'],
                                 year=t.year,
                                 month=t.month,
                                 day=t.day)
        for t in proc_pixel_times
    }

    e5lsm = {'ERA5_land_sea_mask': (PATHS['era5dir'] /
                                    'data/invariants/ecmwf-era5_oper_an_sfc_200001010000.lsm.inv.nc')}

    return e5inputs, e5proc_shear, e5proc_vimfd, e5pixel_inputs, e5lsm


def monthly_meanfield_conditional_inputs(month):
    e5meanfield_inputs = {
        f'era5_{y}': fmtp(Era5MeanField.rule_outputs['meanfield'],
                          year=y,
                          month=month)
        for y in years
    }
    return e5meanfield_inputs


def meanfield_conditional_inputs():
    e5meanfield_inputs = {
        f'era5_{y}_{m}': fmtp(Era5MeanField.rule_outputs['meanfield'],
                              year=y,
                              month=m)
        for y in years
        for m in months
    }
    return e5meanfield_inputs


def open_25hr_data(paths):
    """These datasets have 25 hours of data in them - discard last hour for all but last.

    Originally this was so I could load one day and have all the info to interp to half-hourly
    times.
    """
    datasets = [xr.open_dataset(p) for p in paths]
    return xr.concat(
        [ds.isel(time=slice(24)) for ds in datasets[:-1]] +
        [datasets[-1]],
        dim='time'
    )


def conditional_load_mcs_data(logger, year, month, inputs):
    start = pd.Timestamp(year, month, 1)
    end = start + pd.DateOffset(months=1)
    proc_pixel_times = pd.date_range(start, end - pd.Timedelta(days=1), freq='D')

    tracks = McsTracks.open(inputs['tracks'], None)
    e5pixel_paths = [inputs[f'e5pixel_{t}']
                     for t in proc_pixel_times]
    e5pixel = open_25hr_data(e5pixel_paths)
    return tracks, e5pixel.load()


def conditional_load_data(logger, year, month, inputs):
    start = pd.Timestamp(year, month, 1)
    # end = start + pd.DateOffset(months=1)
    end = start + pd.DateOffset(days=2)
    e5times = pd.date_range(start, end, freq='H')
    proc_pixel_times = pd.date_range(start, end - pd.Timedelta(days=1), freq='D')

    tracks, e5pixel = conditional_load_mcs_data(logger, year, month, inputs)

    e5paths = [inputs[f'era5_{t}_{v}']
               for t in e5times
               for v in ERA5VARS]
    e5proc_shear_paths = [inputs[f'era5p_shear_{t}']
                          for t in proc_pixel_times]
    e5proc_vimfd_paths = [inputs[f'era5p_vimfd_{t}']
                          for t in proc_pixel_times]

    logger.debug('Open Pixel')
    mcs_times = pd.DatetimeIndex(e5pixel.time)

    logger.debug('Open ERA5')
    e5ds = (xr.open_mfdataset(e5paths).sel(latitude=slice(60, -60))
            .interp(time=mcs_times).sel(time=mcs_times))
    logger.debug('Open proc shear')
    e5shear = (open_25hr_data(e5proc_shear_paths)
               .interp(time=mcs_times).sel(time=mcs_times))
    logger.debug('Open proc VIMFD')
    e5vimfd = (open_25hr_data(e5proc_vimfd_paths)
               .interp(time=mcs_times).sel(time=mcs_times))

    return tracks, e5pixel.load(), e5ds.load(), e5shear.load(), e5vimfd.load()


def conditional_load_meanfield_data(logger, inputs):
    e5meanfield = xr.open_mfdataset([v for k, v in inputs.items() if k[:5] == 'era5_'])
    return e5meanfield.mean(dim='time').load()


def get_bins(var):
    if var == 'cape':
        bins = np.linspace(0, 5000, 101)
    elif var == 'tcwv':
        bins = np.linspace(0, 100, 101)
    elif var[-5:] == 'shear':
        bins = np.linspace(0, 100, 101)
    elif var == 'vimfd':
        bins = np.linspace(-1e-5, 1e-5, 101)
    hist_mids = (bins[1:] + bins[:-1]) / 2
    return bins, hist_mids


def load_lsmask(path):
    lsmask = {}
    for lsreg in LS_REGIONS:
        # Build appropriate land-sea mask for region.
        da_lsmask = xr.load_dataarray(path)
        if lsreg == 'all':
            # All ones.
            lsmask['all'] = da_lsmask[0].sel(latitude=slice(60, -60)).values >= 0
        elif lsreg == 'land':
            # LSM has land == 1.
            lsmask['land'] = da_lsmask[0].sel(latitude=slice(60, -60)).values > 0.5
        elif lsreg == 'ocean':
            # LSM has ocean == 0.
            lsmask['ocean'] = da_lsmask[0].sel(latitude=slice(60, -60)).values <= 0.5
        else:
            raise ValueError(f'Unknown region: {lsreg}')
    return lsmask


def gen_region_masks(logger, e5pixel, tracks):
    mcs_core_shield_mask = []
    # Looping over subset of times.
    for i, time in enumerate(e5pixel.time.values):
        pdtime = pd.Timestamp(time)
        time = pdtime.to_pydatetime()
        if pdtime.hour == 0:
            print(time)

        # Get cloudnumbers (cns) for tracks at given time.
        ts = tracks.tracks_at_time(time)
        # tmask is a 2d mask that spans multiple tracks, getting
        # the cloudnumbers at *one time only*, that can be
        # used to get cloudnumbers.
        tmask = (ts.dstracks.base_time == pdtime).values
        if tmask.sum() == 0:
            logger.info(f'No times matched in tracks DB for {pdtime}')
            cns = np.array([])
        else:
            # Each cloudnumber can be used to link to the corresponding
            # cloud in the pixel data.
            cns = ts.dstracks.cloudnumber.values[tmask]
            # Nicer to have sorted values.
            cns.sort()

        # Tracked MCS shield (N.B. close to Tb < 241K but expanded by precip regions).
        # INCLUDES CONV CORE.
        mcs_core_shield_mask.append(e5pixel.cloudnumber[i].isin(cns).values)

    mcs_core_shield_mask = np.array(mcs_core_shield_mask)
    # Convective core Tb < 225K.
    core_mask = e5pixel.tb.values < 225
    # Non-MCS clouds (Tb < 241K). INCLUDES CONV CORE.
    # OPERATOR PRECEDENCE! Brackets are vital here.
    cloud_core_shield_mask = (e5pixel.cloudnumber.values > 0) & ~mcs_core_shield_mask
    # MCS conv core only.
    mcs_core_mask = mcs_core_shield_mask & core_mask
    # Cloud conv core only.
    cloud_core_mask = cloud_core_shield_mask & core_mask
    # Env is everything outside of these two regions.
    env_mask = ~mcs_core_shield_mask & ~cloud_core_shield_mask

    # Remove conv core from shields.
    mcs_shield_mask = mcs_core_shield_mask & ~mcs_core_mask
    cloud_shield_mask = cloud_core_shield_mask & ~cloud_core_mask

    # Verify mutual exclusivity and that all points are covered.
    assert (
        mcs_core_mask.astype(int) +
        mcs_shield_mask.astype(int) +
        cloud_core_mask.astype(int) +
        cloud_shield_mask.astype(int) +
        env_mask.astype(int) == 1
    ).all()

    return mcs_core_mask, mcs_shield_mask, cloud_core_mask, cloud_shield_mask, env_mask


def build_hourly_output_dataset(e5pixel):
    # Build inputs to Dataset
    coords = {'time': e5pixel.time}
    data_vars = {}
    for var in ERA5VARS + ['LLS_shear', 'L2M_shear', 'MLS_shear'] + ['vimfd']:
        bins, hist_mids = get_bins(var)
        hists = np.zeros((len(e5pixel.time), hist_mids.size))

        coords.update({f'{var}_hist_mids': hist_mids, f'{var}_bins': bins})
        for lsreg in LS_REGIONS:
            data_vars.update({
                f'{lsreg}_{var}_MCS_shield': (('time', f'{var}_hist_mid'), hists.copy()),
                f'{lsreg}_{var}_MCS_core': (('time', f'{var}_hist_mid'), hists.copy()),
                f'{lsreg}_{var}_cloud_shield': (('time', f'{var}_hist_mid'), hists.copy()),
                f'{lsreg}_{var}_cloud_core': (('time', f'{var}_hist_mid'), hists.copy()),
                f'{lsreg}_{var}_env': (('time', f'{var}_hist_mid'), hists.copy()),
            })

    # Make a dataset to hold all the histogram data.
    dsout = xr.Dataset(
        coords=coords,
        data_vars=data_vars,
    )
    return dsout


def to_netcdf_tmp_then_copy(ds, outpath, encoding=None):
    if encoding is None:
        encoding = {}
    tmpdir = Path('/work/scratch-nopw/mmuetz')
    assert outpath.is_absolute()
    tmppath = tmpdir / Path(*outpath.parts[1:])
    tmppath.parent.mkdir(exist_ok=True, parents=True)

    ds.to_netcdf(tmppath, encoding=encoding)
    shutil.move(tmppath, outpath)


class ConditionalERA5HistHourly(TaskRule):
    @staticmethod
    def rule_inputs(year, month):
        e5inputs, e5proc_shear, e5proc_vimfd, e5pixel_inputs, e5lsm = conditional_inputs(year, month)
        inputs = {
            **e5inputs,
            **e5proc_shear,
            **e5proc_vimfd,
            **e5pixel_inputs,
            **e5lsm,
        }
        return inputs

    rule_outputs = {'hist': (PATHS['outdir'] / 'conditional_era5_histograms' /
                             '{year}' /
                             'hourly_hist_{year}_{month:02d}.nc')}

    var_matrix = {
        'year': years,
        'month': months,
    }

    depends_on = [
        conditional_load_mcs_data,
        conditional_load_data,
        load_lsmask,
        build_hourly_output_dataset,
        gen_region_masks
    ]

    def rule_run(self):
        self.logger.info('Load data')
        tracks, e5pixel, e5ds, e5shear, e5vimfd = conditional_load_data(
            self.logger,
            self.year,
            self.month,
            self.inputs
        )
        lsmask = load_lsmask(self.inputs['ERA5_land_sea_mask'])

        self.logger.info('Build output datasets')
        dsout = build_hourly_output_dataset(e5pixel)

        self.logger.info('Generate region masks')
        (
            mcs_core_mask,
            mcs_shield_mask,
            cloud_core_mask,
            cloud_shield_mask,
            env_mask
        ) = gen_region_masks(self.logger, e5pixel, tracks)

        level2index = {
            'LLS_shear': 0,
            'L2M_shear': 1,
            'MLS_shear': 2,
        }
        # Pack all dataarrays into a common struct for easy looping.
        dataarrays = (
            # ERA5 variables.
            [(var, e5ds[var])
             for var in ERA5VARS] +
            # Derived shear variables.
            [(var, e5shear.isel(shear_level=level2index[var]).shear)
             for var in ['LLS_shear', 'L2M_shear', 'MLS_shear']] +
            # Derived VIMFD variables.
            [('vimfd', e5vimfd.vertically_integrated_moisture_flux_div)]
        )
        self.logger.info('Calc hists at each time')
        for i, time in enumerate(e5pixel.time.values):
            pdtime = pd.Timestamp(time)
            if pdtime.hour == 0:
                print(pdtime)
            def hist(data):
                # closure to save space.
                return np.histogram(data, bins=dsout.coords[f'{var}_bins'].values)[0]

            for (var, da), lsreg in product(dataarrays, LS_REGIONS):
                data = da.sel(time=pdtime).values
                # Calc hists. These 5 regions are mutually exclusive.
                dsout[f'{lsreg}_{var}_MCS_shield'][i] = hist(data[mcs_shield_mask[i] & lsmask[lsreg]])
                dsout[f'{lsreg}_{var}_MCS_core'][i] = hist(data[mcs_core_mask[i] & lsmask[lsreg]])
                dsout[f'{lsreg}_{var}_cloud_shield'][i] = hist(data[cloud_shield_mask[i] & lsmask[lsreg]])
                dsout[f'{lsreg}_{var}_cloud_core'][i] = hist(data[cloud_core_mask[i] & lsmask[lsreg]])
                dsout[f'{lsreg}_{var}_env'][i] = hist(data[env_mask[i] & lsmask[lsreg]])

        self.logger.info('write dsout')
        to_netcdf_tmp_then_copy(dsout, self.outputs['hist'])


class ConditionalERA5HistGridpoint(TaskRule):
    @staticmethod
    def rule_inputs(year, month):
        e5inputs, e5proc_shear, e5proc_vimfd, e5pixel_inputs, e5lsm = conditional_inputs(year, month)
        inputs = {
            **e5inputs,
            **e5proc_shear,
            **e5proc_vimfd,
            **e5pixel_inputs,
            **e5lsm,
        }
        return inputs

    rule_outputs = {'hist': (PATHS['outdir'] / 'conditional_era5_histograms' /
                             '{year}' /
                             'gridpoint_hist_{year}_{month:02d}.nc')}

    var_matrix = {
        'year': years,
        'month': months,
    }

    depends_on = [
        conditional_load_mcs_data,
        conditional_load_data,
        load_lsmask,
        gen_region_masks
    ]

    def build_gridpoint_output_dataset(self, e5pixel):
        # Build inputs to Dataset
        coords = {'latitude': e5pixel.latitude, 'longitude': e5pixel.longitude}
        data_vars = {}
        for var in ERA5VARS + ['LLS_shear', 'L2M_shear', 'MLS_shear'] + ['vimfd']:
            bins, hist_mids = get_bins(var)
            hists = np.zeros((len(e5pixel.latitude), len(e5pixel.longitude), hist_mids.size))

            coords.update({f'{var}_hist_mids': hist_mids, f'{var}_bins': bins})
            data_vars.update({
                f'{var}_MCS_shield': (('latitude', 'longitude', f'{var}_hist_mid'), hists.copy()),
                f'{var}_MCS_core': (('latitude', 'longitude', f'{var}_hist_mid'), hists.copy()),
                f'{var}_cloud_shield': (('latitude', 'longitude', f'{var}_hist_mid'), hists.copy()),
                f'{var}_cloud_core': (('latitude', 'longitude', f'{var}_hist_mid'), hists.copy()),
                f'{var}_env': (('latitude', 'longitude', f'{var}_hist_mid'), hists.copy()),
            })

        # Make a dataset to hold all the histogram data.
        dsout = xr.Dataset(
            coords=coords,
            data_vars=data_vars,
        )
        return dsout

    def rule_run(self):
        self.logger.info('Load data')
        tracks, e5pixel, e5ds, e5shear, e5vimfd = conditional_load_data(
            self.logger,
            self.year,
            self.month,
            self.inputs
        )

        self.logger.info('Build output datasets')
        dsout = self.build_gridpoint_output_dataset(e5pixel)

        self.logger.info('Generate region masks')
        (
            mcs_core_mask,
            mcs_shield_mask,
            cloud_core_mask,
            cloud_shield_mask,
            env_mask
        ) = gen_region_masks(self.logger, e5pixel, tracks)

        level2index = {
            'LLS_shear': 0,
            'L2M_shear': 1,
            'MLS_shear': 2,
        }
        # Pack all dataarrays into a common struct for easy looping.
        dataarrays = (
            # ERA5 variables.
            [(var, e5ds[var])
             for var in ERA5VARS] +
            # Derived shear variables.
            [(var, e5shear.isel(shear_level=level2index[var]).shear)
             for var in ['LLS_shear', 'L2M_shear', 'MLS_shear']] +
            # Derived VIMFD variables.
            [('vimfd', e5vimfd.vertically_integrated_moisture_flux_div)]
        )

        self.logger.info('Calc hists at each gridpoint')
        for (var, da) in dataarrays:
            def hist(data):
                # closure to save space.
                return np.histogram(data, bins=dsout.coords[f'{var}_bins'].values)[0]

            data = da.values
            for i in range(data.shape[1]):
                if i % 10 == 0:
                    print(var, i, i / data.shape[1])
                for j in range(data.shape[2]):
                    dsout[f'{var}_MCS_shield'][i, j] = hist(data[:, i, j][mcs_shield_mask[:, i, j]])
                    dsout[f'{var}_MCS_core'][i, j] = hist(data[:, i, j][mcs_core_mask[:, i, j]])
                    dsout[f'{var}_cloud_shield'][i, j] = hist(data[:, i, j][cloud_shield_mask[:, i, j]])
                    dsout[f'{var}_cloud_core'][i, j] = hist(data[:, i, j][cloud_core_mask[:, i, j]])
                    dsout[f'{var}_env'][i, j] = hist(data[:, i, j][env_mask[:, i, j]])

        # These files are large. Use compression (makes write faster?).
        self.logger.info('write dsout')
        comp = dict(zlib=True, complevel=4)
        encoding = {var: comp for var in dsout.data_vars}
        to_netcdf_tmp_then_copy(dsout, self.outputs['hist'], encoding)


class ConditionalERA5HistMeanfield(TaskRule):
    @staticmethod
    def rule_inputs(year, month):
        e5inputs, e5proc_shear, e5proc_vimfd, e5pixel_inputs, e5lsm = conditional_inputs(year, month)
        e5meanfield_inputs = meanfield_conditional_inputs()
        inputs = {
            **e5pixel_inputs,
            **e5lsm,
            **e5meanfield_inputs,
        }
        return inputs

    rule_outputs = {'hist': (PATHS['outdir'] / 'conditional_era5_histograms' /
                             '{year}' /
                             'meanfield_hist_{year}_{month:02d}.nc')}

    var_matrix = {
        'year': years,
        'month': months,
    }
    depends_on = [conditional_load_mcs_data, conditional_load_meanfield_data, gen_region_masks]

    def rule_run(self):
        self.logger.info('Load data')
        tracks, e5pixel = conditional_load_mcs_data(
            self.logger,
            self.year,
            self.month,
            self.inputs
        )
        e5meanfield = conditional_load_meanfield_data(
            self.logger,
            self.inputs,
        )
        lsmask = load_lsmask(self.inputs['ERA5_land_sea_mask'])

        self.logger.info('Build output datasets')
        dsout = build_hourly_output_dataset(e5pixel)

        self.logger.info('Generate region masks')
        (
            mcs_core_mask,
            mcs_shield_mask,
            cloud_core_mask,
            cloud_shield_mask,
            env_mask
        ) = gen_region_masks(self.logger, e5pixel, tracks)

        level2index = {
            'LLS_shear': 0,
            'L2M_shear': 1,
            'MLS_shear': 2,
        }
        # Pack all dataarrays into a common struct for easy looping.
        dataarrays = (
            [(var, e5meanfield[var])
             for var in ERA5VARS] +
            # Derived shear variables.
            [(var, e5meanfield.isel(shear_level=level2index[var]).shear)
             for var in ['LLS_shear', 'L2M_shear', 'MLS_shear']] +
            # Derived VIMFD variables.
            [('vimfd', e5meanfield.vertically_integrated_moisture_flux_div)]
        )

        self.logger.info('Calc meanfield hists at each gridpoint')
        for i, time in enumerate(e5pixel.time.values):
            pdtime = pd.Timestamp(time)
            if pdtime.hour == 0:
                print(pdtime)
            def hist(data):
                # closure to save space.
                return np.histogram(data, bins=dsout.coords[f'{var}_bins'].values)[0]

            for (var, da), lsreg in product(dataarrays, LS_REGIONS):
                data = da.values
                # Calc hists. These 5 regions are mutually exclusive.
                dsout[f'{lsreg}_{var}_MCS_shield'][i] = hist(data[mcs_shield_mask[i] & lsmask[lsreg]])
                dsout[f'{lsreg}_{var}_MCS_core'][i] = hist(data[mcs_core_mask[i] & lsmask[lsreg]])
                dsout[f'{lsreg}_{var}_cloud_shield'][i] = hist(data[cloud_shield_mask[i] & lsmask[lsreg]])
                dsout[f'{lsreg}_{var}_cloud_core'][i] = hist(data[cloud_core_mask[i] & lsmask[lsreg]])
                dsout[f'{lsreg}_{var}_env'][i] = hist(data[env_mask[i] & lsmask[lsreg]])

        self.logger.info('write dsout')
        to_netcdf_tmp_then_copy(dsout, self.outputs['hist'])


class CombineConditionalERA5HistGridpoint(TaskRule):
    @staticmethod
    def rule_inputs(year):
        inputs = {
            f'hist_{year}_{month}': fmtp(ConditionalERA5HistGridpoint.rule_outputs['hist'],
                                         year=year,
                                         month=month)
            for month in months
        }
        return inputs

    rule_outputs = {'hist': (PATHS['outdir'] / 'conditional_era5_histograms' /
                             '{year}' /
                             'gridpoint_hist_{year}.nc')}

    var_matrix = {'year': years}
    # Takes a lot of mem to combine these datasets!
    config = {'slurm': {'mem': 512000, 'partition': 'high-mem'}}

    def rule_run(self):
        datasets = [xr.open_dataset(p) for p in self.inputs.values()]
        assert len(datasets) == 12

        self.logger.info('Concat datasets')
        ds = xr.concat(datasets, pd.Index(range(12), name='time_index'))
        dsout = ds.sum(dim='time_index')

        self.logger.info('Write ds.sum')
        comp = dict(zlib=True, complevel=4)
        encoding = {var: comp for var in dsout.data_vars}
        to_netcdf_tmp_then_copy(dsout, self.outputs['hist'], encoding)

