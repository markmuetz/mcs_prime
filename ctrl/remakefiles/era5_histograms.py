import calendar
import datetime as dt
from functools import partial
from itertools import product

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
slurm_config = {'queue': 'short-serial', 'mem': 64000, 'max_runtime': '10:00:00'}
# slurm_config = {'account': 'short4hr', 'queue': 'short-serial-4hr', 'mem': 32000}
era5_histograms = Remake(config=dict(slurm=slurm_config, content_checks=False))

# years = list(range(2000, 2021))
years = [2020]
months = range(1, 13)

# For testing - 5 days.
DATES = pd.date_range(f'{years[0]}-01-01', f'{years[0]}-01-05')
# DATES = pd.date_range(f'{years[0]}-01-01', f'{years[-1]}-12-31')
DATE_KEYS = [(y, m, d) for y, m, d in zip(DATES.year, DATES.month, DATES.day)]


class ERA5Hist(TaskRule):
    enabled = False
    @staticmethod
    def rule_inputs(year, month, var):
        days_in_month = calendar.monthrange(year, month)[1]
        hours = range(24)
        inputs = {f'day_hr_{d}_{h}': (PATHS['era5dir'] / 'data/oper/an_sfc/'
                                      f'{year}/{month:02d}/{d:02d}'
                                      f'/ecmwf-era5_oper_an_sfc_'
                                      f'{year}{month:02d}{d:02d}{h:02d}00.{var}.nc')
                  for d in range(1, days_in_month + 1)
                  for h in hours}
        return inputs
    rule_outputs = {'hist': (PATHS['outdir'] / 'era5_histograms' /
                             '{year}' /
                             'monthly_{var}_hist_{year}_{month:02d}.nc')}

    var_matrix = {'year': years, 'month': months, 'var': ['cape', 'tcwv']}

    def rule_run(self):
        month_start_day = dt.datetime(self.year, self.month, 1).timetuple().tm_yday
        month_files = list(self.inputs.values())
        if self.var == 'cape':
            bins = np.linspace(0, 5000, 501)
        elif self.var == 'tcwv':
            bins = np.linspace(0, 100, 101)
        mids = (bins[1:] + bins[:-1]) / 2

        hists = np.zeros((len(month_files) // 24, bins.size - 1))

        ndays = len(month_files) // 24
        dsout = xr.Dataset(
            coords=dict(
                day=month_start_day + np.arange(ndays),
                hist_mid=mids,
                bins=bins,
            ),
            data_vars={
                f'{self.var}_full': (('day', 'hist_mid'), hists.copy()),
                f'{self.var}_tropics': (('day', 'hist_mid'), hists.copy()),
                f'{self.var}_eq': (('day', 'hist_mid'), hists.copy()),
            }
        )

        for i in range(0, len(month_files), 24):
            j = i // 24
            print(i, j)
            # Just fill in the bins argument.
            hist = partial(np.histogram, bins=bins)

            day_files = month_files[i:i + 24]
            da = xr.open_mfdataset(day_files).sel(latitude=slice(60, -60))[self.var]
            dsout[f'{self.var}_full'][j] = hist(da.values)[0]
            dsout[f'{self.var}_tropics'][j] = hist(da.sel(latitude=slice(20, -20)).values)[0]
            dsout[f'{self.var}_eq'][j] = hist(da.sel(latitude=slice(5, -5)).values)[0]
        dsout.to_netcdf(self.outputs['hist'])


class CombineERA5Hist(TaskRule):
    enabled = False
    @staticmethod
    def rule_inputs(year, var):
        inputs = {f'hist_{m}': fmtp(ERA5Hist.rule_outputs['hist'], year=year, month=m, var=var)
                  for m in months}
        return inputs

    rule_outputs = {'hist': (PATHS['outdir'] / 'era5_histograms' /
                             '{year}' /
                             'yearly_{var}_hist_{year}.nc')}

    var_matrix = {'year': years, 'var': ['cape', 'tcwv']}

    def rule_run(self):
        ds = xr.open_mfdataset(self.inputs.values())
        ds.to_netcdf(self.outputs['hist'])


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


def gen_times(year, month):
    e5times = []
    time = dt.datetime(year, month, 1, 0)
    if month == 12:
        end = dt.datetime(year + 1, 1, 1, 0)
    else:
        end = dt.datetime(year, month + 1, 1, 0)

    while time <= end:
        e5times.append(time)
        time += dt.timedelta(minutes=60)
    return e5times, [t + dt.timedelta(minutes=30) for t in e5times[:-1]]


class OldConditionalERA5ShearHist(TaskRule):
    enabled = False
    @staticmethod
    def rule_inputs(year, month):
        e5times, pixel_times = gen_times(year, month)
        e5inputs = {f'era5_{t}_{var}': (PATHS['era5dir'] /
                                        f'data/oper/an_ml/{t.year}/{t.month:02d}/{t.day:02d}' /
                                        (f'ecmwf-era5_oper_an_ml_{t.year}{t.month:02d}{t.day:02d}'
                                         f'{t.hour:02d}00.{var}.nc'))
                    for t in e5times
                    for var in ['u', 'v']}

        if year == 2000:
            start_date = '20000601'
        else:
            start_date = f'{year}0101'
        # Not all files exist! Do not include these.
        pixel_paths = [(PATHS['pixeldir'] /
                        f'{start_date}.0000_{t.year + 1}0101.0000'
                        f'/{t.year}/{t.month:02d}/{t.day:02d}' /
                        f'mcstrack_{t.year}{t.month:02d}{t.day:02d}_{t.hour:02d}30.nc')
                       for t in pixel_times]
        pixel_inputs = {f'pixel_{t}': p
                        for t, p in zip(pixel_times, pixel_paths)
                        if p.exists()}
        inputs = {**e5inputs, **pixel_inputs}

        inputs['tracks'] = (PATHS['statsdir'] /
                            f'mcs_tracks_final_extc_{start_date}.0000_{year + 1}0101.0000.nc')
        inputs['regridder'] = GenRegridder.rule_outputs['regridder']
        return inputs

    rule_outputs = {'hist': (PATHS['outdir'] / 'conditional_era5_histograms' /
                             '{year}' /
                             'monthly_shear_hist_{year}_{month:02d}.nc')}

    var_matrix = {'year': years, 'month': months}

    def rule_run(self):
        levels = [
            133,  # 1000hPa/107m
            111,  # 804hPa/1911m
            96,  # 508hPa/5469m
            74,  # 197hPa/11890m
        ]
        bins = np.linspace(0, 100, 101)

        tracks = McsTracks.open(self.inputs['tracks'], None)

        e5times, orig_pixel_times = gen_times(self.year, self.month)
        e5inputs = {(t, var): self.inputs[f'era5_{t}_{var}']
                    for t in e5times
                    for var in ['u', 'v']}
        # Note, this is a subset of times based on which times exist.
        pixel_times = [t for t in orig_pixel_times if f'pixel_{t}' in self.inputs]
        pixel_inputs = {t: self.inputs[f'pixel_{t}']
                        for t in pixel_times}

        month_start_day = dt.datetime(self.year, self.month, 1).timetuple().tm_yday
        mids = (bins[1:] + bins[:-1]) / 2

        hists = np.zeros((len(pixel_times), len(levels), mids.size))

        # Make a dataset to hold all the histogram data.
        # max. len time is 24x31.
        dsout = xr.Dataset(
            coords=dict(
                time=pixel_times,
                hist_mid=mids,
                levels=levels,
                bins=bins,
            ),
            data_vars={
                f'shear_MCS': (('time', 'levels', 'hist_mid'), hists.copy()),
                f'shear_conv': (('time', 'levels', 'hist_mid'), hists.copy()),
                f'shear_env': (('time', 'levels', 'hist_mid'), hists.copy()),
            }
        )
        regridder = None
        # Looping over subset of times.
        for i, time in enumerate(pixel_times[:5]):
            print(time)
            # Get cloudnumbers (cns) for tracks at given time.
            pdtime = pd.Timestamp(time)
            ts = tracks.tracks_at_time(time)
            frame = ts.pixel_data.get_frame(time)
            # tmask is a 2d mask that spans multiple tracks, getting
            # the cloudnumbers at *one time only*, that can be
            # used to get cloudnumbers.
            tmask = (ts.dstracks.base_time == pdtime).values
            if tmask.sum() == 0:
                self.logger.warn(f'No times matched in tracks DB for {pdtime}')
                dsout[f'shear_MCS'][i].values[:] = None
                dsout[f'shear_conv'][i].values[:] = None
                dsout[f'shear_env'][i].values[:] = None
                continue
            # Each cloudnumber can be used to link to the corresponding
            # cloud in the pixel data.
            cns = ts.dstracks.cloudnumber.values[tmask]
            # Nicer to have sorted values.
            cns.sort()

            # Load the pixel data.
            dspixel = xr.open_dataset(pixel_inputs[time])
            pixel_precip = dspixel.precipitation.isel(time=0).load()

            # Load the e5 data and interp in time to pixel time.
            e5time1 = time - dt.timedelta(minutes=30)
            e5time2 = time + dt.timedelta(minutes=30)
            e5paths = [e5inputs[(t, var)]
                       for t in [e5time1, e5time2]
                       for var in ['u', 'v']]
            e5data = (xr.open_mfdataset(e5paths)
                      .mean(dim='time').sel(latitude=slice(60, -60), level=levels).load())
            u = e5data.u.values
            v = e5data.v.values
            # Calc LLS, MLS, HLS, DS in one line.
            # LLS = shear[:, :, 0]
            # MLS = shear[:, :, 1]
            # HLS = shear[:, :, 2]
            # DS = shear[:, :, 3]
            shear = np.sqrt((np.roll(u, -1, axis=0) - u)**2 + (np.roll(v, -1, axis=0) - v)**2)

            if regridder is None:
                # Reload the regridder.
                regridder = xe.Regridder(pixel_precip, e5data, 'bilinear', periodic=True,
                                         reuse_weights=True, weights=self.inputs['regridder'])

            # This is a lot faster than doing/cloudnumber! If you don't need the individual
            # cloudnumber info you can use this to partition the 2D grid into MCS/non-MCS.
            e5mcs_mask = regridder(frame.dspixel.cloudnumber.isin(cns).astype(float)) > 0.5
            mcs_mask = e5mcs_mask[0]
            e5tb = regridder(dspixel.tb)
            # N.B. Feng et al. 2021 uses Tb < 225 to define cold cloud cores.
            e5conv_mask = e5tb[0] < 225

            # The three masks used are mutually exclusive:
            # mcs_mask -- all values within MCS CCS.
            # ~mcs_mask & e5conv_mask -- all values not in MCS but in convective regions.
            # ~mcs_mask & ~e5conv_mask -- env.
            assert (mcs_mask.sum() +
                    (~mcs_mask & e5conv_mask).sum() +
                    (~mcs_mask & ~e5conv_mask).sum()) == mcs_mask.size

            # Calc hists.
            hist = partial(np.histogram, bins=bins)
            for j, level in enumerate(levels):
                dsout[f'shear_MCS'][i, j] = hist(shear[j][mcs_mask])[0]
                dsout[f'shear_conv'][i, j] = hist(shear[j][~mcs_mask & e5conv_mask])[0]
                dsout[f'shear_env'][i, j] = hist(shear[j][~mcs_mask & ~e5conv_mask])[0]
        dsout.to_netcdf(self.outputs['hist'])


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
REGIONS = ['all', 'land', 'ocean']


class ConditionalERA5Hist(TaskRule):
    @staticmethod
    def rule_inputs(year, month, day):
        start = dt.datetime(year, month, day)
        # Note there are 25 of these so I can get ERA5 data on the hour either side
        # of MCS dataset data (on the half hour).
        e5times = pd.date_range(start, start + dt.timedelta(days=1), freq='H')

        e5inputs = {f'era5_{t}_{var}': (PATHS['era5dir'] /
                                        f'data/oper/an_sfc/{year}/{month:02d}/{day:02d}' /
                                        (f'ecmwf-era5_oper_an_sfc_{year}{month:02d}{day:02d}'
                                         f'{t.hour:02d}00.{var}.nc'))
                    for t in e5times
                    for var in ERA5VARS}
        e5proc_inputs = {'era5p_shear': fmtp(GenERA5Shear.rule_outputs['shear'],
                                             year=year,
                                             month=month,
                                             day=day),
                         'era5p_vimfd': fmtp(GenERA5VIMoistureFluxDiv.rule_outputs['vi_moisture_flux_div'],
                                             year=year,
                                             month=month,
                                             day=day),
                             }

        e5pixel_inputs = {'e5pixel': fmtp(GenERA5PixelData.rule_outputs['e5pixel'],
                                          year=year,
                                          month=month,
                                          day=day)}
        if year == 2000:
            start_date = '20000601'
        else:
            start_date = f'{year}0101'
        e5pixel_inputs['tracks'] = (PATHS['statsdir'] /
                                    f'mcs_tracks_final_extc_{start_date}.0000_{year + 1}0101.0000.nc')

        inputs = {
            **e5inputs,
            **e5proc_inputs,
            **e5pixel_inputs,
            'ERA5_land_sea_mask': PATHS['era5dir'] / 'data/invariants/ecmwf-era5_oper_an_sfc_200001010000.lsm.inv.nc'}
        return inputs

    rule_outputs = {'hist': (PATHS['outdir'] / 'conditional_era5_histograms' /
                             '{year}' /
                             'daily_hist_{year}_{month:02d}_{day:02d}.nc')}

    var_matrix = {
        ('year', 'month', 'day'): DATE_KEYS,
    }

    def rule_run(self):
        tracks = McsTracks.open(self.inputs['tracks'], None)

        start = dt.datetime(self.year, self.month, self.day)
        # Note there are 25 of these so I can get ERA5 data on the hour either side
        # of MCS dataset data (on the half hour).
        e5times = pd.date_range(start, start + dt.timedelta(days=1), freq='H')

        e5inputs = {(t, v): self.inputs[f'era5_{t}_{v}']
                    for t in e5times
                    for v in ERA5VARS}

        e5pixel = xr.load_dataset(self.inputs['e5pixel'])
        e5shear = xr.load_dataarray(self.inputs['era5p_shear'])
        e5vimfd = xr.load_dataarray(self.inputs['era5p_vimfd'])

        # Build inputs to Dataset
        coords = {'time': e5pixel.time}
        data_vars = {}
        for var in ERA5VARS + ['LLS_shear', 'L2M_shear', 'MLS_shear'] + ['vimfd']:
            if var == 'cape':
                bins = np.linspace(0, 5000, 501)
            elif var == 'tcwv':
                bins = np.linspace(0, 100, 101)
            elif var[-5:] == 'shear':
                bins = np.linspace(0, 100, 101)
            elif var == 'vimfd':
                bins = np.linspace(-1e-5, 1e-5, 101)
            hist_mids = (bins[1:] + bins[:-1]) / 2
            hists = np.zeros((len(e5pixel.time), hist_mids.size))
            coords.update({f'{var}_hist_mids': hist_mids, f'{var}_bins': bins})
            for reg in REGIONS:
                data_vars.update({
                    f'{reg}_{var}_MCS_shield': (('time', f'{var}_hist_mid'), hists.copy()),
                    f'{reg}_{var}_MCS_core': (('time', f'{var}_hist_mid'), hists.copy()),
                    f'{reg}_{var}_cloud_shield': (('time', f'{var}_hist_mid'), hists.copy()),
                    f'{reg}_{var}_cloud_core': (('time', f'{var}_hist_mid'), hists.copy()),
                    f'{reg}_{var}_env': (('time', f'{var}_hist_mid'), hists.copy()),
                })

        # Make a dataset to hold all the histogram data.
        dsout = xr.Dataset(
            coords=coords,
            data_vars=data_vars,
        )

        regmask = {}
        for reg in REGIONS:
            # Build appropriate land-sea mask for region.
            da_lsmask = xr.open_dataarray(self.inputs['ERA5_land_sea_mask'])
            if reg == 'all':
                # All ones.
                regmask['all'] = da_lsmask[0].sel(latitude=slice(60, -60)).values >= 0
            elif reg == 'land':
                # LSM has land == 1.
                regmask['land'] = da_lsmask[0].sel(latitude=slice(60, -60)).values > 0.5
            elif reg == 'ocean':
                # LSM has ocean == 0.
                regmask['ocean'] = da_lsmask[0].sel(latitude=slice(60, -60)).values <= 0.5
            else:
                raise ValueError(f'Unknown region: {reg}')

        # Looping over subset of times.
        for i, time in enumerate(e5pixel.time.values):
            pdtime = pd.Timestamp(time)
            time = pdtime.to_pydatetime()
            print(time)

            # Get cloudnumbers (cns) for tracks at given time.
            ts = tracks.tracks_at_time(time)
            # tmask is a 2d mask that spans multiple tracks, getting
            # the cloudnumbers at *one time only*, that can be
            # used to get cloudnumbers.
            tmask = (ts.dstracks.base_time == pdtime).values
            if tmask.sum() == 0:
                self.logger.info(f'{self}: No times matched in tracks DB for {pdtime}')
                cns = np.array([])
            else:
                # Each cloudnumber can be used to link to the corresponding
                # cloud in the pixel data.
                cns = ts.dstracks.cloudnumber.values[tmask]
                # Nicer to have sorted values.
                cns.sort()

            # Convective core Tb < 225K.
            core_mask = e5pixel.tb[i].values < 225
            # Tracked MCS shield (N.B. close to Tb < 241K but expanded by precip regions).
            # INCLUDES CONV CORE.
            mcs_core_shield_mask = e5pixel.cloudnumber[i].isin(cns).values
            # Non-MCS clouds (Tb < 241K). INCLUDES CONV CORE.
            cloud_core_shield_mask = e5pixel.cloudnumber[i].values > 0 & ~mcs_core_shield_mask
            # MCS conv core only.
            mcs_core_mask = mcs_core_shield_mask & core_mask
            # Cloud conv core only.
            cloud_core_mask = cloud_core_shield_mask & core_mask
            # Env is everything outside of these two regions.
            env_mask = ~mcs_core_shield_mask & ~cloud_core_shield_mask

            # Remove conv core from shields.
            mcs_shield_mask = mcs_core_shield_mask & ~mcs_core_mask
            cloud_shield_mask = cloud_core_shield_mask & ~cloud_core_mask

            # Load the e5 data and interp in time to pixel time.
            # TODO: could be made more efficient. Load only if needed then concat.
            e5time1 = time - dt.timedelta(minutes=30)
            e5time2 = time + dt.timedelta(minutes=30)
            paths = [e5inputs[t, v]
                     for t in [e5time1, e5time2]
                     for v in ERA5VARS]
            e5data = xr.open_mfdataset(paths).mean(dim='time').sel(latitude=slice(60, -60)).load()

            def hist(data):
                # closure to save space.
                return np.histogram(data, bins=dsout.coords[f'{var}_bins'].values)[0]

            for var, reg in product(ERA5VARS, REGIONS):
                data = e5data[var].values
                # Calc hists. These 5 regions are mutually exclusive.
                dsout[f'{reg}_{var}_MCS_shield'][i] = hist(data[mcs_shield_mask & regmask[reg]])
                dsout[f'{reg}_{var}_MCS_core'][i] = hist(data[mcs_core_mask & regmask[reg]])
                dsout[f'{reg}_{var}_cloud_shield'][i] = hist(data[cloud_shield_mask & regmask[reg]])
                dsout[f'{reg}_{var}_cloud_core'][i] = hist(data[cloud_core_mask & regmask[reg]])
                dsout[f'{reg}_{var}_env'][i] = hist(data[env_mask & regmask[reg]])

            level2index = {
                'LLS_shear': 0,
                'L2M_shear': 1,
                'MLS_shear': 2,
            }

            for var, reg in product(['LLS_shear', 'L2M_shear', 'MLS_shear'], REGIONS):
                level_index = level2index[var]
                # Time mean and select each shear level.
                shear = e5shear[i:i + 2, level_index].mean(dim='time').values
                # Calc hists. These 5 regions are mutually exclusive.
                dsout[f'{reg}_{var}_MCS_shield'][i] = hist(shear[mcs_shield_mask & regmask[reg]])
                dsout[f'{reg}_{var}_MCS_core'][i] = hist(shear[mcs_core_mask & regmask[reg]])
                dsout[f'{reg}_{var}_cloud_shield'][i] = hist(shear[cloud_shield_mask & regmask[reg]])
                dsout[f'{reg}_{var}_cloud_core'][i] = hist(shear[cloud_core_mask & regmask[reg]])
                dsout[f'{reg}_{var}_env'][i] = hist(shear[env_mask & regmask[reg]])

            for var, reg in product(['vimfd'], REGIONS):
                # Time mean.
                vimfd = e5vimfd[i:i + 2].mean(dim='time').values
                # Calc hists. These 5 regions are mutually exclusive.
                dsout[f'{reg}_{var}_MCS_shield'][i] = hist(vimfd[mcs_shield_mask & regmask[reg]])
                dsout[f'{reg}_{var}_MCS_core'][i] = hist(vimfd[mcs_core_mask & regmask[reg]])
                dsout[f'{reg}_{var}_cloud_shield'][i] = hist(vimfd[cloud_shield_mask & regmask[reg]])
                dsout[f'{reg}_{var}_cloud_core'][i] = hist(vimfd[cloud_core_mask & regmask[reg]])
                dsout[f'{reg}_{var}_env'][i] = hist(vimfd[env_mask & regmask[reg]])
        dsout.to_netcdf(self.outputs['hist'])


class CombineConditionalERA5Hist(TaskRule):
    @staticmethod
    def rule_inputs(year):
        inputs = {f'hist_{d}': fmtp(ConditionalERA5Hist.rule_outputs['hist'],
                                    year=d.year, month=d.month, day=d.day)
                  for d in DATES[DATES.year == year]}
        return inputs

    rule_outputs = {'hist': (PATHS['outdir'] / 'conditional_era5_histograms' /
                             '{year}' /
                             'yearly_hist_{year}.nc')}

    var_matrix = {'year': years}

    def rule_run(self):
        # Note, some datasets have zero length time dim.
        # These will raise a "Cannot handle size zero dimensions" exception
        # on xr.open_mfdataset.
        # This is a simple method that ignores these files.
        filtered_paths = []
        for path in self.inputs.values():
            ds = xr.open_dataset(path)
            if ds.time.size != 0:
                filtered_paths.append(path)

        with xr.open_mfdataset(filtered_paths) as ds:
            ds.to_netcdf(self.outputs['hist'])

