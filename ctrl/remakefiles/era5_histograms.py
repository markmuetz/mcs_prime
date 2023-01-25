import calendar
import datetime as dt
from functools import partial

import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe


from remake import Remake, TaskRule
from remake.util import format_path as fmtp

from mcs_prime import PATHS, McsTracks, PixelData

slurm_config = {'account': 'short4hr', 'queue': 'short-serial-4hr', 'mem': 32000}
era5_histograms = Remake(config=dict(slurm=slurm_config, content_checks=False))

years = [2020]
months = range(1, 13)

dates = pd.date_range(f'{years[0]}-01-01', f'{years[0]}-12-31')
date_keys = [(y, m, d) for y, m, d in zip(dates.year, dates.month, dates.day)]


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


class OldConditionalERA5Hist(TaskRule):
    enabled = False
    @staticmethod
    def rule_inputs(year, month, var):
        e5times, pixel_times = gen_times(year, month)
        e5inputs = {f'era5_{t}': (PATHS['era5dir'] /
                                  f'data/oper/an_sfc/{t.year}/{t.month:02d}/{t.day:02d}' /
                                  (f'ecmwf-era5_oper_an_sfc_{t.year}{t.month:02d}{t.day:02d}'
                                   f'{t.hour:02d}00.{var}.nc'))
                    for t in e5times}

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
                             'monthly_{var}_hist_{year}_{month:02d}.nc')}

    var_matrix = {'year': years, 'month': months, 'var': ['cape', 'tcwv']}

    def rule_run(self):
        tracks = McsTracks.open(self.inputs['tracks'], None)

        e5times, orig_pixel_times = gen_times(self.year, self.month)
        e5inputs = {t: self.inputs[f'era5_{t}']
                    for t in e5times}
        # Note, this is a subset of times based on which times exist.
        pixel_times = [t for t in orig_pixel_times if f'pixel_{t}' in self.inputs]
        pixel_inputs = {t: self.inputs[f'pixel_{t}']
                        for t in pixel_times}

        month_start_day = dt.datetime(self.year, self.month, 1).timetuple().tm_yday
        if self.var == 'cape':
            bins = np.linspace(0, 5000, 501)
        elif self.var == 'tcwv':
            bins = np.linspace(0, 100, 101)
        mids = (bins[1:] + bins[:-1]) / 2

        hists = np.zeros((len(pixel_times), mids.size))

        # Make a dataset to hold all the histogram data.
        # max. len time is 24x31.
        dsout = xr.Dataset(
            coords=dict(
                time=pixel_times,
                hist_mid=mids,
                bins=bins,
            ),
            data_vars={
                f'{self.var}_MCS': (('time', 'hist_mid'), hists.copy()),
                f'{self.var}_conv': (('time', 'hist_mid'), hists.copy()),
                f'{self.var}_env': (('time', 'hist_mid'), hists.copy()),
            }
        )
        regridder = None
        # Looping over subset of times.
        for i, time in enumerate(pixel_times):
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
                dsout[f'{self.var}_MCS'][i].values[:] = None
                dsout[f'{self.var}_conv'][i].values[:] = None
                dsout[f'{self.var}_env'][i].values[:] = None
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
            e5data = (xr.open_mfdataset([e5inputs[t] for t in [e5time1, e5time2]])[self.var]
                      .mean(dim='time').sel(latitude=slice(60, -60)).load())

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
            dsout[f'{self.var}_MCS'][i] = hist(e5data.values[mcs_mask])[0]
            dsout[f'{self.var}_conv'][i] = hist(e5data.values[~mcs_mask & e5conv_mask])[0]
            dsout[f'{self.var}_env'][i] = hist(e5data.values[~mcs_mask & ~e5conv_mask])[0]
        dsout.to_netcdf(self.outputs['hist'])


class OldCombineConditionalERA5Hist(TaskRule):
    enabled = False
    @staticmethod
    def rule_inputs(year, var):
        inputs = {f'hist_{m}': fmtp(ConditionalERA5Hist.rule_outputs['hist'],
                                    year=year, month=m, var=var)
                  for m in months}
        return inputs

    rule_outputs = {'hist': (PATHS['outdir'] / 'conditional_era5_histograms' /
                             '{year}' /
                             'yearly_{var}_hist_{year}.nc')}

    var_matrix = {'year': years, 'var': ['cape', 'tcwv']}

    def rule_run(self):
        ds = xr.open_mfdataset(self.inputs.values())
        ds.to_netcdf(self.outputs['hist'])


class ConditionalERA5ShearHist(TaskRule):
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


class CombineConditionalERA5ShearHist(TaskRule):
    enabled = False
    @staticmethod
    def rule_inputs(year):
        inputs = {f'hist_{m}': fmtp(ConditionalERA5ShearHist.rule_outputs['hist'],
                                    year=year, month=m)
                  for m in months}
        return inputs

    rule_outputs = {'hist': (PATHS['outdir'] / 'conditional_era5_histograms' /
                             '{year}' /
                             'yearly_shear_hist_{year}.nc')}

    var_matrix = {'year': years}

    def rule_run(self):
        ds = xr.open_mfdataset(self.inputs.values())
        ds.to_netcdf(self.outputs['hist'])


class GenERA5PixelData(TaskRule):
    """Generate ERA5 pixel data by regridding native MCS dataset masks/cloudnumbers

    The MCS dataset pixel-level data has a field cloudnumber, which maps onto the
    corresponding cloudnumber field in the tracks data. Here, for each time (on
    the half-hour, to match MCS dataset), I regrid this data to the ERA5 grid.
    This is non-trivial, because they use different longitude endpoints. The regridding/
    coarsening is done by xesmf, which handled this and periodic longitude.

    The scheme for this is:
    * load pixel to ERA5 regridder.
    * for each hour for which there is data, do regridding.
    * load pixel-level cloudnumber/Tb.
    * for each cloudnumber, regrid the mask for that cloudnumber, keeping all values
      greater than 0.5.
    * combine into one integer field.
    * regrid a "core" indicator, where Tb < 225 (to match Feng et al. 2021 core definition).
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

    var_matrix = {('year', 'month', 'day'): date_keys}

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


class ConditionalERA5Hist(TaskRule):
    @staticmethod
    def rule_inputs(year, month, day, var):
        start = dt.datetime(year, month, day)
        # Note there are 25 of these so I can get ERA5 data on the hour either side
        # of MCS dataset data (on the half hour).
        e5times = pd.date_range(start, start + dt.timedelta(days=1), freq='H')

        e5inputs = {f'era5_{t}': (PATHS['era5dir'] /
                                  f'data/oper/an_sfc/{year}/{month:02d}/{day:02d}' /
                                  (f'ecmwf-era5_oper_an_sfc_{year}{month:02d}{day:02d}'
                                   f'{t.hour:02d}00.{var}.nc'))
                    for t in e5times}

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

        inputs = {**e5inputs, **e5pixel_inputs}
        return inputs

    rule_outputs = {'hist': (PATHS['outdir'] / 'conditional_era5_histograms' /
                             '{year}' /
                             'daily_{var}_hist_{year}_{month:02d}_{day:02d}.nc')}

    var_matrix = {('year', 'month', 'day'): date_keys, 'var': ['cape', 'tcwv']}

    def rule_run(self):
        tracks = McsTracks.open(self.inputs['tracks'], None)

        start = dt.datetime(self.year, self.month, self.day)
        # Note there are 25 of these so I can get ERA5 data on the hour either side
        # of MCS dataset data (on the half hour).
        e5times = pd.date_range(start, start + dt.timedelta(days=1), freq='H')

        e5inputs = {t: self.inputs[f'era5_{t}']
                    for t in e5times}

        e5pixel = xr.load_dataset(self.inputs['e5pixel'])

        if self.var == 'cape':
            bins = np.linspace(0, 5000, 501)
        elif self.var == 'tcwv':
            bins = np.linspace(0, 100, 101)
        mids = (bins[1:] + bins[:-1]) / 2

        hists = np.zeros((len(e5pixel.time), mids.size))

        # Make a dataset to hold all the histogram data.
        dsout = xr.Dataset(
            coords=dict(
                time=e5pixel.time,
                hist_mid=mids,
                bins=bins,
            ),
            data_vars={
                f'{self.var}_MCS_shield': (('time', 'hist_mid'), hists.copy()),
                f'{self.var}_MCS_core': (('time', 'hist_mid'), hists.copy()),
                f'{self.var}_cloud_shield': (('time', 'hist_mid'), hists.copy()),
                f'{self.var}_cloud_core': (('time', 'hist_mid'), hists.copy()),
                f'{self.var}_env': (('time', 'hist_mid'), hists.copy()),
            }
        )

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
                self.logger.warn(f'No times matched in tracks DB for {pdtime}')
                dsout[f'{self.var}_MCS_shield'][i].values[:] = None
                dsout[f'{self.var}_MCS_core'][i].values[:] = None
                dsout[f'{self.var}_cloud_shield'][i].values[:] = None
                dsout[f'{self.var}_cloud_core'][i].values[:] = None
                dsout[f'{self.var}_env'][i].values[:] = None
                continue
            # Each cloudnumber can be used to link to the corresponding
            # cloud in the pixel data.
            cns = ts.dstracks.cloudnumber.values[tmask]
            # Nicer to have sorted values.
            cns.sort()

            # Load the e5 data and interp in time to pixel time.
            e5time1 = time - dt.timedelta(minutes=30)
            e5time2 = time + dt.timedelta(minutes=30)
            e5data = (xr.open_mfdataset([e5inputs[t] for t in [e5time1, e5time2]])[self.var]
                      .mean(dim='time').sel(latitude=slice(60, -60)).load())

            core_mask = e5pixel.tb[i].values < 225
            mcs_shield_mask = e5pixel.cloudnumber[i].isin(cns).values
            cloud_shield_mask = e5pixel.cloudnumber[i].values > 0 & ~mcs_shield_mask
            mcs_core_mask = mcs_shield_mask & core_mask
            cloud_core_mask = cloud_shield_mask & core_mask
            env_mask = ~mcs_shield_mask & ~cloud_shield_mask

            # Calc hists.
            hist = partial(np.histogram, bins=bins)
            dsout[f'{self.var}_MCS_shield'][i] = hist(e5data.values[mcs_shield_mask])[0]
            dsout[f'{self.var}_MCS_core'][i] = hist(e5data.values[mcs_core_mask])[0]
            dsout[f'{self.var}_cloud_shield'][i] = hist(e5data.values[cloud_shield_mask])[0]
            dsout[f'{self.var}_cloud_core'][i] = hist(e5data.values[cloud_core_mask])[0]
            dsout[f'{self.var}_env'][i] = hist(e5data.values[env_mask])[0]
        dsout.to_netcdf(self.outputs['hist'])


class CombineConditionalERA5Hist(TaskRule):
    enabled = False
    @staticmethod
    def rule_inputs(year, var):
        dates = pd.date_range(f'{year}-01-01', f'{year}-12-31')
        inputs = {f'hist_{d}': fmtp(ConditionalERA5Hist.rule_outputs['hist'],
                                    year=d.year, month=d.month, day=d.day, var=var)
                  for d in dates}
        return inputs

    rule_outputs = {'hist': (PATHS['outdir'] / 'conditional_era5_histograms' /
                             '{year}' /
                             'yearly_{var}_hist_{year}.nc')}

    var_matrix = {'year': years, 'var': ['cape', 'tcwv']}

    def rule_run(self):
        ds = xr.open_mfdataset(self.inputs.values())
        ds.to_netcdf(self.outputs['hist'])

