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

slurm_config = {'account': 'short4hr', 'queue': 'short-serial-4hr', 'mem': 16000}
era5_histograms = Remake(config=dict(slurm=slurm_config, content_checks=False))

years = [2020]
months = range(1, 13)


class ERA5Hist(TaskRule):
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

    var_matrix = {'year': years, 'month': months, 'var': ['cape']}

    def rule_run(self):
        month_start_day = dt.datetime(self.year, self.month, 1).timetuple().tm_yday
        month_files = list(self.inputs.values())
        bins = np.linspace(0, 5000, 501)
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
    @staticmethod
    def rule_inputs(year, var):
        inputs = {f'hist_{m}': fmtp(ERA5Hist.rule_outputs['hist'], year=year, month=m, var=var)
                  for m in months}
        return inputs

    rule_outputs = {'hist': (PATHS['outdir'] / 'era5_histograms' /
                             '{year}' /
                             'yearly_{var}_hist_{year}.nc')}

    var_matrix = {'year': years, 'var': ['cape']}

    def rule_run(self):
        ds = xr.open_mfdataset(self.inputs.values())
        ds.to_netcdf(self.outputs['hist'])


class GenRegridder(TaskRule):
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


class ConditionalERA5Hist(TaskRule):
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

    var_matrix = {'year': years, 'month': months, 'var': ['cape']}

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
        bins = np.linspace(0, 5000, 501)
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

            if True:
                # This is a lot faster! If you don't need the individual cloudnumber info
                # you can use this to partition the 2D grid into MCS/non-MCS.
                e5mcs_mask = regridder(frame.dspixel.cloudnumber.isin(cns).astype(float)) > 0.5
                mcs_mask = e5mcs_mask[0]
            else:
                # Quite slow.
                # Perform the regrid for each cloudnumber.
                mask_regridded = []
                for i in cns:
                    print(i)
                    mask_regridded.append(regridder((frame.dspixel.cloudnumber == i).astype(float)))

                da_mask_regridded = xr.concat(mask_regridded, pd.Index(cns, name='cn'))
                cn_e5 = ((da_mask_regridded > 0.5).astype(int) * da_mask_regridded.cn).sum(dim='cn')

                # Make masks that let me select out different regions.
                mcs_mask = cn_e5.values[0] > 0.5
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


class CombineConditionalERA5Hist(TaskRule):
    @staticmethod
    def rule_inputs(year, var):
        inputs = {f'hist_{m}': fmtp(ConditionalERA5Hist.rule_outputs['hist'], year=year, month=m, var=var)
                  for m in months}
        return inputs

    rule_outputs = {'hist': (PATHS['outdir'] / 'conditional_era5_histograms' /
                             '{year}' /
                             'yearly_{var}_hist_{year}.nc')}

    var_matrix = {'year': years, 'var': ['cape']}

    def rule_run(self):
        ds = xr.open_mfdataset(self.inputs.values())
        ds.to_netcdf(self.outputs['hist'])

