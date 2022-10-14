import calendar
import datetime as dt
from functools import partial

import numpy as np
import xarray as xr

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
