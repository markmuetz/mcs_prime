from collections import defaultdict

import numpy as np
import pandas as pd
import xarray as xr

from remake import Remake, TaskRule
from remake.util import tmp_to_actual_path

import mcs_prime.mcs_prime_config_util as cu
from mcs_prime.era5_calc import ERA5Calc

slurm_config = {'account': 'short4hr', 'queue': 'short-serial-4hr', 'mem': 16000}
rmk = Remake(config=dict(slurm=slurm_config, content_checks=False))

class CheckModelLevelToPressure(TaskRule):
    @staticmethod
    def rule_inputs(date):
        dates = [date + pd.Timedelta(hours=h) for h in range(24)]
        inputs = {f'lnsp_{d}': cu.era5_ml_fmtp('lnsp', d.year, d.month, d.day, d.hour)
                  for d in dates}
        return inputs

    @staticmethod
    def rule_outputs(date):
        filename = f'check_model_level_to_pressure_{date.year}{date.month:02d}{date.day:02d}.hdf'
        return {'output': cu.PATHS['outdir'] / 'check_model_level_to_pressure' / filename}

    var_matrix = {'date': [pd.Timestamp(2020, m, 1) for m in range(1, 13)]}

    def rule_run(self):
        u_path = cu.era5_ml_fmtp('u', self.date.year, self.date.month, self.date.day, 0)
        u = xr.open_dataarray(u_path).isel(time=0).sel(latitude=slice(60, -60))
        lsm = xr.open_dataarray(cu.PATH_ERA5_LAND_SEA_MASK).isel(time=0).sel(latitude=slice(60, -60))

        model_levels = cu.PATH_ERA5_MODEL_LEVELS
        e5calc = ERA5Calc(model_levels)
        data = []
        for h in range(24):
            date = self.date + pd.Timedelta(hours=h)
            lnsp_path = self.inputs[f'lnsp_{date}']
            lnsp = xr.open_dataarray(lnsp_path).isel(time=0).sel(latitude=slice(60, -60))
            p = e5calc.calc_pressure(lnsp.values)
            da_p = xr.DataArray(
                p / 100, # convert from Pa to hPa
                dims=['level', 'latitude', 'longitude'],
                coords=dict(
                    level=u.level,
                    latitude=u.latitude,
                    longitude=u.longitude,
                ),
                attrs=dict(
                    units='hPa',
                    standard='air_pressure',
                )
            )
            print(date, h)
            for level in [136, 111, 100, 90]:
                plev = da_p.sel(level=level)
                all_per = np.percentile(plev.values.flatten(), [1, 10, 25, 50, 75, 90, 99])
                sea_per = np.percentile(plev.values[lsm.values == 0], [1, 10, 25, 50, 75, 90, 99])
                land_per = np.percentile(plev.values[lsm.values == 1], [1, 10, 25, 50, 75, 90, 99])
                data.append([date, 'all', level] + all_per.tolist())
                data.append([date, 'sea', level] + sea_per.tolist())
                data.append([date, 'land', level] + land_per.tolist())

        df = pd.DataFrame(data, columns=['time', 'reg', 'level', 'p1', 'p10', 'p25', 'p50', 'p75', 'p90', 'p99'])
        df.to_hdf(self.outputs['output'], 'check_model_level_to_pressure')


class ModelLevelToPressureInfo(TaskRule):
    @staticmethod
    def rule_inputs():
        dates = [pd.Timestamp(2020, m, 1) for m in range(1, 13)]
        inputs = {f'{d}': CheckModelLevelToPressure.rule_outputs(d)['output']
                  for d in dates}
        return inputs

    rule_outputs = {'pressure_info': (cu.PATHS['figdir'] /
                                      'check_model_level_to_pressure' /
                                      f'check_model_level_to_pressure_2020.csv')}

    def rule_run(self):
        df = pd.concat([pd.read_hdf(p) for p in self.inputs.values()])
        data = defaultdict(list)
        for level in [136, 111, 100, 90]:
            data['level'].append(level)
            for region in ['all', 'land', 'sea']:
                data[region].append(df[(df.reg == region) & (df.level == level)].p50.mean())
        print(data)
        data['refereced as'] = ['surface', 800, 600, 400]
        dfout = pd.DataFrame(data)
        print(dfout)
        print(self.outputs['pressure_info'])
        dfout.to_csv(self.outputs['pressure_info'])
