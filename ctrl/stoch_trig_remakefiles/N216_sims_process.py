import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe

from remake import Remake, TaskRule

import mcs_prime.mcs_prime_config_util as cu

DATADIR = cu.PATHS['datadir']
SIMDIR = DATADIR / 'UM_sims'
N_ENS_MEM = 10

EXPT_SIM = {
    'ctrl': 'u-di727',
    'vanillaMCSP': 'u-di728',
    'stochMCSP': 'u-dg135',
}


IMERG_FINAL_30MIN_DIR = DATADIR / 'GPM_IMERG_final/30min'
# These times exactly match the UM sims.
# UM_TIMES = pd.date_range('2020-07-01 04:00', '2020-07-11 03:00', freq='H')
UM_TIMES = (('2020-07-01 04:00', '2020-07-11 03:00'), {'freq': 'H'}) # args, kwargs for date_range.

slurm_config = {'account': 'short4hr', 'queue': 'short-serial-4hr', 'mem': 64000}
rmk = Remake(config=dict(slurm=slurm_config, content_checks=False))


class N216ExtractCombinePrecip(TaskRule):
    """Extract and combine precipitation at all times and for all EMs.
    """
    @staticmethod
    def rule_inputs(expt):
        suite = EXPT_SIM[expt]
        inputs = {
            f'pa_em{em_idx}_{h:03d}': SIMDIR / f'{suite}/share/cycle/20200701T0000Z/engl/um/em{em_idx}/englaa_pa{h:03d}.iris.nc'
            for em_idx in range(N_ENS_MEM)
            for h in range(0, 217, 24)
        }
        return inputs

    @staticmethod
    def rule_outputs(expt):
        suite = EXPT_SIM[expt]
        outputs = {
            f'pflux': SIMDIR / f'{suite}/processed/{expt}/engla_pa.precip.nc'
        }
        return outputs


    var_matrix = {
        'expt': list(EXPT_SIM.keys())
    }

    def rule_run(self):
        def load_em_pflux(inputs, ens_idx):
            keep_coords = ['time', 'latitude', 'longitude']
            time_pfluxes = []

            for h in range(0, 217, 24):
                self.logger.debug(f'  Opening time {h}')

                em_path = inputs[f'pa_em{ens_idx}_{h:03d}']
                dsa = xr.open_dataset(em_path)
                # What's the difference between the two fluxes?
                # dsa.precipitation_flux has no time mean (i.e. it's instantaneous)
                # dsa.precipitation_flux_0 has 1-hr time mean.
                # I think it's better to use instantaneous to e.g. compare with IMERG.
                pflux = dsa.precipitation_flux
                # Why does time 0 have different names for all variables?
                if h == 0:
                    pflux = pflux.rename(time_0='time')
                # Drop all variables that are not needed. This means concat will work.
                # (This drops all other coords with _0 suffix.)
                coord_names = [c.name for c in list(pflux.coords.values())]
                drop_coords = sorted(set(coord_names) - set(keep_coords))
                pflux = pflux.drop_vars(drop_coords)

                time_pfluxes.append(pflux)

            pflux = xr.concat(time_pfluxes, dim='time')
            return pflux.load()

        em_pfluxes = []
        for ens_idx in range(N_ENS_MEM):
            self.logger.info(f'Loading EM {ens_idx}')
            em_pfluxes.append(load_em_pflux(self.inputs, ens_idx))
        pflux = xr.concat(em_pfluxes, dim=pd.Index(range(N_ENS_MEM), name='ens_mem'))

        pflux.attrs['UM simulation'] = EXPT_SIM[self.expt]
        pflux.attrs['MCS:PRIME expt'] = self.expt

        cu.to_netcdf_tmp_then_copy(pflux, self.outputs['pflux'])


class RegridImergToN216(TaskRule):
    """Regrid IMERG to the same grid as N216 simulations.
    """
    @staticmethod
    def rule_inputs(times_args):
        # Just load IMERG on the hour "S??0000".
        args, kwargs = times_args
        times = pd.date_range(*args, **kwargs)
        def h_to_m(h):
            return h * 60
        inputs = {
            f'imerg_{t}': (
                IMERG_FINAL_30MIN_DIR /
                f'{t.year}/{t.month:02d}/{t.day:02d}/' /
                f'3B-HHR.MS.MRG.3IMERG.{t.year}{t.month:02d}{t.day:02d}-S{t.hour:02d}0000-E{t.hour:02d}2959.{h_to_m(t.hour):04d}.V07B.HDF5.nc4'
            )
            for t in times
        }
        inputs['pflux'] = N216ExtractCombinePrecip.rule_outputs('stochMCSP')['pflux']

        return inputs

    @staticmethod
    def rule_outputs(times_args):
        args, kwargs = times_args
        times = pd.date_range(*args, **kwargs)
        d0 = str(times[0]).replace(' ', '_')
        dlast = str(times[-1]).replace(' ', '_')
        outputs = {
            'output': (
                cu.PATHS['outdir']
                / f'imerg_processed/N216grid/{d0}-{dlast}/3B-HHR.MS.MRG.3IMERG.{d0}-{dlast}.hourly.V07B.nc'
            )
        }
        return outputs

    var_matrix = {
        'times_args': [UM_TIMES],
    }

    def rule_run(self):
        pflux = xr.open_dataarray(self.inputs['pflux'])
        imerg = xr.open_mfdataset([v for k, v in self.inputs.items() if k.startswith('imerg')])
        print(pflux)
        print(imerg)
        # TODO: conservative method.
        regridder = xe.Regridder(imerg.precipitation, pflux, method='bilinear')
        imerg_N216 = regridder(imerg.precipitation)
        cu.to_netcdf_tmp_then_copy(imerg_N216, self.outputs['output'])

