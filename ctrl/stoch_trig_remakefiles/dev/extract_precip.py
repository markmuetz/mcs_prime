import xarray as xr

from remake import Remake, TaskRule

import mcs_prime.mcs_prime_config_util as cu

DATADIR = cu.PATHS['datadir']
OUTDIR = cu.PATHS['outdir']
SIMDIR = DATADIR / 'UM_sims'

slurm_config = {'account': 'short4hr', 'queue': 'short-serial-4hr', 'mem': 64000}
rmk = Remake(config=dict(slurm=slurm_config, content_checks=False))

SUITES = ['u-dg040', 'u-dg041', 'u-dg042']


def get_daily_mean_precip_flux(nc_path):
    df = xr.open_dataset(nc_path)
    tindex = xr.CFTimeIndex(df.precipitation_flux.time_1.values)
    # precipitation_flux is included twice in this stream with different time avgs.
    # Extract the daily mean values.
    tfilter = (tindex.hour == 12) & (tindex.minute == 0)
    pflux = df.precipitation_flux.isel(dim0=tfilter)
    return pflux


class ExtractPrecip(TaskRule):
    """
    """

    @staticmethod
    def rule_inputs(suite):
        nc_paths = sorted((SIMDIR / suite).glob(f'{suite[2:]}a.pd*.nc'))
        return {p: p for p in nc_paths}

    rule_outputs = {'out_nc': OUTDIR / 'um_sims_analysis/dev/extract_precip/{suite}_extracted_precip.nc'}

    var_matrix = {
        'suite': SUITES,
    }

    depends_on = [get_daily_mean_precip_flux]

    def rule_run(self):
        print(f'Extract precip from {self.suite}')
        nc_paths = self.inputs.values()
        pflux = xr.concat([get_daily_mean_precip_flux(p) for p in nc_paths], dim='dim0')
        cu.to_netcdf_tmp_then_copy(pflux, self.outputs['out_nc'])

