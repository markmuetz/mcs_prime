import cdsapi
import pandas as pd

from remake import Remake, TaskRule
from remake.util import tmp_to_actual_path

import mcs_prime.mcs_prime_config_util as cu

c = cdsapi.Client()

slurm_config = {'queue': 'short-serial', 'mem': 64000, 'max_runtime': '20:00:00'}
era5_download = Remake(config=dict(slurm=slurm_config, content_checks=False))

DATADIR = cu.PATHS['datadir']
YEARS_MONTHS = [(y, m) for y in cu.YEARS for m in cu.MONTHS]

# E.g. (generated from https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form):
"""
c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': 'convective_inhibition',
        'year': '2020',
        'month': '01',
        'day': '01',
        'time': '00:00',
    },
    'download.nc')
"""

class Era5DownloadSfcYear(TaskRule):
    """Download additional ERA5 Surface data

    Output is different from BADC ERA5 data:
    Only one file saved per year to cut down on (slow) queue times on Copernicus API.
    Some variables are not present that would be useful, e.g. CIN.
    """
    rule_inputs = {}
    rule_outputs = {'output': (DATADIR / 'ecmwf-era5/data/oper/an_sfc/'
                               'ecmwf-era5_oper_an_sfc_'
                               '{year}.{variable}.nc')}

    var_matrix = {
        'year': cu.YEARS,
        'variable': ['cin']
    }
    req_var_names = {
        'cin': 'convective_inhibition',
    }

    def rule_run(self):
        var_name = self.req_var_names[self.variable]

        msg = f'Download {self.variable} for {self.year}'
        print(msg)
        print('=' * len(msg))

        output_path = self.outputs['output']
        # Create a tmp path, save download to this, then mv to actual output_path.
        # Ensures that output_path will *only* be present if download has completed.
        request_dict = {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': var_name,
            'year': f'{self.year}',
            'month': [f'{m:02d}' for m in range(1, 13)],
            'day': [f'{d:02d}' for d in range(1, 32)],
            'time': [f'{h:02d}:00' for h in range(24)],
        }
        c.retrieve(
            'reanalysis-era5-single-levels',
            request_dict,
            str(output_path)
        )


class Era5DownloadSfc(TaskRule):
    enabled = False
    """Download additional ERA5 Surface data

    Output in exactly the same filename format at BADC ERA5 data:
    e.g. /badc/ecmwf-era5/data/oper/an_sfc/2020/01/01/ecmwf-era5_oper_an_sfc_202001010000.cape.nc
    Some variables are not present that would be useful, e.g. CIN.
    """
    @staticmethod
    def rule_inputs(year, month, variable):
        if year != cu.YEARS[0]:
            inputs = {'prev_year_month': (DATADIR / 'ecmwf-era5/data/oper/an_sfc/'
                                          f'{year - 1}_{month:02d}_{variable}.done')}
        else:
            inputs = {}
        return inputs

    @staticmethod
    def rule_outputs(year, month, variable):
        starttime = pd.Timestamp(year, month, 1, 0, 0)
        endtime = starttime + pd.DateOffset(months=1) - pd.Timedelta(hours=1)
        datetimes = pd.date_range(starttime, endtime, freq='H')

        outputs = {f'{t}': (DATADIR / 'ecmwf-era5/data/oper/an_sfc/'
                            f'{t.year}/{t.month:02d}/{t.day:02d}/'
                            f'ecmwf-era5_oper_an_sfc_'
                            f'{t.year}{t.month:02d}{t.day:02d}{t.hour:02d}{t.minute:02d}.{variable}.nc')
                            for t in datetimes}
        outputs['year_month'] = (DATADIR / 'ecmwf-era5/data/oper/an_sfc/'
                                 f'{year}_{month:02d}_{variable}.done')
        return outputs

    var_matrix = {
        ('year', 'month'): YEARS_MONTHS,
        'variable': ['cin']
    }
    req_var_names = {
        'cin': 'convective_inhibition',
    }

    def rule_run(self):
        starttime = pd.Timestamp(self.year, self.month, 1, 0, 0)
        endtime = starttime + pd.DateOffset(months=1) - pd.Timedelta(hours=1)
        datetimes = pd.date_range(starttime, endtime, freq='H')
        var_name = self.req_var_names[self.variable]

        for time in datetimes:
            msg = f'Download {self.variable} for {time}'
            print(msg)
            print('=' * len(msg))

            output_path = self.outputs[f'{time}']
            # If download has completed, skip file.
            if output_path.exists() or tmp_to_actual_path(output_path).exists():
                print(f'Skipping because {output_path} has been written')
                continue
            # Create a tmp path, save download to this, then mv to actual output_path.
            # Ensures that output_path will *only* be present if download has completed.
            tmp_path = output_path.parent / ('.cdsapi_tmp' + output_path.name)
            request_dict = {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': var_name,
                'year': f'{time.year}',
                'month': f'{time.month:02d}',
                'day': f'{time.day:02d}',
                'time': f'{time.hour:02d}:{time.minute:02d}',
            }
            c.retrieve(
                'reanalysis-era5-single-levels',
                request_dict,
                str(tmp_path)
            )
            tmp_path.rename(output_path)
        self.outputs['year_month'].write_text('done')
