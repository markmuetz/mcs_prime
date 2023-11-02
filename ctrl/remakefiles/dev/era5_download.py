import cdsapi
import pandas as pd

from remake import Remake, TaskRule

import mcs_prime.mcs_prime_config_util as cu

c = cdsapi.Client()

era5_download = Remake(config=dict(content_checks=False))

DATADIR = cu.PATHS['datadir']
YEARS = range(2020, 2021)
MONTHS = range(1, 13)
YEARS_MONTHS = [(y, m) for y in YEARS for m in MONTHS]

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

class Era5DownloadSfc(TaskRule):
    """Download additional ERA5 Surface data

    Output in exactly the same filename format at BADC ERA5 data:
    e.g. /badc/ecmwf-era5/data/oper/an_sfc/2020/01/01/ecmwf-era5_oper_an_sfc_202001010000.cape.nc
    Some variables are not present that would be useful, e.g. CIN.
    """
    rule_inputs = {}
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
                str(output_path)
            )
