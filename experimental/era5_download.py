from pathlib import Path

import cdsapi

from remake import Remake, TaskRule

from mcs_prime import PATHS, McsTracks, McsTrack, PixelData

c = cdsapi.Client()

era5_download = Remake()

YEARS = [2019]
MONTHS = [1]


class Era5DownloadGlobalSingleLevelVars(TaskRule):
    rule_inputs = {}
    rule_outputs = {f'out': PATHS['datadir'] / '/ERA5/GlobalSingleLevel/era5_global_single_level_{year}_{month:02d}.nc'}
    var_matrix = {
        'year': YEARS,
        'month': MONTHS,
    }

    def rule_run(self):
        output_path = self.outputs['out']

        year = self.year
        month = f'{self.month:02d}'

        c = cdsapi.Client()

        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': [
                    'cloud_base_height',
                    'convective_available_potential_energy',
                    'convective_inhibition',
                    'total_column_water',
                ],
                'year': year,
                'month': month,
                # 'day': [
                #     '01', '02', '03',
                #     '04', '05', '06',
                #     '07', '08', '09',
                #     '10', '11', '12',
                #     '13', '14', '15',
                #     '16', '17', '18',
                #     '19', '20', '21',
                #     '22', '23', '24',
                #     '25', '26', '27',
                #     '28', '29', '30',
                #     '31',
                #     ],
                'day': [
                    '01',
                ],
                'time': [
                    '00:00',
                    '01:00',
                    '02:00',
                    '03:00',
                    '04:00',
                    '05:00',
                    '06:00',
                    '07:00',
                    '08:00',
                    '09:00',
                    '10:00',
                    '11:00',
                    '12:00',
                    '13:00',
                    '14:00',
                    '15:00',
                    '16:00',
                    '17:00',
                    '18:00',
                    '19:00',
                    '20:00',
                    '21:00',
                    '22:00',
                    '23:00',
                ],
            },
            str(output_path),
        )
