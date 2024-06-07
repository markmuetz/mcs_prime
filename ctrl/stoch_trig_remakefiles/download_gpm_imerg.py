"""remakefile for downloading data -- one task per year.

Downloads GMP IMERG (late/final) daily data for Asia.

Will not download same file twice"""
import datetime as dt
from pathlib import Path
from random import randint
from time import sleep
from timeit import default_timer as timer

import pandas as pd
import requests

from remake import Remake, TaskRule

import mcs_prime.mcs_prime_config_util as cu

DATADIR = cu.PATHS['datadir']
IMERG_FINAL_30MIN_DIR = DATADIR / 'GPM_IMERG_final/30min'

# Asia.
# FINAL_30MIN_URL_TPL = 'https://gpm1.gesdisc.eosdis.nasa.gov/opendap/GPM_L3/GPM_3IMERGHH.06/{year}/{doy}/{filename}?precipitationCal[0:0][2350:3329][889:1479],time,lon[2350:3329],lat[889:1479]'
# Global. Reverse engineered. Not sure how I originally generated this URL!
# FINAL_30MIN_URL_TPL = 'https://gpm1.gesdisc.eosdis.nasa.gov/opendap/GPM_L3/GPM_3IMERGHH.07/{year}/{doy}/{filename}?precipitationCal[0:0][0:3599][0:1799],time,lon[0:3599],lat[0:1799]'
# FINAL_30MIN_FILENAME_TPL = '3B-HHR.MS.MRG.3IMERG.{datestr}-S{start_time}-E{end_time}.{minutes}.V07B.HDF5.nc4'
# https://gpm1.gesdisc.eosdis.nasa.gov/opendap/GPM_L3/GPM_3IMERGHH.07/2023/365/3B-HHR.MS.MRG.3IMERG.20231231-S000000-E002959.0000.V07B.HDF5.nc4?precipitation,time,lon,lat
# Note these are the V7 (current as of 2024-06-04 versions)
FINAL_30MIN_URL_TPL = 'https://gpm1.gesdisc.eosdis.nasa.gov/opendap/GPM_L3/GPM_3IMERGHH.07/{year}/{doy}/{filename}?precipitation,time,lon,lat'
FINAL_30MIN_FILENAME_TPL = '3B-HHR.MS.MRG.3IMERG.{datestr}-S{start_time}-E{end_time}.{minutes}.V07B.HDF5.nc4'

YEARS = range(2020, 2021)
MONTHS = [6]

start_date = '2020-07-01'
end_date = '2020-07-03'

# Create the date range
date_range = pd.date_range(start=start_date, end=end_date)
dates = list(date_range)

downloader = Remake()


class GpmDatetime:
    """Useful fmt_methods added to datetime class"""
    @staticmethod
    def fmt_date(date):
        return date.strftime('%Y%m%d')

    @staticmethod
    def fmt_year(date):
        return date.strftime('%Y')

    @staticmethod
    def fmt_month(date):
        return date.strftime('%m')

    @staticmethod
    def fmt_doy(date):
        return date.strftime('%j')

    @staticmethod
    def fmt_time(date):
        return date.strftime('%H%M%S')

    @staticmethod
    def fmt_minutes(date):
        minutes = date.hour * 60 + date.minute
        return f'{minutes:04d}'


def get_from_gpm(url, filename, num_retries=6):
    """Retrive from NASA GPM IMERG data repository

    note: $HOME/.netrc must be set!
    https://disc.gsfc.nasa.gov/data-access#python-requests
    """
    retries = num_retries
    while True:
        try:
            result = requests.get(url)
            break
        except Exception as e:
            print('Connection error')
            print(e)
            if not retries:
                raise
            retries -= 1
            sleep((num_retries - retries) * 10 + randint(0, 20))

    try:
        result.raise_for_status()
        with open(filename,'wb') as f:
            f.write(result.content)
    except:
        print('requests.get() returned an error code ' + str(result.status_code))
        raise


def gen_dates_urls_filenames(filename_tpl, url_tpl, start_date, end_date):
    """Generates dates, urls and filenames in 30min intervals"""
    dates = pd.date_range(start_date, end_date, freq='30min')
    for date in dates:
        curr_date = date.to_pydatetime()
        next_date = curr_date + dt.timedelta(minutes=30)
        filename = filename_tpl.format(datestr=GpmDatetime.fmt_date(curr_date),
                                       start_time=GpmDatetime.fmt_time(curr_date),
                                       end_time=GpmDatetime.fmt_time(next_date - dt.timedelta(seconds=1)),
                                       minutes=GpmDatetime.fmt_minutes(curr_date))
        url = url_tpl.format(year=GpmDatetime.fmt_year(curr_date),
                             doy=GpmDatetime.fmt_doy(curr_date),
                             filename=filename)
        yield curr_date, url, filename
        curr_date = next_date




class GpmImerg30MinDownload(TaskRule):
    rule_inputs = {}
    rule_outputs = {'output_filenames': IMERG_FINAL_30MIN_DIR / '{date.year}' / 'download.{date}.done'}
    var_matrix = {'date': dates}

    def rule_run(self):

        start_date = self.date
        end_date = self.date + pd.Timedelta(days=1) - pd.Timedelta(minutes=30)

        outputs = {}
        filename_tpl = FINAL_30MIN_FILENAME_TPL
        url_tpl = FINAL_30MIN_URL_TPL
        dates_urls_filenames = list(gen_dates_urls_filenames(filename_tpl,
                                                             url_tpl,
                                                             start_date,
                                                             end_date))
        all_filenames = []
        for i, (date, url, filename) in enumerate(dates_urls_filenames):
            output_filename = self.outputs['output_filenames'].parent / f'{date.month:02d}/{date.day:02d}' / filename
            output_filename.parent.mkdir(exist_ok=True, parents=True)
            # output_filename = Path(IMERG_FINAL_DIR / date.fmt_year() / filename)
            if not output_filename.exists():
                outputs[url] = output_filename
            all_filenames.append(str(output_filename))

        for url, output_filename in outputs.items():
            print(url, output_filename)
            start = timer()
            tmp_filename = Path(output_filename.parent / ('.tmp.gpm_download.' + output_filename.name))
            tmp_filename.parent.mkdir(exist_ok=True)
            get_from_gpm(url, tmp_filename)
            assert tmp_filename.exists()
            tmp_filename.rename(output_filename)
            print(f'-> downloaded in {(timer() - start):.2f}s')
        else:
            print(f'No files to download for {self.date}')

        self.outputs['output_filenames'].write_text('\n'.join(all_filenames) + '\n')

