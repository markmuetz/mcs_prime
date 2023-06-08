from itertools import product

import numpy as np
import pandas as pd
import xarray as xr


from remake import Remake, TaskRule
from remake.util import format_path as fmtp

from mcs_prime import McsTracks

from era5_config_utils import *


TODOS = '''
* Make sure filenames are consistent
* Make sure variables names are sensible/consistent
* Docstrings for all fns, classes
* Validate all data
* Shear levels as coord?
* Consistent attrs for all created .nc files
* Units on data vars etc.
'''
print(TODOS)

slurm_config = {'queue': 'short-serial', 'mem': 64000, 'max_runtime': '20:00:00'}
# slurm_config = {'account': 'short4hr', 'queue': 'short-serial-4hr', 'mem': 64000}
era5_histograms = Remake(config=dict(slurm=slurm_config, content_checks=False, no_check_input_exist=True))

def conditional_inputs(year, month, precursor_time=0):
    start = pd.Timestamp(year, month, 1)
    end = start + pd.DateOffset(months=1)
    daily_pixel_times = pd.date_range(start, end - pd.Timedelta(days=1), freq='D')
    e5times = (gen_era5_times_for_month(year, month, include_precursor_offset=False) -
               pd.Timedelta(hours=precursor_time))

    # OK, this is a little complicated.
    # The logic is the same as above in GenPixelDataOnERA5Grid.
    # I only want to add the output of GenPixelDataOnERA5Grid, if the original MCS pixel data
    # exists. But I now need to do for one month.
    e5pixel_inputs = {}
    for daily_pixel_time in daily_pixel_times:
        pixel_times, pixel_inputs = pixel_inputs_cache.all_pixel_inputs[(daily_pixel_time.year,
                                                                         daily_pixel_time.month,
                                                                         daily_pixel_time.day)]
        pixel_output_paths = [fmtp(FMT_PATH_PIXEL_ON_ERA5, year=t.year, month=t.month, day=t.day, hour=t.hour)
                              for t in pixel_times]
        e5pixel_inputs.update({f'e5pixel_{t}': p
                               for t, p in zip(pixel_times, pixel_output_paths)})
    e5pixel_inputs['tracks'] = fmt_mcs_stats_path(year)

    e5inputs = {f'era5_{t}_{var}': fmtp(FMT_PATH_ERA5_SFC,
                                        year=t.year, month=t.month, day=t.day, hour=t.hour, var=var)
                for t in e5times
                for var in ERA5VARS}
    e5proc_shear = {
        f'era5p_shear_{t}': fmtp(FMT_PATH_ERA5P_SHEAR,
                                 year=t.year, month=t.month, day=t.day, hour=t.hour)
        for t in e5times
    }
    e5proc_vimfd = {
        f'era5p_vimfd_{t}': fmtp(FMT_PATH_ERA5P_VIMFD,
                                 year=t.year, month=t.month, day=t.day, hour=t.hour)
        for t in e5times
    }

    e5lsm = {'ERA5_land_sea_mask': PATH_ERA5_LAND_SEA_MASK}

    return e5inputs, e5proc_shear, e5proc_vimfd, e5pixel_inputs, e5lsm


def monthly_meanfield_conditional_inputs(month):
    rule_outputs = {
        f'era5_{y}': fmtp(FMT_PATH_ERA5_MEANFIELD,
                          year=y,
                          month=month)
        for y in YEARS
    }
    return e5meanfield_inputs


def meanfield_conditional_inputs():
    e5meanfield_inputs = {
        f'era5_{y}_{m}': fmtp(FMT_PATH_ERA5_MEANFIELD,
                              year=y,
                              month=m)
        for y in YEARS
        for m in MONTHS
    }
    return e5meanfield_inputs


def conditional_load_mcs_data(logger, year, month, inputs):
    logger.debug('Open tracks')
    tracks = McsTracks.open(inputs['tracks'], None)
    e5pixel_paths = [v for k, v in inputs.items()
                     if k[:7] == 'e5pixel']
    logger.debug('Open Pixel')
    e5pixel = xr.open_mfdataset(e5pixel_paths, concat_dim='time', combine='nested')
    return tracks, e5pixel


def conditional_load_data(logger, year, month, inputs, precursor_time=0):
    e5times = (gen_era5_times_for_month(year, month, include_precursor_offset=False) -
               pd.Timedelta(hours=precursor_time))

    tracks, e5pixel = conditional_load_mcs_data(logger, year, month, inputs)

    e5paths = [inputs[f'era5_{t}_{v}']
               for t in e5times
               for v in ERA5VARS]
    e5proc_shear_paths = [inputs[f'era5p_shear_{t}']
                          for t in e5times]
    e5proc_vimfd_paths = [inputs[f'era5p_vimfd_{t}']
                          for t in e5times]

    mcs_times = pd.DatetimeIndex(e5pixel.time)

    logger.debug('Open ERA5')
    e5ds = (xr.open_mfdataset(e5paths).sel(latitude=slice(60, -60))
            .interp(time=mcs_times).sel(time=mcs_times))
    logger.debug('Open proc shear')
    e5shear = (xr.open_mfdataset(e5proc_shear_paths, concat_dim='time', combine='nested')
               .interp(time=mcs_times).sel(time=mcs_times))
    logger.debug('Open proc VIMFD')
    e5vimfd = (xr.open_mfdataset(e5proc_vimfd_paths, concat_dim='time', combine='nested')
               .interp(time=mcs_times).sel(time=mcs_times))

    logger.debug('Load data')
    return tracks, e5pixel.load(), e5ds.load(), e5shear.load(), e5vimfd.load()


def conditional_load_meanfield_data(logger, inputs):
    e5meanfield = xr.open_mfdataset([v for k, v in inputs.items() if k[:5] == 'era5_'])
    return e5meanfield.mean(dim='time').load()


def get_bins(var):
    if var == 'cape':
        bins = np.linspace(0, 5000, 101)
    elif var == 'tcwv':
        bins = np.linspace(0, 100, 101)
    elif var[-5:] == 'shear':
        bins = np.linspace(0, 100, 101)
    elif var == 'vimfd':
        bins = np.linspace(-2e-3, 2e-3, 101)
    hist_mids = (bins[1:] + bins[:-1]) / 2
    return bins, hist_mids


def load_lsmask(path):
    lsmask = {}
    for lsreg in LS_REGIONS:
        # Build appropriate land-sea mask for region.
        da_lsmask = xr.load_dataarray(path)
        if lsreg == 'all':
            # All ones.
            lsmask['all'] = da_lsmask[0].sel(latitude=slice(60, -60)).values >= 0
        elif lsreg == 'land':
            # LSM has land == 1.
            lsmask['land'] = da_lsmask[0].sel(latitude=slice(60, -60)).values > 0.5
        elif lsreg == 'ocean':
            # LSM has ocean == 0.
            lsmask['ocean'] = da_lsmask[0].sel(latitude=slice(60, -60)).values <= 0.5
        else:
            raise ValueError(f'Unknown region: {lsreg}')
    return lsmask


def gen_region_masks(logger, e5pixel, tracks, core_method='tb'):
    mcs_core_shield_mask = []
    # Looping over subset of times.
    for i, time in enumerate(e5pixel.time.values):
        pdtime = pd.Timestamp(time)
        time = pdtime.to_pydatetime()
        if pdtime.hour == 0:
            print(time)

        # Get cloudnumbers (cns) for tracks at given time.
        ts = tracks.tracks_at_time(time)
        # tmask is a 2d mask that spans multiple tracks, getting
        # the cloudnumbers at *one time only*, that can be
        # used to get cloudnumbers.
        tmask = (ts.dstracks.base_time == pdtime).values
        if tmask.sum() == 0:
            logger.info(f'No times matched in tracks DB for {pdtime}')
            cns = np.array([])
        else:
            # Each cloudnumber can be used to link to the corresponding
            # cloud in the pixel data.
            cns = ts.dstracks.cloudnumber.values[tmask]
            # Nicer to have sorted values.
            cns.sort()

        # Tracked MCS shield (N.B. close to Tb < 241K but expanded by precip regions).
        # INCLUDES CONV CORE.
        mcs_core_shield_mask.append(e5pixel.cloudnumber[i].isin(cns).values)

    mcs_core_shield_mask = np.array(mcs_core_shield_mask)
    if core_method == 'tb':
        # Convective core Tb < 225K.
        core_mask = e5pixel.tb.values < 225
    elif core_method == 'precip':
        core_mask = e5pixel.precipitation.values > 2  # mm/hr
    # Non-MCS clouds (Tb < 241K). INCLUDES CONV CORE.
    # OPERATOR PRECEDENCE! Brackets are vital here.
    cloud_core_shield_mask = (e5pixel.cloudnumber.values > 0) & ~mcs_core_shield_mask
    # MCS conv core only.
    mcs_core_mask = mcs_core_shield_mask & core_mask
    # Cloud conv core only.
    cloud_core_mask = cloud_core_shield_mask & core_mask
    # Env is everything outside of these two regions.
    env_mask = ~mcs_core_shield_mask & ~cloud_core_shield_mask

    # Remove conv core from shields.
    mcs_shield_mask = mcs_core_shield_mask & ~mcs_core_mask
    cloud_shield_mask = cloud_core_shield_mask & ~cloud_core_mask

    # Verify mutual exclusivity and that all points are covered.
    assert (
        mcs_core_mask.astype(int) +
        mcs_shield_mask.astype(int) +
        cloud_core_mask.astype(int) +
        cloud_shield_mask.astype(int) +
        env_mask.astype(int) == 1
    ).all()

    return mcs_core_mask, mcs_shield_mask, cloud_core_mask, cloud_shield_mask, env_mask


def build_hourly_output_dataset(e5pixel):
    # Build inputs to Dataset
    coords = {'time': e5pixel.time}
    data_vars = {}
    for var in ERA5VARS + ['LLS_shear', 'L2M_shear', 'MLS_shear'] + ['vimfd']:
        bins, hist_mids = get_bins(var)
        hists = np.zeros((len(e5pixel.time), hist_mids.size))

        coords.update({f'{var}_hist_mids': hist_mids, f'{var}_bins': bins})
        for lsreg in LS_REGIONS:
            data_vars.update({
                f'{lsreg}_{var}_MCS_shield': (('time', f'{var}_hist_mid'), hists.copy()),
                f'{lsreg}_{var}_MCS_core': (('time', f'{var}_hist_mid'), hists.copy()),
                f'{lsreg}_{var}_cloud_shield': (('time', f'{var}_hist_mid'), hists.copy()),
                f'{lsreg}_{var}_cloud_core': (('time', f'{var}_hist_mid'), hists.copy()),
                f'{lsreg}_{var}_env': (('time', f'{var}_hist_mid'), hists.copy()),
            })

    # Make a dataset to hold all the histogram data.
    dsout = xr.Dataset(
        coords=coords,
        data_vars=data_vars,
    )
    return dsout


class PrecursorConditionalERA5HistHourly(TaskRule):
    @staticmethod
    def rule_inputs(year, month, precursor_time):
        e5inputs, e5proc_shear, e5proc_vimfd, e5pixel_inputs, e5lsm = conditional_inputs(year, month, precursor_time)
        inputs = {
            **e5inputs,
            **e5proc_shear,
            **e5proc_vimfd,
            **e5pixel_inputs,
            **e5lsm,
        }
        return inputs

    rule_outputs = {'hist': FMT_PATH_PRECURSOR_COND_HIST}

    var_matrix = {
        'year': YEARS,
        'month': MONTHS,
        'precursor_time': [1, 3, 6],
    }

    depends_on = [
        conditional_load_mcs_data,
        conditional_load_data,
        load_lsmask,
        build_hourly_output_dataset,
        get_bins,
        gen_region_masks
    ]

    def rule_run(self):
        self.logger.info('Load data')
        tracks, e5pixel, e5ds, e5shear, e5vimfd = conditional_load_data(
            self.logger,
            self.year,
            self.month,
            self.inputs,
            self.precursor_time
        )
        lsmask = load_lsmask(self.inputs['ERA5_land_sea_mask'])

        self.logger.info('Build output datasets')
        dsout = build_hourly_output_dataset(e5pixel)

        self.logger.info('Generate region masks')
        (
            mcs_core_mask,
            mcs_shield_mask,
            cloud_core_mask,
            cloud_shield_mask,
            env_mask
        ) = gen_region_masks(self.logger, e5pixel, tracks, core_method='tb')

        level2index = {
            'LLS_shear': 0,
            'L2M_shear': 1,
            'MLS_shear': 2,
        }
        # Pack all dataarrays into a common struct for easy looping.
        dataarrays = (
            # ERA5 variables.
            [(var, e5ds[var])
             for var in ERA5VARS] +
            # Derived shear variables.
            [(var, e5shear.isel(shear_level=level2index[var]).shear)
             for var in ['LLS_shear', 'L2M_shear', 'MLS_shear']] +
            # Derived VIMFD variables.
            [('vimfd', e5vimfd.vertically_integrated_moisture_flux_div)]
        )
        self.logger.info('Calc hists at each time')
        for i, time in enumerate(e5pixel.time.values):
            pdtime = pd.Timestamp(time)
            if pdtime.hour == 0:
                print(pdtime)
            def hist(data):
                # closure to save space.
                return np.histogram(data, bins=dsout.coords[f'{var}_bins'].values)[0]

            for (var, da), lsreg in product(dataarrays, LS_REGIONS):
                data = da.sel(time=pdtime).values
                # Calc hists. These 5 regions are mutually exclusive.
                dsout[f'{lsreg}_{var}_MCS_shield'][i] = hist(data[mcs_shield_mask[i] & lsmask[lsreg]])
                dsout[f'{lsreg}_{var}_MCS_core'][i] = hist(data[mcs_core_mask[i] & lsmask[lsreg]])
                dsout[f'{lsreg}_{var}_cloud_shield'][i] = hist(data[cloud_shield_mask[i] & lsmask[lsreg]])
                dsout[f'{lsreg}_{var}_cloud_core'][i] = hist(data[cloud_core_mask[i] & lsmask[lsreg]])
                dsout[f'{lsreg}_{var}_env'][i] = hist(data[env_mask[i] & lsmask[lsreg]])

        self.logger.info('write dsout')
        to_netcdf_tmp_then_copy(dsout, self.outputs['hist'])


class ConditionalERA5HistHourly(TaskRule):
    @staticmethod
    def rule_inputs(year, month, core_method):
        e5inputs, e5proc_shear, e5proc_vimfd, e5pixel_inputs, e5lsm = conditional_inputs(year, month)
        inputs = {
            **e5inputs,
            **e5proc_shear,
            **e5proc_vimfd,
            **e5pixel_inputs,
            **e5lsm,
        }
        return inputs

    rule_outputs = {'hist': FMT_PATH_COND_HIST_HOURLY}

    var_matrix = {
        'year': YEARS,
        'month': MONTHS,
        'core_method': ['tb', 'precip'],
    }

    depends_on = [
        conditional_load_mcs_data,
        conditional_load_data,
        load_lsmask,
        build_hourly_output_dataset,
        get_bins,
        gen_region_masks
    ]

    def rule_run(self):
        self.logger.info('Load data')
        tracks, e5pixel, e5ds, e5shear, e5vimfd = conditional_load_data(
            self.logger,
            self.year,
            self.month,
            self.inputs
        )
        lsmask = load_lsmask(self.inputs['ERA5_land_sea_mask'])

        self.logger.info('Build output datasets')
        dsout = build_hourly_output_dataset(e5pixel)

        self.logger.info('Generate region masks')
        (
            mcs_core_mask,
            mcs_shield_mask,
            cloud_core_mask,
            cloud_shield_mask,
            env_mask
        ) = gen_region_masks(self.logger, e5pixel, tracks, core_method=self.core_method)

        level2index = {
            'LLS_shear': 0,
            'L2M_shear': 1,
            'MLS_shear': 2,
        }
        # Pack all dataarrays into a common struct for easy looping.
        dataarrays = (
            # ERA5 variables.
            [(var, e5ds[var])
             for var in ERA5VARS] +
            # Derived shear variables.
            [(var, e5shear.isel(shear_level=level2index[var]).shear)
             for var in ['LLS_shear', 'L2M_shear', 'MLS_shear']] +
            # Derived VIMFD variables.
            [('vimfd', e5vimfd.vertically_integrated_moisture_flux_div)]
        )
        self.logger.info('Calc hists at each time')
        for i, time in enumerate(e5pixel.time.values):
            pdtime = pd.Timestamp(time)
            if pdtime.hour == 0:
                print(pdtime)
            def hist(data):
                # closure to save space.
                return np.histogram(data, bins=dsout.coords[f'{var}_bins'].values)[0]

            for (var, da), lsreg in product(dataarrays, LS_REGIONS):
                data = da.sel(time=pdtime).values
                # Calc hists. These 5 regions are mutually exclusive.
                dsout[f'{lsreg}_{var}_MCS_shield'][i] = hist(data[mcs_shield_mask[i] & lsmask[lsreg]])
                dsout[f'{lsreg}_{var}_MCS_core'][i] = hist(data[mcs_core_mask[i] & lsmask[lsreg]])
                dsout[f'{lsreg}_{var}_cloud_shield'][i] = hist(data[cloud_shield_mask[i] & lsmask[lsreg]])
                dsout[f'{lsreg}_{var}_cloud_core'][i] = hist(data[cloud_core_mask[i] & lsmask[lsreg]])
                dsout[f'{lsreg}_{var}_env'][i] = hist(data[env_mask[i] & lsmask[lsreg]])

        self.logger.info('write dsout')
        to_netcdf_tmp_then_copy(dsout, self.outputs['hist'])


class ConditionalERA5HistGridpoint(TaskRule):
    @staticmethod
    def rule_inputs(year, month):
        e5inputs, e5proc_shear, e5proc_vimfd, e5pixel_inputs, e5lsm = conditional_inputs(year, month)
        inputs = {
            **e5inputs,
            **e5proc_shear,
            **e5proc_vimfd,
            **e5pixel_inputs,
            **e5lsm,
        }
        return inputs

    rule_outputs = {'hist': FMT_PATH_COND_HIST_GRIDPOINT}

    var_matrix = {
        'year': YEARS,
        'month': MONTHS,
    }

    depends_on = [
        conditional_load_mcs_data,
        conditional_load_data,
        load_lsmask,
        get_bins,
        gen_region_masks
    ]

    def build_gridpoint_output_dataset(self, e5pixel):
        # Build inputs to Dataset
        coords = {'latitude': e5pixel.latitude, 'longitude': e5pixel.longitude}
        data_vars = {}
        for var in ERA5VARS + ['LLS_shear', 'L2M_shear', 'MLS_shear'] + ['vimfd']:
            bins, hist_mids = get_bins(var)
            hists = np.zeros((len(e5pixel.latitude), len(e5pixel.longitude), hist_mids.size))

            coords.update({f'{var}_hist_mids': hist_mids, f'{var}_bins': bins})
            data_vars.update({
                f'{var}_MCS_shield': (('latitude', 'longitude', f'{var}_hist_mid'), hists.copy()),
                f'{var}_MCS_core': (('latitude', 'longitude', f'{var}_hist_mid'), hists.copy()),
                f'{var}_cloud_shield': (('latitude', 'longitude', f'{var}_hist_mid'), hists.copy()),
                f'{var}_cloud_core': (('latitude', 'longitude', f'{var}_hist_mid'), hists.copy()),
                f'{var}_env': (('latitude', 'longitude', f'{var}_hist_mid'), hists.copy()),
            })

        # Make a dataset to hold all the histogram data.
        dsout = xr.Dataset(
            coords=coords,
            data_vars=data_vars,
        )
        return dsout

    def rule_run(self):
        self.logger.info('Load data')
        tracks, e5pixel, e5ds, e5shear, e5vimfd = conditional_load_data(
            self.logger,
            self.year,
            self.month,
            self.inputs
        )

        self.logger.info('Build output datasets')
        dsout = self.build_gridpoint_output_dataset(e5pixel)

        self.logger.info('Generate region masks')
        (
            mcs_core_mask,
            mcs_shield_mask,
            cloud_core_mask,
            cloud_shield_mask,
            env_mask
        ) = gen_region_masks(self.logger, e5pixel, tracks)

        level2index = {
            'LLS_shear': 0,
            'L2M_shear': 1,
            'MLS_shear': 2,
        }
        # Pack all dataarrays into a common struct for easy looping.
        dataarrays = (
            # ERA5 variables.
            [(var, e5ds[var])
             for var in ERA5VARS] +
            # Derived shear variables.
            [(var, e5shear.isel(shear_level=level2index[var]).shear)
             for var in ['LLS_shear', 'L2M_shear', 'MLS_shear']] +
            # Derived VIMFD variables.
            [('vimfd', e5vimfd.vertically_integrated_moisture_flux_div)]
        )

        self.logger.info('Calc hists at each gridpoint')
        for (var, da) in dataarrays:
            def hist(data):
                # closure to save space.
                return np.histogram(data, bins=dsout.coords[f'{var}_bins'].values)[0]

            data = da.values
            for i in range(data.shape[1]):
                if i % 10 == 0:
                    print(var, i, i / data.shape[1])
                for j in range(data.shape[2]):
                    dsout[f'{var}_MCS_shield'][i, j] = hist(data[:, i, j][mcs_shield_mask[:, i, j]])
                    dsout[f'{var}_MCS_core'][i, j] = hist(data[:, i, j][mcs_core_mask[:, i, j]])
                    dsout[f'{var}_cloud_shield'][i, j] = hist(data[:, i, j][cloud_shield_mask[:, i, j]])
                    dsout[f'{var}_cloud_core'][i, j] = hist(data[:, i, j][cloud_core_mask[:, i, j]])
                    dsout[f'{var}_env'][i, j] = hist(data[:, i, j][env_mask[:, i, j]])

        # These files are large. Use compression (makes write faster?).
        self.logger.info('write dsout')
        comp = dict(zlib=True, complevel=4)
        encoding = {var: comp for var in dsout.data_vars}
        to_netcdf_tmp_then_copy(dsout, self.outputs['hist'], encoding)


class ConditionalERA5HistMeanfield(TaskRule):
    @staticmethod
    def rule_inputs(year, month):
        e5inputs, e5proc_shear, e5proc_vimfd, e5pixel_inputs, e5lsm = conditional_inputs(year, month)
        e5meanfield_inputs = meanfield_conditional_inputs()
        inputs = {
            **e5pixel_inputs,
            **e5lsm,
            **e5meanfield_inputs,
        }
        return inputs

    rule_outputs = {'hist': FMT_PATH_COND_HIST_MEANFIELD}

    var_matrix = {
        'year': YEARS,
        'month': MONTHS,
    }
    depends_on = [
        conditional_load_mcs_data,
        conditional_load_meanfield_data,
        gen_region_masks,
        get_bins,
    ]

    def rule_run(self):
        self.logger.info('Load data')
        tracks, e5pixel = conditional_load_mcs_data(
            self.logger,
            self.year,
            self.month,
            self.inputs
        )
        e5meanfield = conditional_load_meanfield_data(
            self.logger,
            self.inputs,
        )
        lsmask = load_lsmask(self.inputs['ERA5_land_sea_mask'])

        self.logger.info('Build output datasets')
        dsout = build_hourly_output_dataset(e5pixel)

        self.logger.info('Generate region masks')
        (
            mcs_core_mask,
            mcs_shield_mask,
            cloud_core_mask,
            cloud_shield_mask,
            env_mask
        ) = gen_region_masks(self.logger, e5pixel, tracks)

        level2index = {
            'LLS_shear': 0,
            'L2M_shear': 1,
            'MLS_shear': 2,
        }
        # Pack all dataarrays into a common struct for easy looping.
        dataarrays = (
            [(var, e5meanfield[var])
             for var in ERA5VARS] +
            # Derived shear variables.
            [(var, e5meanfield.isel(shear_level=level2index[var]).shear)
             for var in ['LLS_shear', 'L2M_shear', 'MLS_shear']] +
            # Derived VIMFD variables.
            [('vimfd', e5meanfield.vertically_integrated_moisture_flux_div)]
        )

        self.logger.info('Calc meanfield hists at each gridpoint')
        for i, time in enumerate(e5pixel.time.values):
            pdtime = pd.Timestamp(time)
            if pdtime.hour == 0:
                print(pdtime)
            def hist(data):
                # closure to save space.
                return np.histogram(data, bins=dsout.coords[f'{var}_bins'].values)[0]

            for (var, da), lsreg in product(dataarrays, LS_REGIONS):
                data = da.values
                # Calc hists. These 5 regions are mutually exclusive.
                dsout[f'{lsreg}_{var}_MCS_shield'][i] = hist(data[mcs_shield_mask[i] & lsmask[lsreg]])
                dsout[f'{lsreg}_{var}_MCS_core'][i] = hist(data[mcs_core_mask[i] & lsmask[lsreg]])
                dsout[f'{lsreg}_{var}_cloud_shield'][i] = hist(data[cloud_shield_mask[i] & lsmask[lsreg]])
                dsout[f'{lsreg}_{var}_cloud_core'][i] = hist(data[cloud_core_mask[i] & lsmask[lsreg]])
                dsout[f'{lsreg}_{var}_env'][i] = hist(data[env_mask[i] & lsmask[lsreg]])

        self.logger.info('write dsout')
        to_netcdf_tmp_then_copy(dsout, self.outputs['hist'])


class CombineConditionalERA5HistGridpoint(TaskRule):
    @staticmethod
    def rule_inputs(year):
        inputs = {
            f'hist_{year}_{month}': fmtp(ConditionalERA5HistGridpoint.rule_outputs['hist'],
                                         year=year,
                                         month=month)
            for month in MONTHS
        }
        return inputs

    rule_outputs = {'hist': FMT_PATH_COMBINED_COND_HIST_GRIDPOINT}

    var_matrix = {'year': YEARS}
    # Takes a lot of mem to combine these datasets!
    config = {'slurm': {'mem': 512000, 'partition': 'high-mem'}}

    def rule_run(self):
        datasets = [xr.open_dataset(p) for p in self.inputs.values()]
        assert len(datasets) == 12

        self.logger.info('Concat datasets')
        ds = xr.concat(datasets, pd.Index(range(12), name='time_index'))
        dsout = ds.sum(dim='time_index')

        self.logger.info('Write ds.sum')
        comp = dict(zlib=True, complevel=4)
        encoding = {var: comp for var in dsout.data_vars}
        to_netcdf_tmp_then_copy(dsout, self.outputs['hist'], encoding)

