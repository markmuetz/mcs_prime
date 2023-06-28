from collections import defaultdict
from itertools import product

import numpy as np
import pandas as pd
import xarray as xr
from mcs_prime import McsTracks
from remake import Remake, TaskRule
from remake.util import format_path as fmtp

import mcs_prime.mcs_prime_config_util as cu
from mcs_prime.mcs_prime_config_util import gen_region_masks

TODOS = '''
TODOS
* Make sure filenames are consistent
* Make sure variables names are sensible/consistent
* Docstrings for all fns, classes
* Validate all data
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
    e5times = cu.gen_era5_times_for_month(year, month, include_precursor_offset=False) - pd.Timedelta(
        hours=precursor_time
    )

    # OK, this is a little complicated.
    # The logic is the same as above in GenPixelDataOnERA5Grid.
    # I only want to add the output of GenPixelDataOnERA5Grid, if the original MCS pixel data
    # exists. But I now need to do for one month.
    pixel_on_e5_inputs = {}
    for daily_pixel_time in daily_pixel_times:
        pixel_times, pixel_inputs = cu.pixel_inputs_cache.all_pixel_inputs[
            (daily_pixel_time.year, daily_pixel_time.month, daily_pixel_time.day)
        ]
        pixel_output_paths = [
            fmtp(cu.FMT_PATH_PIXEL_ON_ERA5, year=t.year, month=t.month, day=t.day, hour=t.hour) for t in pixel_times
        ]
        pixel_on_e5_inputs.update({f'pixel_on_e5_{t}': p for t, p in zip(pixel_times, pixel_output_paths)})
    pixel_on_e5_inputs['tracks'] = cu.fmt_mcs_stats_path(year)

    e5inputs = {
        f'era5_{t}_{var}': fmtp(cu.FMT_PATH_ERA5_SFC, year=t.year, month=t.month, day=t.day, hour=t.hour, var=var)
        for t in e5times
        for var in cu.ERA5VARS
    }
    e5proc_shear = {
        f'era5p_shear_{t}': fmtp(cu.FMT_PATH_ERA5P_SHEAR, year=t.year, month=t.month, day=t.day, hour=t.hour)
        for t in e5times
    }
    e5proc_vimfd = {
        f'era5p_vimfd_{t}': fmtp(cu.FMT_PATH_ERA5P_VIMFD, year=t.year, month=t.month, day=t.day, hour=t.hour)
        for t in e5times
    }

    e5lsm = {'ERA5_land_sea_mask': cu.PATH_ERA5_LAND_SEA_MASK}

    return e5inputs, e5proc_shear, e5proc_vimfd, pixel_on_e5_inputs, e5lsm


def monthly_meanfield_conditional_inputs(month):
    rule_outputs = {f'era5_{y}': fmtp(cu.FMT_PATH_ERA5_MEANFIELD, year=y, month=month) for y in cu.YEARS}
    return e5meanfield_inputs


def meanfield_conditional_inputs():
    e5meanfield_inputs = {
        f'era5_{y}_{m}': fmtp(cu.FMT_PATH_ERA5_MEANFIELD, year=y, month=m) for y in cu.YEARS for m in cu.MONTHS
    }
    return e5meanfield_inputs


def conditional_load_mcs_data(logger, year, month, inputs):
    logger.debug('Open tracks')
    tracks = McsTracks.open(inputs['tracks'], None)
    pixel_on_e5_paths = [v for k, v in inputs.items() if k.startswith('pixel_on_e5')]

    logger.debug('Open Pixel')
    pixel_on_e5 = xr.open_mfdataset(pixel_on_e5_paths)
    return tracks, pixel_on_e5


def conditional_load_data(logger, year, month, inputs, precursor_time=0):
    e5times = cu.gen_era5_times_for_month(year, month, include_precursor_offset=False) - pd.Timedelta(
        hours=precursor_time
    )

    tracks, pixel_on_e5 = conditional_load_mcs_data(logger, year, month, inputs)

    e5paths = [inputs[f'era5_{t}_{v}'] for t in e5times for v in cu.ERA5VARS]
    e5proc_shear_paths = [inputs[f'era5p_shear_{t}'] for t in e5times]
    e5proc_vimfd_paths = [inputs[f'era5p_vimfd_{t}'] for t in e5times]

    mcs_times = pd.DatetimeIndex(pixel_on_e5.time)

    logger.debug('Open ERA5')
    e5ds = xr.open_mfdataset(e5paths).sel(latitude=slice(60, -60)).interp(time=mcs_times).sel(time=mcs_times)
    logger.debug('Open proc shear')
    e5shear = xr.open_mfdataset(e5proc_shear_paths).interp(time=mcs_times).sel(time=mcs_times)
    logger.debug('Open proc VIMFD')
    e5vimfd = xr.open_mfdataset(e5proc_vimfd_paths).interp(time=mcs_times).sel(time=mcs_times)

    logger.debug('Load data')
    return tracks, pixel_on_e5, xr.merge([e5ds.load(), e5shear.load(), e5vimfd.load()])


def conditional_load_meanfield_data(logger, inputs):
    e5meanfield = xr.open_mfdataset([v for k, v in inputs.items() if k.startswith('era5_')])
    return e5meanfield.mean(dim='time').load()


def calc_frac_growth(dstracks):
    mask = ~np.isnan(dstracks.area)

    dt = 1
    fractional_area_growth = dstracks.area.values.copy()
    fractional_area_growth[:, 1:] = (1 / dstracks.area[:, 1:].values) * (
        (dstracks.area[:, 1:].values - dstracks.area[:, :-1].values) / dt
    )
    area_mask = mask.copy()
    area_mask[:, 0] = False
    return fractional_area_growth, area_mask


def calc_growth_masks(fractional_area_growth, thresh=0.5):
    growth_mask = fractional_area_growth >= thresh
    stable_mask = (fractional_area_growth > -thresh) & (fractional_area_growth < thresh)
    decay_mask = fractional_area_growth <= thresh

    return growth_mask, stable_mask, decay_mask


def gen_mcs_lifecycle_region_masks(logger, pixel_on_e5, tracks):
    # Calc some masks to select MCSs in different phases.
    init_mask = np.zeros_like(tracks.dstracks.area.values, dtype=bool)
    init_mask[:, :2] = 1
    growth_mask, stable_mask, decay_mask = calc_growth_masks(calc_frac_growth(tracks.dstracks)[0])
    phase_masks = {
        'init': init_mask,
        'growth': growth_mask,
        'stable': stable_mask,
        'decay': decay_mask,
    }

    mcs_mask_for_phase = defaultdict(list)

    # Looping over subset of times.
    for i, time in enumerate(pixel_on_e5.time.values):
        pdtime = pd.Timestamp(time)
        if pdtime.hour == 0:
            print(pdtime)

        # Get cloudnumbers (cns) for tracks at given time.
        # tmask is a 2d mask that spans multiple tracks, getting
        # the cloudnumbers at *one time only*, that can be
        # used to get cloudnumbers.
        tmask = (tracks.dstracks.base_time == pdtime).values
        for phase, phase_mask in phase_masks.items():
            if (tmask & phase_mask).sum() == 0:
                logger.info(f'No times matched in tracks DB for {pdtime}')
                cns = np.array([])
            else:
                # Each cloudnumber can be used to link to the corresponding
                # cloud in the pixel data.
                cns = tracks.dstracks.cloudnumber.values[tmask & phase_mask]
                # Nicer to have sorted values.
                cns.sort()

            # Tracked MCS shield (N.B. close to Tb < 241K but expanded by precip regions).
            # INCLUDES CONV CORE.
            mcs_mask_for_phase[phase].append(pixel_on_e5.cloudnumber[i].isin(cns).values)

    for phase in phase_masks.keys():
        mcs_mask_for_phase[phase] = np.array(mcs_mask_for_phase[phase])

    # Convective core Tb < 225K.
    core_mask = pixel_on_e5.tb.values < 225

    mcs_core_mask_for_phase = defaultdict(list)
    mcs_shield_mask_for_phase = defaultdict(list)

    for phase in phase_masks.keys():
        mcs_core_mask_for_phase[phase] = mcs_mask_for_phase[phase] & core_mask
        mcs_shield_mask_for_phase[phase] = mcs_mask_for_phase[phase] & ~core_mask

    return mcs_core_mask_for_phase, mcs_shield_mask_for_phase


def build_hourly_output_dataset(pixel_on_e5):
    # Build inputs to Dataset
    coords = {'time': pixel_on_e5.time}
    data_vars = {}
    for var in cu.EXTENDED_ERA5VARS:
        bins, hist_mids = cu.get_bins(var)
        hists = np.zeros((len(pixel_on_e5.time), hist_mids.size))

        coords.update({f'{var}_hist_mids': hist_mids, f'{var}_bins': bins})
        for lsreg in cu.LS_REGIONS:
            data_vars.update(
                {
                    f'{lsreg}_{var}_MCS_shield': (('time', f'{var}_hist_mid'), hists.copy()),
                    f'{lsreg}_{var}_MCS_core': (('time', f'{var}_hist_mid'), hists.copy()),
                    f'{lsreg}_{var}_cloud_shield': (('time', f'{var}_hist_mid'), hists.copy()),
                    f'{lsreg}_{var}_cloud_core': (('time', f'{var}_hist_mid'), hists.copy()),
                    f'{lsreg}_{var}_env': (('time', f'{var}_hist_mid'), hists.copy()),
                }
            )

    # Make a dataset to hold all the histogram data.
    dsout = xr.Dataset(
        coords=coords,
        data_vars=data_vars,
    )
    return dsout


class PrecursorConditionalERA5HistHourly(TaskRule):
    enabled = False

    @staticmethod
    def rule_inputs(year, month, precursor_time):
        e5inputs, e5proc_shear, e5proc_vimfd, pixel_on_e5_inputs, e5lsm = conditional_inputs(
            year, month, precursor_time
        )
        inputs = {
            **e5inputs,
            **e5proc_shear,
            **e5proc_vimfd,
            **pixel_on_e5_inputs,
            **e5lsm,
        }
        return inputs

    rule_outputs = {'hist': cu.FMT_PATH_PRECURSOR_COND_HIST}

    var_matrix = {
        'year': cu.YEARS,
        'month': cu.MONTHS,
        'precursor_time': [1, 3, 6],
    }

    depends_on = [
        conditional_load_mcs_data,
        conditional_load_data,
        cu.load_lsmask,
        build_hourly_output_dataset,
        cu.get_bins,
        gen_region_masks,
    ]

    def rule_run(self):
        self.logger.info('Load data')
        tracks, pixel_on_e5, e5ds = conditional_load_data(
            self.logger, self.year, self.month, self.inputs, self.precursor_time
        )
        lsmask = cu.load_lsmask(self.inputs['ERA5_land_sea_mask'])

        self.logger.info('Build output datasets')
        dsout = build_hourly_output_dataset(pixel_on_e5)

        self.logger.info('Generate region masks')
        (mcs_core_mask, mcs_shield_mask, cloud_core_mask, cloud_shield_mask, env_mask) = gen_region_masks(
            self.logger, pixel_on_e5, tracks, core_method='tb'
        )

        self.logger.info('Calc hists at each time')
        for i, time in enumerate(pixel_on_e5.time.values):
            pdtime = pd.Timestamp(time)
            if pdtime.hour == 0:
                print(pdtime)

            def hist(data):
                # closure to save space.
                return np.histogram(data, bins=dsout.coords[f'{var}_bins'].values)[0]

            for var, lsreg in product(e5ds.data_vars.keys(), cu.LS_REGIONS):
                data = e5ds[var].sel(time=pdtime).values
                # Calc hists. These 5 regions are mutually exclusive.
                dsout[f'{lsreg}_{var}_MCS_shield'][i] = hist(data[mcs_shield_mask[i] & lsmask[lsreg]])
                dsout[f'{lsreg}_{var}_MCS_core'][i] = hist(data[mcs_core_mask[i] & lsmask[lsreg]])
                dsout[f'{lsreg}_{var}_cloud_shield'][i] = hist(data[cloud_shield_mask[i] & lsmask[lsreg]])
                dsout[f'{lsreg}_{var}_cloud_core'][i] = hist(data[cloud_core_mask[i] & lsmask[lsreg]])
                dsout[f'{lsreg}_{var}_env'][i] = hist(data[env_mask[i] & lsmask[lsreg]])

        self.logger.info('write dsout')
        cu.to_netcdf_tmp_then_copy(dsout, self.outputs['hist'])


class ConditionalERA5HistHourly(TaskRule):
    @staticmethod
    def rule_inputs(year, month, core_method):
        e5inputs, e5proc_shear, e5proc_vimfd, pixel_on_e5_inputs, e5lsm = conditional_inputs(year, month)
        inputs = {
            **e5inputs,
            **e5proc_shear,
            **e5proc_vimfd,
            **pixel_on_e5_inputs,
            **e5lsm,
        }
        return inputs

    rule_outputs = {'hist': cu.FMT_PATH_COND_HIST_HOURLY}

    var_matrix = {
        'year': cu.YEARS,
        'month': cu.MONTHS,
        'core_method': ['tb', 'precip'],
    }

    depends_on = [
        conditional_load_mcs_data,
        conditional_load_data,
        cu.load_lsmask,
        build_hourly_output_dataset,
        cu.get_bins,
        gen_region_masks,
    ]

    def rule_run(self):
        self.logger.info('Load data')
        tracks, pixel_on_e5, e5ds = conditional_load_data(self.logger, self.year, self.month, self.inputs)
        lsmask = cu.load_lsmask(self.inputs['ERA5_land_sea_mask'])

        self.logger.info('Build output datasets')
        dsout = build_hourly_output_dataset(pixel_on_e5)

        self.logger.info('Generate region masks')
        (mcs_core_mask, mcs_shield_mask, cloud_core_mask, cloud_shield_mask, env_mask) = gen_region_masks(
            self.logger, pixel_on_e5, tracks, core_method=self.core_method
        )

        self.logger.info('Calc hists at each time')
        for i, time in enumerate(pixel_on_e5.time.values):
            pdtime = pd.Timestamp(time)
            if pdtime.hour == 0:
                print(pdtime)

            def hist(data):
                # closure to save space.
                return np.histogram(data, bins=dsout.coords[f'{var}_bins'].values)[0]

            for var, lsreg in product(e5ds.data_vars.keys(), cu.LS_REGIONS):
                data = e5ds[var].sel(time=pdtime).values
                # Calc hists. These 5 regions are mutually exclusive.
                dsout[f'{lsreg}_{var}_MCS_shield'][i] = hist(data[mcs_shield_mask[i] & lsmask[lsreg]])
                dsout[f'{lsreg}_{var}_MCS_core'][i] = hist(data[mcs_core_mask[i] & lsmask[lsreg]])
                dsout[f'{lsreg}_{var}_cloud_shield'][i] = hist(data[cloud_shield_mask[i] & lsmask[lsreg]])
                dsout[f'{lsreg}_{var}_cloud_core'][i] = hist(data[cloud_core_mask[i] & lsmask[lsreg]])
                dsout[f'{lsreg}_{var}_env'][i] = hist(data[env_mask[i] & lsmask[lsreg]])

        self.logger.info('write dsout')
        cu.to_netcdf_tmp_then_copy(dsout, self.outputs['hist'])


class ConditionalERA5HistHourlyMCSLifecycle(TaskRule):
    @staticmethod
    def rule_inputs(year, month):
        e5inputs, e5proc_shear, e5proc_vimfd, pixel_on_e5_inputs, e5lsm = conditional_inputs(year, month)
        inputs = {
            **e5inputs,
            **e5proc_shear,
            **e5proc_vimfd,
            **pixel_on_e5_inputs,
            **e5lsm,
        }
        return inputs

    rule_outputs = {'hist': cu.FMT_PATH_COND_MCS_LIFECYCLE_HIST_HOURLY}

    var_matrix = {
        'year': cu.YEARS,
        'month': cu.MONTHS,
    }

    depends_on = [
        conditional_load_mcs_data,
        conditional_load_data,
        cu.load_lsmask,
        build_hourly_output_dataset,
        cu.get_bins,
        gen_mcs_lifecycle_region_masks,
    ]

    def build_mcs_lifecycle_hourly_output_dataset(self, pixel_on_e5, mcs_core_mask_for_phase):
        # Build inputs to Dataset
        coords = {'time': pixel_on_e5.time}
        data_vars = {}
        for var in cu.EXTENDED_ERA5VARS:
            bins, hist_mids = cu.get_bins(var)
            hists = np.zeros((len(pixel_on_e5.time), hist_mids.size))

            coords.update({f'{var}_hist_mids': hist_mids, f'{var}_bins': bins})
            for lsreg in cu.LS_REGIONS:
                for phase in mcs_core_mask_for_phase.keys():
                    data_vars.update(
                        {
                            f'{lsreg}_{var}_{phase}_MCS_shield': (('time', f'{var}_hist_mid'), hists.copy()),
                            f'{lsreg}_{var}_{phase}_MCS_core': (('time', f'{var}_hist_mid'), hists.copy()),
                        }
                    )

        # Make a dataset to hold all the histogram data.
        dsout = xr.Dataset(
            coords=coords,
            data_vars=data_vars,
        )
        return dsout

    def rule_run(self):
        self.logger.info('Load data')
        tracks, pixel_on_e5, e5ds = conditional_load_data(self.logger, self.year, self.month, self.inputs)
        lsmask = cu.load_lsmask(self.inputs['ERA5_land_sea_mask'])

        self.logger.info('Generate region masks')
        (mcs_core_mask_for_phase, mcs_shield_mask_for_phase) = gen_mcs_lifecycle_region_masks(
            self.logger, pixel_on_e5, tracks
        )

        self.logger.info('Build output datasets')
        dsout = self.build_mcs_lifecycle_hourly_output_dataset(pixel_on_e5, mcs_core_mask_for_phase)

        self.logger.info('Calc hists at each time')
        for i, time in enumerate(pixel_on_e5.time.values):
            pdtime = pd.Timestamp(time)
            if pdtime.hour == 0:
                print(pdtime)

            def hist(data):
                # closure to save space.
                return np.histogram(data, bins=dsout.coords[f'{var}_bins'].values)[0]

            for var, lsreg in product(e5ds.data_vars.keys(), cu.LS_REGIONS):
                data = e5ds[var].sel(time=pdtime).values
                for phase in mcs_core_mask_for_phase.keys():
                    core_mask = mcs_core_mask_for_phase[phase]
                    shield_mask = mcs_shield_mask_for_phase[phase]
                    dsout[f'{lsreg}_{var}_{phase}_MCS_core'][i] = hist(data[core_mask[i] & lsmask[lsreg]])
                    dsout[f'{lsreg}_{var}_{phase}_MCS_shield'][i] = hist(data[shield_mask[i] & lsmask[lsreg]])

        self.logger.info('write dsout')
        cu.to_netcdf_tmp_then_copy(dsout, self.outputs['hist'])


class ConditionalERA5HistGridpoint(TaskRule):
    @staticmethod
    def rule_inputs(year, month):
        e5inputs, e5proc_shear, e5proc_vimfd, pixel_on_e5_inputs, e5lsm = conditional_inputs(year, month)
        inputs = {
            **e5inputs,
            **e5proc_shear,
            **e5proc_vimfd,
            **pixel_on_e5_inputs,
            **e5lsm,
        }
        return inputs

    rule_outputs = {'hist': cu.FMT_PATH_COND_HIST_GRIDPOINT}

    var_matrix = {
        'year': cu.YEARS,
        'month': cu.MONTHS,
    }

    depends_on = [conditional_load_mcs_data, conditional_load_data, cu.load_lsmask, cu.get_bins, gen_region_masks]

    def build_gridpoint_output_dataset(self, pixel_on_e5):
        # Build inputs to Dataset
        coords = {'latitude': pixel_on_e5.latitude, 'longitude': pixel_on_e5.longitude}
        data_vars = {}
        for var in cu.EXTENDED_ERA5VARS:
            bins, hist_mids = cu.get_bins(var)
            hists = np.zeros((len(pixel_on_e5.latitude), len(pixel_on_e5.longitude), hist_mids.size))

            coords.update({f'{var}_hist_mids': hist_mids, f'{var}_bins': bins})
            data_vars.update(
                {
                    f'{var}_MCS_shield': (('latitude', 'longitude', f'{var}_hist_mid'), hists.copy()),
                    f'{var}_MCS_core': (('latitude', 'longitude', f'{var}_hist_mid'), hists.copy()),
                    f'{var}_cloud_shield': (('latitude', 'longitude', f'{var}_hist_mid'), hists.copy()),
                    f'{var}_cloud_core': (('latitude', 'longitude', f'{var}_hist_mid'), hists.copy()),
                    f'{var}_env': (('latitude', 'longitude', f'{var}_hist_mid'), hists.copy()),
                }
            )

        # Make a dataset to hold all the histogram data.
        dsout = xr.Dataset(
            coords=coords,
            data_vars=data_vars,
        )
        return dsout

    def rule_run(self):
        self.logger.info('Load data')
        tracks, pixel_on_e5, e5ds = conditional_load_data(self.logger, self.year, self.month, self.inputs)

        self.logger.info('Build output datasets')
        dsout = self.build_gridpoint_output_dataset(pixel_on_e5)

        self.logger.info('Generate region masks')
        (mcs_core_mask, mcs_shield_mask, cloud_core_mask, cloud_shield_mask, env_mask) = gen_region_masks(
            self.logger, pixel_on_e5, tracks
        )

        self.logger.info('Calc hists at each gridpoint')
        for var in e5ds.data_vars.keys():

            def hist(data):
                # closure to save space.
                return np.histogram(data, bins=dsout.coords[f'{var}_bins'].values)[0]

            data = e5ds[var].values
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
        cu.to_netcdf_tmp_then_copy(dsout, self.outputs['hist'], encoding)


class ConditionalERA5HistMeanfield(TaskRule):
    @staticmethod
    def rule_inputs(year, month):
        e5inputs, e5proc_shear, e5proc_vimfd, pixel_on_e5_inputs, e5lsm = conditional_inputs(year, month)
        e5meanfield_inputs = meanfield_conditional_inputs()
        inputs = {
            **pixel_on_e5_inputs,
            **e5lsm,
            **e5meanfield_inputs,
        }
        return inputs

    rule_outputs = {'hist': cu.FMT_PATH_COND_HIST_MEANFIELD}

    var_matrix = {
        'year': cu.YEARS,
        'month': cu.MONTHS,
    }
    depends_on = [
        conditional_load_mcs_data,
        conditional_load_meanfield_data,
        gen_region_masks,
        cu.get_bins,
    ]

    def rule_run(self):
        self.logger.info('Load data')
        tracks, pixel_on_e5 = conditional_load_mcs_data(self.logger, self.year, self.month, self.inputs)
        e5meanfield = conditional_load_meanfield_data(
            self.logger,
            self.inputs,
        )
        lsmask = cu.load_lsmask(self.inputs['ERA5_land_sea_mask'])

        self.logger.info('Build output datasets')
        dsout = build_hourly_output_dataset(pixel_on_e5)

        self.logger.info('Generate region masks')
        (mcs_core_mask, mcs_shield_mask, cloud_core_mask, cloud_shield_mask, env_mask) = gen_region_masks(
            self.logger, pixel_on_e5, tracks
        )

        self.logger.info('Calc meanfield hists at each gridpoint')
        for i, time in enumerate(pixel_on_e5.time.values):
            pdtime = pd.Timestamp(time)
            if pdtime.hour == 0:
                print(pdtime)

            def hist(data):
                # closure to save space.
                return np.histogram(data, bins=dsout.coords[f'{var}_bins'].values)[0]

            for var, lsreg in product(e5meanfield.data_vars.keys(), cu.LS_REGIONS):
                data = e5meanfield[var].values
                # Calc hists. These 5 regions are mutually exclusive.
                dsout[f'{lsreg}_{var}_MCS_shield'][i] = hist(data[mcs_shield_mask[i] & lsmask[lsreg]])
                dsout[f'{lsreg}_{var}_MCS_core'][i] = hist(data[mcs_core_mask[i] & lsmask[lsreg]])
                dsout[f'{lsreg}_{var}_cloud_shield'][i] = hist(data[cloud_shield_mask[i] & lsmask[lsreg]])
                dsout[f'{lsreg}_{var}_cloud_core'][i] = hist(data[cloud_core_mask[i] & lsmask[lsreg]])
                dsout[f'{lsreg}_{var}_env'][i] = hist(data[env_mask[i] & lsmask[lsreg]])

        self.logger.info('write dsout')
        cu.to_netcdf_tmp_then_copy(dsout, self.outputs['hist'])


class CombineConditionalERA5HistGridpoint(TaskRule):
    @staticmethod
    def rule_inputs(year):
        inputs = {
            f'hist_{year}_{month}': fmtp(ConditionalERA5HistGridpoint.rule_outputs['hist'], year=year, month=month)
            for month in cu.MONTHS
        }
        return inputs

    rule_outputs = {'hist': cu.FMT_PATH_COMBINED_COND_HIST_GRIDPOINT}

    var_matrix = {'year': cu.YEARS}
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
        cu.to_netcdf_tmp_then_copy(dsout, self.outputs['hist'], encoding)
