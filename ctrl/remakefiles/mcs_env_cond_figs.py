from itertools import product

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import xarray as xr
from timeit import default_timer

from remake import Remake, TaskRule
from remake.util import format_path as fmtp
from remake.global_timer import get_global_timer

from mcs_prime import PATHS, McsTracks, mcs_mask_plotter
import mcs_prime.mcs_prime_config_util as cu

slurm_config = {'account': 'short4hr', 'queue': 'short-serial-4hr', 'mem': 64000}
era5_histograms_plotting = Remake(config=dict(slurm=slurm_config, content_checks=False))


def get_labels(var):
    labels = {
        'cape': 'CAPE (J kg$^{-1}$)',
        'tcwv': 'TCWV (mm)',
        'vertically_integrated_moisture_flux_div': 'VIMFD (kg m$^{-2}$ s$^{-1}$)',
        'shear_0': 'LLS (m s$^{-1}$)',
        'shear_1': 'L2MS (m s$^{-1}$)',
        'shear_2': 'M2HS (m s$^{-1}$)',
        'shear_3': 'DS (m s$^{-1}$)',
        'theta_e_mid': r'$\theta_e$ (K)',
        'RHlow': 'RHlow (-)',
        'RHmid': 'RHmid (-)',
        'delta_3h_cape': r'CAPE $\Delta$ 3h (J kg$^{-1}$)',
        'delta_3h_tcwv': r'TCWV $\Delta$ 3h (mm)',
    }
    return labels[var]


def plot_hist(ds, ax=None, reg='all', var='cape', log=True):
    def _plot_hist(ds, ax, h, fmt, title):
        if var == 'vertically_integrated_moisture_flux_div':
            x = -ds[f'{var}_hist_mids'].values * 1e4
        else:
            x = ds[f'{var}_hist_mids'].values
        ax.plot(x, h, fmt, label=title)

    # ax.set_title(f'{var.upper()} distributions')
    # _plot_hist(ds, ax, np.nansum(ds[f'{reg}_{var}_MCS_core'].values, axis=0), 'r-', 'MCS core')
    # _plot_hist(ds, ax, np.nansum(ds[f'{reg}_{var}_MCS_shield'].values, axis=0), 'r--', 'MCS shield')
    # _plot_hist(ds, ax, np.nansum(ds[f'{reg}_{var}_cloud_core'].values, axis=0), 'b-', 'cloud core')
    # _plot_hist(ds, ax, np.nansum(ds[f'{reg}_{var}_cloud_shield'].values, axis=0), 'b--', 'cloud shield')
    # _plot_hist(ds, ax, np.nansum(ds[f'{reg}_{var}_env'].values, axis=0), 'k-', 'env')
    _plot_hist(ds, ax, ds[f'{reg}_{var}_MCS_core_hist_data'].values, 'r-', 'MCS core')
    _plot_hist(ds, ax, ds[f'{reg}_{var}_MCS_shield_hist_data'].values, 'r--', 'MCS shield')
    _plot_hist(ds, ax, ds[f'{reg}_{var}_cloud_core_hist_data'].values, 'b-', 'cloud core')
    _plot_hist(ds, ax, ds[f'{reg}_{var}_cloud_shield_hist_data'].values, 'b--', 'cloud shield')
    _plot_hist(ds, ax, ds[f'{reg}_{var}_env_hist_data'].values, 'k-', 'env')
    # ax.legend()
    if log:
        ax.set_yscale('log')

    # ax.set_xlabel(get_labels(var))


def plot_hist_mode(ds, mode, ax=None, reg='all', var='cape', log=True):
    def _plot_hist(ds, ax, h, fmt, title):
        if var == 'vertically_integrated_moisture_flux_div':
            x = -ds[f'{var}_hist_mids'].values * 1e4
        else:
            x = ds[f'{var}_hist_mids'].values
        ax.plot(x, h, fmt, label=title)

    # ax.set_title(f'{var.upper()} distributions')
    # _plot_hist(ds, ax, np.nansum(ds[f'{reg}_{var}_MCS_core'].values, axis=0), 'r-', 'MCS core')
    # _plot_hist(ds, ax, np.nansum(ds[f'{reg}_{var}_MCS_shield'].values, axis=0), 'r--', 'MCS shield')
    # _plot_hist(ds, ax, np.nansum(ds[f'{reg}_{var}_cloud_core'].values, axis=0), 'b-', 'cloud core')
    # _plot_hist(ds, ax, np.nansum(ds[f'{reg}_{var}_cloud_shield'].values, axis=0), 'b--', 'cloud shield')
    # _plot_hist(ds, ax, np.nansum(ds[f'{reg}_{var}_env'].values, axis=0), 'k-', 'env')
    _plot_hist(ds, ax, ds[f'{reg}_{var}_MCS_core_hist_{mode}_data'].values, 'r-', 'MCS core')
    _plot_hist(ds, ax, ds[f'{reg}_{var}_MCS_shield_hist_{mode}_data'].values, 'r--', 'MCS shield')
    _plot_hist(ds, ax, ds[f'{reg}_{var}_cloud_core_hist_{mode}_data'].values, 'b-', 'cloud core')
    _plot_hist(ds, ax, ds[f'{reg}_{var}_cloud_shield_hist_{mode}_data'].values, 'b--', 'cloud shield')
    _plot_hist(ds, ax, ds[f'{reg}_{var}_env_hist_{mode}_data'].values, 'k-', 'env')
    # ax.legend()
    if log:
        ax.set_yscale('log')

    # ax.set_xlabel(get_labels(var))


def plot_hist_probs(ds, ax=None, reg='all', var='cape'):
    if var == 'vertically_integrated_moisture_flux_div':
        x = -ds[f'{var}_hist_mids'].values * 1e4
    else:
        x = ds[f'{var}_hist_mids'].values
    # ax.set_title(f'{var.upper()} probabilities')
    ax.plot(x, ds[f'{reg}_{var}_MCS_core_prob_data'].values, 'r-', label='MCS core')
    ax.plot(x, ds[f'{reg}_{var}_MCS_shield_prob_data'].values, 'r--', label='MCS shield')
    ax.plot(x, ds[f'{reg}_{var}_cloud_core_prob_data'].values, 'b-', label='cloud core')
    ax.plot(x, ds[f'{reg}_{var}_cloud_shield_prob_data'].values, 'b--', label='cloud shield')
    ax.plot(x, ds[f'{reg}_{var}_env_prob_data'].values, 'k-', label='env')
    # ax.legend()


def plot_hist_probs_mode(ds, mode, ax=None, reg='all', var='cape'):
    if var == 'vertically_integrated_moisture_flux_div':
        x = -ds[f'{var}_hist_mids'].values * 1e4
    else:
        x = ds[f'{var}_hist_mids'].values
    # ax.set_title(f'{var.upper()} probabilities')
    ax.plot(x, ds[f'{reg}_{var}_MCS_core_prob_{mode}_data'].values, 'r-', label='MCS core')
    ax.plot(x, ds[f'{reg}_{var}_MCS_shield_prob_{mode}_data'].values, 'r--', label='MCS shield')
    ax.plot(x, ds[f'{reg}_{var}_cloud_core_prob_{mode}_data'].values, 'b-', label='cloud core')
    ax.plot(x, ds[f'{reg}_{var}_cloud_shield_prob_{mode}_data'].values, 'b--', label='cloud shield')
    ax.plot(x, ds[f'{reg}_{var}_env_prob_{mode}_data'].values, 'k-', label='env')
    # ax.legend()


def plot_hists_for_var(ds, var):
    xlim_ylim_title = {
        'cape': ((0, 2500), (0, 0.0014), 'CAPE {reg}'),
        'tcwv': ((None, None), (0, 0.08), 'TCWV {reg}'),
        'shear_0': ((None, None), (None, None), 'LLS {reg}'),
        'shear_1': ((None, None), (None, None), 'L2MS {reg}'),
        'shear_2': ((None, None), (None, None), 'M2HS {reg}'),
        'shear_3': ((None, None), (None, None), 'DS {reg}'),
        'vertically_integrated_moisture_flux_div': ((None, None), (None, None), 'VIMFD {reg}'),
        'RHlow': ((None, None), (None, None), 'RHlow {reg}'),
        'RHmid': ((None, None), (None, None), 'RHmid {reg}'),
        'theta_e_mid': ((None, None), (None, None), r'$\theta_e$ {reg}'),
        'delta_3h_cape': ((None, None), (None, None), r'\Delta 3h CAPE {reg}'),
        'delta_3h_tcwv': ((None, None), (None, None), r'\Delta 3h TCWV {reg}'),
    }
    fig, axes = plt.subplots(2, 3, sharex=True)
    fig.set_size_inches((20, 10))

    for ax, reg in zip(axes[0], ['all', 'land', 'ocean']):
        plot_hist(ds, ax=ax, reg=reg, var=var, log=False)
        xlim, ylim, title = xlim_ylim_title[var]
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(title.format(reg=reg))

    for ax, reg in zip(axes[1], ['all', 'land', 'ocean']):
        plot_hist_probs(ds, reg=reg, var=var, ax=ax)
        xlim, ylim, title = xlim_ylim_title[var]
        ax.set_xlim(xlim)
        ax.set_ylim((0, 1))
        ax.set_title(title.format(reg=reg))

    return fig, axes


def plot_combined_hists_for_var(ax0, ax1, ds, var, show_legends=True):
    xlim_ylim_title = {
        'cape': ((0, 2500), (0, 0.0014), 'CAPE'),
        'tcwv': ((0, 80), (0, 0.08), 'TCWV'),
        'shear_0': ((0, 40), (None, None), 'LLS'),
        'shear_1': ((0, 40), (None, None), 'L2MS'),
        'shear_2': ((0, 40), (None, None), 'M2HS'),
        'shear_3': ((None, None), (None, None), 'DS'),
        'RHlow': ((None, None), (None, None), 'RHlow'),
        'RHmid': ((None, None), (None, None), 'RHmid'),
        'theta_e_mid': ((None, None), (None, None), r'$\theta_e$'),
        'vertically_integrated_moisture_flux_div': ((None, None), (None, None), 'VIMFD'),
        'delta_3h_cape': ((None, None), (None, None), r'\Delta 3h CAPE'),
        'delta_3h_tcwv': ((None, None), (None, None), r'\Delta 3h TCWV'),
    }

    plot_hist(ds, ax=ax0, reg='all', var=var, log=False)
    xlim, ylim, title = xlim_ylim_title[var]
    ax0.set_xlim(xlim)
    ax0.set_ylim(ylim)
    # ax.set_title(title.format(reg=reg))

    plot_hist_probs(ds, reg='all', var=var, ax=ax1)

    xlim, ylim, title = xlim_ylim_title[var]
    ax1.set_xlim(xlim)
    ax1.set_ylim((0, 1))
    # ax1.set_title(title.format(reg=reg))


def plot_combined_hists_for_var_mode(ax0, ax1, ds, var, mode, show_legends=True):
    xlim_ylim_title = {
        'cape': ((0, 2500), (0, 0.0014), 'CAPE'),
        'tcwv': ((0, 80), (0, 0.08), 'TCWV'),
        'shear_0': ((0, 40), (None, None), 'LLS'),
        'shear_1': ((0, 40), (None, None), 'L2MS'),
        'shear_2': ((0, 40), (None, None), 'M2HS'),
        'shear_3': ((None, None), (None, None), 'DS'),
        'RHlow': ((None, None), (None, None), 'RHlow'),
        'RHmid': ((None, None), (None, None), 'RHmid'),
        'theta_e_mid': ((None, None), (None, None), r'$\theta_e$'),
        'vertically_integrated_moisture_flux_div': ((None, None), (None, None), 'VIMFD'),
        'delta_3h_cape': ((None, None), (None, None), r'\Delta 3h CAPE'),
        'delta_3h_tcwv': ((None, None), (None, None), r'\Delta 3h TCWV'),
    }

    plot_hist_mode(ds, mode, ax=ax0, reg='all', var=var, log=False)
    xlim, ylim, title = xlim_ylim_title[var]
    ax0.set_xlim(xlim)
    ax0.set_ylim(ylim)
    # ax.set_title(title.format(reg=reg))

    plot_hist_probs_mode(ds, mode, reg='all', var=var, ax=ax1)

    xlim, ylim, title = xlim_ylim_title[var]
    ax1.set_xlim(xlim)
    ax1.set_ylim((0, 1))
    # ax1.set_title(title.format(reg=reg))


class GenBootstrapDataForConditionalERA5Hist(TaskRule):
    @staticmethod
    def rule_inputs(e5vars, bootstrap, nbootstrap):
        inputs = {
            f'hist_{year}_{month}': fmtp(cu.FMT_PATH_COND_HIST_HOURLY, year=year, month=month, core_method='tb')
            for year in cu.YEARS
            for month in cu.MONTHS
        }
        return inputs

    rule_outputs = {
        'bootstrap_data': (PATHS['figdir'] / 'fig_data' /
                           'combined_hist_{e5vars}_bootstrap-{bootstrap}_nbootstrap-{nbootstrap:05d}.nc')
    }

    var_matrix = {
        'e5vars': [
            'cape-tcwv-vertically_integrated_moisture_flux_div',
            # 'shear_0-shear_1-shear_2',
            # 'shear_0-shear_1-shear_3',
            # 'RHlow-RHmid-theta_e_mid',
        ],
        # First task has no bootstrapping.
        # Subsequent 100 tasks split up the bootstrappint (10000 in total).
        ('bootstrap', 'nbootstrap'): [(False, 0)] + [(True, i) for i in list(range(0, 10000, 100))],
    }

    def rule_run(self):

        e5vars = self.e5vars.split('-')
        mcs_regs = ['MCS_core', 'MCS_shield', 'cloud_core', 'cloud_shield', 'env']

        # Need to get list of vars so I know which ones to drop.
        # Is there a nicer way of doing this?
        ds_one = xr.open_mfdataset(list(self.inputs.values())[0])
        keep_vars = set([
            f'all_{var}_{mcs_reg}'
            for var in e5vars
            for mcs_reg in mcs_regs
        ])
        drop_vars = [
            k
            for k in ds_one.data_vars.keys()
            if k not in keep_vars
        ]
        with xr.open_mfdataset(list(self.inputs.values()), drop_variables=drop_vars) as ds:
            ds.load()

        coords = {k: ds[k] for k in ds.mean(dim='time').coords.keys()}
        if not self.bootstrap:
            coords['bootstrap_index'] = np.arange(1)
        else:
            coords['bootstrap_index'] = np.arange(self.nbootstrap, self.nbootstrap + 100)
        data_vars = {}
        for var in e5vars:
            for mcs_reg in mcs_regs:
                if not self.bootstrap:
                    blank_hist_data = np.zeros((1, 100))
                else:
                    blank_hist_data = np.zeros((100, 100))
                data_vars[f'all_{var}_{mcs_reg}_hist_data'] = (('bootstrap_index', f'{var}_hist_mid'), blank_hist_data.copy())
                data_vars[f'all_{var}_{mcs_reg}_prob_data'] = (('bootstrap_index', f'{var}_hist_mid'), blank_hist_data.copy())

        dsout = xr.Dataset(
            coords=coords,
            data_vars=data_vars,
        )

        def _gen_hist_data(var, ds, h):
            bins = ds[f'{var}_bins'].values
            width = bins[1] - bins[0]
            h_density = h / (h.sum() * width)
            return h_density

        def _gen_prob_data(var, ds, mcs_regs):
            assert len(mcs_regs) == 5
            counts = np.zeros((5, ds[f'all_{var}_MCS_core'].shape[1]))
            for i, mcs_reg in enumerate(mcs_regs):
                counts[i] = np.nansum(ds[f'all_{var}_{mcs_reg}'].values, axis=0)
            probs = counts / counts.sum(axis=0)[None, :]
            return probs

        if not self.bootstrap:
            N = 1
        else:
            N = 100
        for i in range(N):
            for var in e5vars:
                if not self.bootstrap:
                    print(var)
                    ds_bs = ds
                else:
                    print(var, 'bootstrap', i)
                    # Resample with replacement.
                    bs_idx = np.sort(np.random.randint(0, len(ds.time), len(ds.time)))
                    ds_bs = ds.isel(time=bs_idx)

                prob_data = _gen_prob_data(var, ds_bs, mcs_regs)
                for j, mcs_reg in enumerate(mcs_regs):
                    dsout[f'all_{var}_{mcs_reg}_hist_data'][i] = _gen_hist_data(
                        var,
                        ds_bs,
                        np.nansum(ds_bs[f'all_{var}_{mcs_reg}'].values, axis=0),
                    )
                    dsout[f'all_{var}_{mcs_reg}_prob_data'][i] = prob_data[j]

        cu.to_netcdf_tmp_then_copy(dsout, self.outputs['bootstrap_data'])


class GenDataForConditionalERA5HistSeasonalDC(TaskRule):
    @staticmethod
    def rule_inputs(e5vars):
        # Note, the DC data can be used to generate seasonal data as well.
        # This is because it is the same as the hourly data when viewed over the course of
        # one complete day (or season).
        inputs = {
            f'hist_{year}_{month}': fmtp(cu.FMT_PATH_COND_HIST_DC, year=year, month=month, core_method='tb')
            for year in cu.YEARS
            for month in cu.MONTHS
        }
        return inputs

    rule_outputs = {
        'season_dc_data': (PATHS['figdir'] / 'fig_data' /
                             'combined_hist_{e5vars}_season_dc.nc')
    }

    var_matrix = {
        'e5vars': [
            'cape-tcwv-vertically_integrated_moisture_flux_div',
            # 'shear_0-shear_1-shear_2',
            # 'shear_0-shear_1-shear_3',
            # 'RHlow-RHmid-theta_e_mid',
        ],
    }

    def rule_run(self):

        e5vars = self.e5vars.split('-')
        mcs_regs = ['MCS_core', 'MCS_shield', 'cloud_core', 'cloud_shield', 'env']
        # Need to get list of vars so I know which ones to drop.
        # Is there a nicer way of doing this?
        ds_one = xr.open_mfdataset(list(self.inputs.values())[0])
        keep_vars = set([
            f'all_{var}_{mcs_reg}'
            for var in e5vars
            for mcs_reg in mcs_regs
        ])
        drop_vars = [
            k
            for k in ds_one.data_vars.keys()
            if k not in keep_vars
        ]
        with xr.open_mfdataset(list(self.inputs.values()), drop_variables=drop_vars) as ds:
            ds.load()
        print('Loaded data')

        coords = {k: ds[k] for k in ds.mean(dim='time').coords.keys()}
        coords['season'] = np.arange(4)
        coords['diurnal_cycle'] = np.arange(24)

        data_vars = {}
        for var in e5vars:
            for mcs_reg in mcs_regs:
                blank_hist_season_data = np.zeros((4, 100))
                blank_hist_dc_data = np.zeros((24, 100))
                data_vars[f'all_{var}_{mcs_reg}_hist_season_data'] = (
                    ('season', f'{var}_hist_mid'), blank_hist_season_data.copy()
                )
                data_vars[f'all_{var}_{mcs_reg}_prob_season_data'] = (
                    ('season', f'{var}_hist_mid'), blank_hist_season_data.copy()
                )
                data_vars[f'all_{var}_{mcs_reg}_hist_diurnal_cycle_data'] = (
                    ('diurnal_cycle', f'{var}_hist_mid'), blank_hist_dc_data.copy()
                )
                data_vars[f'all_{var}_{mcs_reg}_prob_diurnal_cycle_data'] = (
                    ('diurnal_cycle', f'{var}_hist_mid'), blank_hist_dc_data.copy()
                )

        dsout = xr.Dataset(
            coords=coords,
            data_vars=data_vars,
        )

        # Generate time filters that can be used to subset the data
        # in ds by either season or diurnal cycle.
        times = pd.DatetimeIndex(ds.time)
        time_filters = {}
        month_lookup = {
            'djf': [12, 1, 2],
            'mam': [3, 4, 5],
            'jja': [6, 7, 8],
            'son': [9, 10, 11],
        }
        for i, season in enumerate(['djf', 'mam', 'jja', 'son']):
            time_filters[('season', i)] = times.month.isin(month_lookup[season])
        for h in range(24):
            time_filters[('diurnal_cycle', h)] = times.hour == h

        def _gen_hist_data(var, ds, h):
            bins = ds[f'{var}_bins'].values
            width = bins[1] - bins[0]
            h_density = h / (h.sum() * width)
            return h_density

        def _gen_prob_data(var, ds, mcs_regs):
            assert len(mcs_regs) == 5
            counts = np.zeros((5, ds[f'all_{var}_MCS_core'].shape[1]))
            for i, mcs_reg in enumerate(mcs_regs):
                counts[i] = np.nansum(ds[f'all_{var}_{mcs_reg}'].values, axis=0)
            probs = counts / counts.sum(axis=0)[None, :]
            return probs

        for var in e5vars:
            for key, time_filter in time_filters.items():
                filter_type, filter_idx = key
                # Apply filter.
                ds_filtered = ds.isel(time=time_filter)
                prob_data = _gen_prob_data(var, ds_filtered, mcs_regs)
                for j, mcs_reg in enumerate(mcs_regs):
                    hist_key = f'all_{var}_{mcs_reg}_hist_{filter_type}_data'
                    prob_key = f'all_{var}_{mcs_reg}_prob_{filter_type}_data'
                    dsout[hist_key][filter_idx] = _gen_hist_data(
                        var,
                        ds_filtered,
                        np.nansum(ds_filtered[f'all_{var}_{mcs_reg}'].values, axis=0),
                    )
                    dsout[prob_key][filter_idx] = prob_data[j]

        cu.to_netcdf_tmp_then_copy(dsout, self.outputs['season_dc_data'])


class FigPlotCombineVarConditionalERA5HistSeasonalDC(TaskRule):
    rule_inputs = GenDataForConditionalERA5HistSeasonalDC.rule_outputs
    rule_outputs = {
        'fig_season': (PATHS['figdir'] / 'mcs_env_cond_figs' / 'combined_hist_{e5vars}_season.png'),
        'fig_diurnal_cycle': (PATHS['figdir'] / 'mcs_env_cond_figs' / 'combined_hist_{e5vars}_dc.png'),
    }

    depends_on = [get_labels, plot_hist_mode, plot_hist_probs_mode, plot_combined_hists_for_var_mode]

    var_matrix = {
        'e5vars': [
            'cape-tcwv-vertically_integrated_moisture_flux_div',
            # 'shear_0-shear_1-shear_2',
            # 'shear_0-shear_1-shear_3',
            # 'RHlow-RHmid-theta_e_mid',
        ],
    }

    def rule_run(self):

        e5vars = self.e5vars.split('-')

        with xr.open_mfdataset(self.inputs['season_dc_data']) as ds:
            ds.load()

        for mode in ['season', 'diurnal_cycle']:
            indices = np.arange(len(ds[mode]))

            fig, axes = plt.subplots(2, 3, sharex='col')
            fig.set_size_inches((15, 8))

            for i in indices:
                for (ax0, ax1), var in zip(axes.T, e5vars):
                    print(var, mode, i)
                    kwargs = {mode: i}
                    plot_combined_hists_for_var_mode(ax0, ax1, ds.isel(**kwargs), var, mode, False)

                    if var == 'vertically_integrated_moisture_flux_div':
                        ax1.set_xlabel('MFC (10$^{-4}$ kg m$^{-2}$ s$^{-1}$)')
                    else:
                        ax1.set_xlabel(get_labels(var))

            plt.subplots_adjust(left=0.05, right=0.95, bottom=0.08, top=0.95, wspace=0.2, hspace=0.2)
            plt.savefig(self.outputs[f'fig_{mode}'])


class FigPlotVarConditionalERA5HistSeasonalDC(TaskRule):
    rule_inputs = GenDataForConditionalERA5HistSeasonalDC.rule_outputs
    @staticmethod
    def rule_outputs(e5vars):
        e5vars = e5vars.split('-')
        rule_outputs = {}
        for var in e5vars:
            rule_outputs.update({
                f'fig_season_{var}': (PATHS['figdir'] / 'mcs_env_cond_figs' / f'hist_{var}_season.png'),
                f'fig_diurnal_cycle_{var}': (PATHS['figdir'] / 'mcs_env_cond_figs' / f'hist_{var}_dc.png'),
            })
        return rule_outputs

    depends_on = [get_labels, plot_hist_mode, plot_hist_probs_mode, plot_combined_hists_for_var_mode]

    var_matrix = {
        'e5vars': [
            'cape-tcwv-vertically_integrated_moisture_flux_div',
            # 'shear_0-shear_1-shear_2',
            # 'shear_0-shear_1-shear_3',
            # 'RHlow-RHmid-theta_e_mid',
        ],
    }

    def rule_run(self):

        e5vars = self.e5vars.split('-')
        mcs_regs = ['MCS_core', 'MCS_shield', 'cloud_core', 'cloud_shield', 'env']

        with xr.open_mfdataset(self.inputs['season_dc_data']) as ds:
            ds.load()

        cmap = mpl.colormaps['twilight_shifted']
        xlim_ylim_title = {
            'cape': ((0, 2500), (0, 0.0014), 'CAPE'),
            'tcwv': ((0, 80), (0, 0.08), 'TCWV'),
            'shear_0': ((0, 40), (None, None), 'LLS'),
            'shear_1': ((0, 40), (None, None), 'L2MS'),
            'shear_2': ((0, 40), (None, None), 'M2HS'),
            'shear_3': ((None, None), (None, None), 'DS'),
            'RHlow': ((None, None), (None, None), 'RHlow'),
            'RHmid': ((None, None), (None, None), 'RHmid'),
            'theta_e_mid': ((None, None), (None, None), r'$\theta_e$'),
            'vertically_integrated_moisture_flux_div': ((None, None), (None, None), 'VIMFD'),
            'delta_3h_cape': ((None, None), (None, None), r'\Delta 3h CAPE'),
            'delta_3h_tcwv': ((None, None), (None, None), r'\Delta 3h TCWV'),
        }

        for mode in ['season', 'diurnal_cycle']:
            indices = np.arange(len(ds[mode]))

            for var in e5vars:
                fig, axes = plt.subplots(1, 6, gridspec_kw={'width_ratios':[1, 1, 1, 1, 1, 0.05]})
                fig.set_size_inches((15, 4))

                for i in indices:
                    print(var, mode, i)

                    kwargs = {mode: i}
                    ds_filtered = ds.isel(**kwargs)

                    if var == 'vertically_integrated_moisture_flux_div':
                        x = -ds[f'{var}_hist_mids'].values * 1e4
                    else:
                        x = ds[f'{var}_hist_mids'].values
                    reg = 'all'
                    c = cmap(i / len(indices))
                    for ax, mcs_reg in zip(axes, mcs_regs):
                        ax.set_title(mcs_reg.replace('_', ' '))
                        var_key = f'{reg}_{var}_{mcs_reg}_prob_{mode}_data'
                        ax.plot(x, ds_filtered[var_key].values, color=c, label=mcs_reg)
                        if var == 'vertically_integrated_moisture_flux_div':
                            ax.set_xlabel('MFC (10$^{-4}$ kg m$^{-2}$ s$^{-1}$)')
                        else:
                            ax.set_xlabel(get_labels(var))
                        ax.set_ylim((0, 1))
                xlim, ylim, title = xlim_ylim_title[var]
                for ax in axes:
                    ax.set_xlim(xlim)

                if mode == 'diurnal_cycle':
                    norm = mpl.colors.Normalize(vmin=0, vmax=24)
                    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                    sm.set_array([])
                    plt.colorbar(sm, cax=axes[-1],
                                 ticks=np.arange(24), boundaries=np.linspace(-0.5, 23.5, 25))
                elif mode == 'season':
                    norm = mpl.colors.Normalize(vmin=0, vmax=4)
                    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                    sm.set_array([])
                    labels = ['DJF', 'MAM', 'JJA', 'SON']
                    cb = plt.colorbar(sm, cax=axes[-1],
                                      ticks=np.arange(4), boundaries=np.linspace(-0.5, 3.5, 5))
                    cb.set_ticklabels(labels)

                plt.subplots_adjust(left=0.05, right=0.95, bottom=0.15, top=0.92, wspace=0.2, hspace=0.2)
                plt.savefig(self.outputs[f'fig_{mode}_{var}'])


class FigPlotCombineVarConditionalERA5HistBootstrap(TaskRule):
    @staticmethod
    def rule_inputs(e5vars):
        inputs = {
            'no_bootstrap': fmtp(GenBootstrapDataForConditionalERA5Hist.rule_outputs['bootstrap_data'],
                                 e5vars=e5vars, bootstrap=False, nbootstrap=0)
        }
        inputs.update({
            f'bootstrap_{i:05d}': fmtp(GenBootstrapDataForConditionalERA5Hist.rule_outputs['bootstrap_data'],
                                       e5vars=e5vars, bootstrap=True, nbootstrap=i)
            for i in range(0, 10000, 100)
        })
        return inputs

    rule_outputs = {
        'fig': (PATHS['figdir'] / 'mcs_env_cond_figs' / 'combined_hist_{e5vars}_bootstrap.png')
    }

    depends_on = [get_labels, plot_hist, plot_hist_probs, plot_combined_hists_for_var]

    var_matrix = {
        'e5vars': [
            'cape-tcwv-vertically_integrated_moisture_flux_div',
            'shear_0-shear_1-shear_2',
            'shear_0-shear_1-shear_3',
            'RHlow-RHmid-theta_e_mid',
        ],
    }

    def rule_run(self):

        print(self.inputs)
        fig, axes = plt.subplots(2, 3, sharex='col')
        fig.set_size_inches((15, 8))
        e5vars = self.e5vars.split('-')

        with (
            xr.open_mfdataset(self.inputs['no_bootstrap']) as ds,
            xr.open_mfdataset([self.inputs[k] for k in self.inputs.keys() if k != 'no_bootstrap']) as ds_bs
        ):
            ds.load()
            ds_bs.load()

        for (ax0, ax1), var in zip(axes.T, e5vars):
            print(var)
            plot_combined_hists_for_var(ax0, ax1, ds.isel(bootstrap_index=0), var)

            if var == 'vertically_integrated_moisture_flux_div':
                ax1.set_xlabel('MFC (10$^{-4}$ kg m$^{-2}$ s$^{-1}$)')
            else:
                ax1.set_xlabel(get_labels(var))

        for i in range(10000):
            for (ax0, ax1), var in zip(axes.T, e5vars):
                print(var, 'bootstrap', i)
                plot_combined_hists_for_var(ax0, ax1, ds_bs.isel(bootstrap_index=i), var, False)

                if var == 'vertically_integrated_moisture_flux_div':
                    ax1.set_xlabel('MFC (10$^{-4}$ kg m$^{-2}$ s$^{-1}$)')
                else:
                    ax1.set_xlabel(get_labels(var))

        # axes[0, 0].legend()
        # for ax in axes[0]:
        #     ax.set_xticklabels([])
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.08, top=0.95, wspace=0.2, hspace=0.2)
        plt.savefig(self.outputs['fig'])


def plot_convection_hourly_hists(ds, var):
    fig, (ax, ax2) = plt.subplots(2, 1, sharex=True)

    if var == 'vertically_integrated_moisture_flux_div':
        ax.set_title('MFC')
    else:
        ax.set_title(var.upper())
    for lsreg in ['all', 'land', 'ocean']:
        d1 = ds[f'{lsreg}_{var}_MCS_core'].sum(dim='time').values
        d2 = ds[f'{lsreg}_{var}_cloud_core'].sum(dim='time').values
        d3 = ds[f'{lsreg}_{var}_MCS_shield'].sum(dim='time').values
        d4 = ds[f'{lsreg}_{var}_cloud_shield'].sum(dim='time').values
        d5 = ds[f'{lsreg}_{var}_env'].sum(dim='time').values
        dt = d1 + d2 + d3 + d4 + d5

        with np.errstate(invalid='ignore', divide='ignore'):
            d = d1 / (d1 + d2)

        if var == 'vertically_integrated_moisture_flux_div':
            x = ds[f'{var}_hist_mids'].values * -1e4
        else:
            x = ds[f'{var}_hist_mids'].values

        p = ax.plot(x, d, label=lsreg)
        ax2.plot(x, dt / dt.sum(), label=lsreg, color=p[0].get_color(), linestyle='--')

        if var == 'cape':
            plt.xlabel('J kg$^{-1}$')
            plt.xlim((0, 5000))
        elif var == 'tcwv':
            plt.xlabel('mm')
            plt.xlim((0, 100))
        elif var == 'vertically_integrated_moisture_flux_div':
            plt.xlabel('10$^{-4}$ kg m$^{-2}$ s$^{-1}$')
            plt.xlim((-0.002 * 1e4, 0.002 * 1e4))
        elif var.startswith('shear'):
            plt.xlabel('m s$^{-1}$')
    ax.set_ylabel('p(MCS conv|conv)')
    ax2.set_ylabel('pdf')

    ax.set_ylim((0, 1))
    ax2.set_ylim((0, None))

    ax.legend()


class GenDataForCombineConvectionConditionalERA5Hist(TaskRule):
    @staticmethod
    def rule_inputs(e5var, years):
        inputs = {
            f'hist_{year}_{month}': fmtp(cu.FMT_PATH_COND_HIST_DC, year=year, month=month, core_method='tb')
            for year in years
            for month in cu.MONTHS
        }
        return inputs

    @staticmethod
    def rule_outputs(e5var, years):
        ystr = f'{years[0]}-{years[-1]}'
        outputs = {
            'conv_data': (
                PATHS['figdir']
                / 'fig_data'
                / f'convection_yearly_hist_{e5var}_{ystr}_tb.nc'
            )
        }
        return outputs

    depends_on = [plot_convection_hourly_hists]

    var_matrix = {
        'years': [
            cu.YEARS,
            # [2020],
        ],
        'e5var':cu.EXTENDED_ERA5VARS,
    }

    def rule_run(self):

        var = self.e5var
        mcs_regs = ['MCS_core', 'MCS_shield', 'cloud_core', 'cloud_shield', 'env']
        print('open .nc')
        with xr.open_mfdataset(self.inputs.values()) as ds:
            print(var)

            coords = {k: ds[k] for k in ds.mean(dim='time').coords.keys()}
            coords['season'] = np.arange(4)
            coords['diurnal_cycle'] = np.arange(24)

            data_vars = {}
            for lsreg in ['all', 'land', 'ocean']:
                blank_conv_data = np.zeros(100)
                blank_conv_season_data = np.zeros((4, 100))
                blank_conv_dc_data = np.zeros((24, 100))
                data_vars[f'{lsreg}_{var}_conv_data'] = ((f'{var}_hist_mid'), blank_conv_data.copy())
                data_vars[f'{lsreg}_{var}_total_data'] = ((f'{var}_hist_mid'), blank_conv_data.copy())
                data_vars[f'{lsreg}_{var}_conv_season_data'] = (('season', f'{var}_hist_mid'), blank_conv_season_data.copy())
                data_vars[f'{lsreg}_{var}_total_season_data'] = (('season', f'{var}_hist_mid'), blank_conv_season_data.copy())
                data_vars[f'{lsreg}_{var}_conv_diurnal_cycle_data'] = (('diurnal_cycle', f'{var}_hist_mid'), blank_conv_dc_data.copy())
                data_vars[f'{lsreg}_{var}_total_diurnal_cycle_data'] = (('diurnal_cycle', f'{var}_hist_mid'), blank_conv_dc_data.copy())

            dsout = xr.Dataset(
                coords=coords,
                data_vars=data_vars,
            )

            # Generate time filters that can be used to subset the data
            # in ds by either full, season or diurnal cycle.
            times = pd.DatetimeIndex(ds.time)
            time_filters = {}
            month_lookup = {
                'djf': [12, 1, 2],
                'mam': [3, 4, 5],
                'jja': [6, 7, 8],
                'son': [9, 10, 11],
            }
            time_filters[('full', 0)] = times.year > 0  # all true!
            for i, season in enumerate(['djf', 'mam', 'jja', 'son']):
                time_filters[('season', i)] = times.month.isin(month_lookup[season])
            for h in range(24):
                time_filters[('diurnal_cycle', h)] = times.hour == h

            print('load data')
            for lsreg, mcs_reg in product(['all', 'land', 'ocean'], mcs_regs):
                ds[f'{lsreg}_{var}_MCS_core'].load()

            print('calc conv data')
            for lsreg in ['all', 'land', 'ocean']:
                for key, time_filter in time_filters.items():
                    print(key)
                    filter_type, filter_idx = key
                    # Apply filter.
                    ds_filtered = ds.isel(time=time_filter)

                    d1 = ds_filtered[f'{lsreg}_{var}_MCS_core'].sum(dim='time').values
                    d2 = ds_filtered[f'{lsreg}_{var}_cloud_core'].sum(dim='time').values
                    d3 = ds_filtered[f'{lsreg}_{var}_MCS_shield'].sum(dim='time').values
                    d4 = ds_filtered[f'{lsreg}_{var}_cloud_shield'].sum(dim='time').values
                    d5 = ds_filtered[f'{lsreg}_{var}_env'].sum(dim='time').values
                    dt = d1 + d2 + d3 + d4 + d5

                    with np.errstate(invalid='ignore', divide='ignore'):
                        d = d1 / (d1 + d2)

                    if filter_type == 'full':
                        dsout[f'{lsreg}_{var}_conv_data'].values = d
                        dsout[f'{lsreg}_{var}_total_data'].values = dt / dt.sum()
                    else:
                        dsout[f'{lsreg}_{var}_conv_{filter_type}_data'].values[filter_idx] = d
                        dsout[f'{lsreg}_{var}_total_{filter_type}_data'].values[filter_idx] = dt / dt.sum()

            cu.to_netcdf_tmp_then_copy(dsout, self.outputs['conv_data'])


class FigCombineConvectionConditionalERA5Hist(TaskRule):
    @staticmethod
    def rule_inputs(e5var, years):
        return GenDataForCombineConvectionConditionalERA5Hist.rule_outputs(e5var, years)

    @staticmethod
    def rule_outputs(e5var, years):
        ystr = f'{years[0]}-{years[-1]}'
        outputs = {
            f'fig_{mode}': (
                PATHS['figdir']
                / 'mcs_env_cond_figs'
                / f'convection_yearly_hist_{e5var}_{ystr}_{mode}_tb.png'
            )
            for mode in ['full', 'diurnal_cycle', 'season']
        }
        return outputs

    depends_on = [plot_convection_hourly_hists]

    # var_matrix = {'core_method': ['tb', 'precip']}
    var_matrix = {
        'years': [
            cu.YEARS,
            # [2020],
        ],
        'e5var':cu.EXTENDED_ERA5VARS,
    }

    def rule_run(self):

        var = self.e5var

        with xr.open_dataset(self.inputs['conv_data']) as ds:
            print(var)
            fig, (ax, ax2) = plt.subplots(2, 1, sharex=True)
            if var == 'vertically_integrated_moisture_flux_div':
                ax.set_title('MFC')
            else:
                ax.set_title(var.upper())
            for lsreg in ['all', 'land', 'ocean']:
                d = ds[f'{lsreg}_{var}_conv_data'].values
                dt = ds[f'{lsreg}_{var}_total_data'].values

                if var == 'vertically_integrated_moisture_flux_div':
                    x = ds[f'{var}_hist_mids'].values * -1e4
                else:
                    x = ds[f'{var}_hist_mids'].values

                p = ax.plot(x, d, label=lsreg)
                ax2.plot(x, dt, label=lsreg, color=p[0].get_color(), linestyle='--')

                if var == 'cape':
                    plt.xlabel('J kg$^{-1}$')
                    plt.xlim((0, 5000))
                elif var == 'tcwv':
                    plt.xlabel('mm')
                    plt.xlim((0, 100))
                elif var == 'vertically_integrated_moisture_flux_div':
                    plt.xlabel('10$^{-4}$ kg m$^{-2}$ s$^{-1}$')
                    plt.xlim((-0.002 * 1e4, 0.002 * 1e4))
                elif var.startswith('shear'):
                    plt.xlabel('m s$^{-1}$')
            ax.set_ylabel('p(MCS conv|conv)')
            ax2.set_ylabel('pdf')
            ax.set_ylim((0, 1))
            ax.set_xlim(x[[0, -1]])
            ax2.set_xlim(x[[0, -1]])
            plt.legend()
            plt.savefig(self.outputs['fig_full'])

            cmap = mpl.colormaps['twilight_shifted']
            for mode in ['diurnal_cycle', 'season']:
                fig, axes = plt.subplot_mosaic(
                    [['a', 'cb'], ['b', 'cb']],
                    layout='constrained',
                    gridspec_kw={'width_ratios':[1, 0.05]}
                )
                ax = axes['a']
                ax2 = axes['b']
                cax = axes['cb']
                lsreg = 'all'
                if var == 'vertically_integrated_moisture_flux_div':
                    ax.set_title('MFC')
                else:
                    ax.set_title(var.upper())

                d = ds[f'{lsreg}_{var}_conv_{mode}_data'].values
                dt = ds[f'{lsreg}_{var}_total_{mode}_data'].values
                for i in ds[mode].values:
                    print(i)
                    if var == 'vertically_integrated_moisture_flux_div':
                        x = ds[f'{var}_hist_mids'].values * -1e4
                    else:
                        x = ds[f'{var}_hist_mids'].values

                    c = cmap(i / len(ds[mode].values))
                    p = ax.plot(x, d[i], label=lsreg, color=c)
                    ax2.plot(x, dt[i], label=lsreg, color=c, linestyle='--')

                    if var == 'cape':
                        plt.xlabel('J kg$^{-1}$')
                        plt.xlim((0, 5000))
                    elif var == 'tcwv':
                        plt.xlabel('mm')
                        plt.xlim((0, 100))
                    elif var == 'vertically_integrated_moisture_flux_div':
                        plt.xlabel('10$^{-4}$ kg m$^{-2}$ s$^{-1}$')
                        plt.xlim((-0.002 * 1e4, 0.002 * 1e4))
                    elif var.startswith('shear'):
                        plt.xlabel('m s$^{-1}$')
                ax.set_ylabel('p(MCS conv|conv)')
                ax2.set_ylabel('pdf')
                ax.set_ylim((0, 1))
                ax.set_xlim(x[[0, -1]])
                ax2.set_xlim(x[[0, -1]])

                if mode == 'diurnal_cycle':
                    norm = mpl.colors.Normalize(vmin=0, vmax=24)
                    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                    sm.set_array([])
                    plt.colorbar(sm, cax=cax,
                                 ticks=np.arange(24), boundaries=np.linspace(-0.5, 23.5, 25))
                elif mode == 'season':
                    norm = mpl.colors.Normalize(vmin=0, vmax=4)
                    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                    sm.set_array([])
                    labels = ['DJF', 'MAM', 'JJA', 'SON']
                    cb = plt.colorbar(sm, cax=cax,
                                      ticks=np.arange(4), boundaries=np.linspace(-0.5, 3.5, 5))
                    cb.set_ticklabels(labels)
                plt.savefig(self.outputs[f'fig_{mode}'])



def plot_gridpoint_hists(ds, var, lsmask):
    plt.figure()
    plt.title(var.upper())
    for lsreg in ['all', 'land', 'ocean']:
        d1 = ds[f'{var}_MCS_core'].values
        d2 = ds[f'{var}_cloud_core'].values
        d1 = (d1 * lsmask[lsreg][:, :, None]).sum(axis=(0, 1))
        d2 = (d2 * lsmask[lsreg][:, :, None]).sum(axis=(0, 1))
        with np.errstate(invalid='ignore', divide='ignore'):
            d = d1 / (d1 + d2)
        plt.plot(ds[f'{var}_hist_mids'].values, d, label=lsreg)
        if var == 'cape':
            plt.xlabel('J kg$^{-1}$')
        elif var == 'tcwv':
            plt.xlabel('mm')
        elif var == 'vimfd':
            plt.xlabel('kg m$^{-2}$ s$^{-1}$')
        elif var.startswith('shear'):
            plt.xlabel('m s$^{-1}$')

    plt.ylabel('p(MCS conv|conv)')
    plt.ylim((0, 1))
    plt.legend()


def plot_gridpoint_prob_dist(ds, var, pmin=25, pmax=75, step=16):
    plt.figure()
    plt.title(var.upper())
    d1 = ds[f'{var}_MCS_core'].coarsen(latitude=step, longitude=step, boundary='trim').sum().values
    d2 = ds[f'{var}_cloud_core'].coarsen(latitude=step, longitude=step, boundary='trim').sum().values
    with np.errstate(invalid='ignore', divide='ignore'):
        d = d1 / (d1 + d2)
    dmin, d50, dmax = np.nanpercentile(d, [pmin, 50, pmax], axis=(0, 1))

    plt.plot(ds[f'{var}_hist_mids'].values, d50, 'b-')
    plt.plot(ds[f'{var}_hist_mids'].values, dmin, 'b-', alpha=0.7)
    plt.plot(ds[f'{var}_hist_mids'].values, dmax, 'b-', alpha=0.7)
    plt.fill_between(ds[f'{var}_hist_mids'].values, dmin, dmax, color='b', alpha=0.3)


def plot_gridpoint_2d_prob_dist(ds, var, pmin=25, pmax=75, step=16):
    nbin = 21

    plt.figure()
    plt.title(var.upper())
    h = np.zeros((len(ds[f'{var}_hist_mids'].values), nbin - 1))
    d1 = ds[f'{var}_MCS_core'].coarsen(latitude=step, longitude=step, boundary='trim').sum().values
    d2 = ds[f'{var}_cloud_core'].coarsen(latitude=step, longitude=step, boundary='trim').sum().values
    with np.errstate(invalid='ignore', divide='ignore'):
        d = d1 / (d1 + d2)
    for i in range(d.shape[2]):
        h[i, :] = np.histogram(d[:, :, i], bins=np.linspace(0, 1, nbin))[0]
    h = h.reshape(-1, len(h) // (nbin - 1), nbin - 1).sum(axis=1)
    # TODO: Calc properly.
    if var == 'cape':
        aspect = 5000
    elif var == 'tcwv':
        aspect = 100
    elif var == 'vertically_integrated_moisture_flux_div':
        aspect = 1 / 300
    elif var.startswith('shear'):
        aspect = 100
    else:
        aspect = 100
    plt.imshow(
        h.T,
        origin='lower',
        extent=(ds[f'{var}_hist_mids'].values[0], ds[f'{var}_hist_mids'].values[-1], 0, 1),
        aspect=aspect,
    )
    dmin, d50, dmax = np.nanpercentile(d, [pmin, 50, pmax], axis=(0, 1))

    plt.plot(ds[f'{var}_hist_mids'].values, d50, 'r-')
    plt.plot(ds[f'{var}_hist_mids'].values, dmin, 'r--', alpha=0.7)
    plt.plot(ds[f'{var}_hist_mids'].values, dmax, 'r--', alpha=0.7)


def plot_gridpoint_lat_band_hist(ds, var, latstep=20):
    plt.figure()
    plt.title(var.upper())
    for latstart in range(60, -51, -latstep):
        d1 = ds[f'{var}_MCS_core'].sel(latitude=slice(latstart, latstart - latstep + 0.25)).values.sum(axis=(0, 1))
        d2 = ds[f'{var}_cloud_core'].sel(latitude=slice(latstart, latstart - latstep + 0.25)).values.sum(axis=(0, 1))
        with np.errstate(invalid='ignore', divide='ignore'):
            d = d1 / (d1 + d2)
        plt.plot(ds[f'{var}_hist_mids'].values, d, label=f'{latstart - latstep / 2} lat')
    if var == 'cape':
        plt.xlabel('J/kg')
    elif var == 'tcwv':
        plt.xlabel('mm')

    plt.ylabel('p(MCS conv|conv)')
    plt.ylim((0, 1))
    plt.legend()


class PlotGridpointConditionalERA5Hist(TaskRule):
    enabled = False
    @staticmethod
    def rule_inputs(year):
        inputs = {
            f'hist_{year}': fmtp(cu.FMT_PATH_COMBINED_COND_HIST_GRIDPOINT, year=year),
            'ERA5_land_sea_mask': cu.PATH_ERA5_LAND_SEA_MASK,
        }
        return inputs

    rule_outputs = {
        f'fig_{output}_{var}': (
            PATHS['figdir'] / 'conditional_era5_histograms' / f'gridpoint_{output}_yearly_hist_{var}_{{year}}.png'
        )
        for var in cu.EXTENDED_ERA5VARS
        for output in ['conv', 'conv_prob_dist', 'conv_2d_prob_dist', 'conv_lat_band']
    }

    depends_on = [
        cu.load_lsmask,
        plot_convection_hourly_hists,
        plot_gridpoint_prob_dist,
        plot_gridpoint_2d_prob_dist,
        plot_gridpoint_lat_band_hist,
    ]

    var_matrix = {'year': cu.YEARS}

    def rule_run(self):

        lsmask = cu.load_lsmask(self.inputs['ERA5_land_sea_mask'])
        with xr.open_dataset(self.inputs[f'hist_{self.year}']) as ds:
            ds.load()
            for var in cu.EXTENDED_ERA5VARS:
                print(var)
                plot_gridpoint_hists(ds, var, lsmask)
                plt.savefig(self.outputs[f'fig_conv_{var}'])

                plot_gridpoint_prob_dist(ds, var)
                plt.savefig(self.outputs[f'fig_conv_prob_dist_{var}'])

                plot_gridpoint_2d_prob_dist(ds, var)
                plt.savefig(self.outputs[f'fig_conv_2d_prob_dist_{var}'])

                plot_gridpoint_2d_prob_dist(ds, var)
                plt.savefig(self.outputs[f'fig_conv_2d_prob_dist_{var}'])

                plot_gridpoint_lat_band_hist(ds, var)
                plt.savefig(self.outputs[f'fig_conv_lat_band_{var}'])


def rmse(a, b):
    return np.sqrt(np.nanmean((a[None, None, :] - b) ** 2, axis=2))


def integral_diff(a, b, dx):
    return np.nansum(b - a[None, None, :], axis=2) * dx


def gen_rmse_integral_diff(ds, vars):
    step = 16
    dataarrays = []
    for var in vars:
        d1 = ds[f'{var}_MCS_core'].values.sum(axis=(0, 1))
        d2 = ds[f'{var}_cloud_core'].values.sum(axis=(0, 1))
        with np.errstate(invalid='ignore', divide='ignore'):
            d = d1 / (d1 + d2)
        da1 = ds[f'{var}_MCS_core'].coarsen(latitude=step, longitude=step, boundary='trim').sum()
        da2 = ds[f'{var}_cloud_core'].coarsen(latitude=step, longitude=step, boundary='trim').sum()

        d1 = da1.values
        d2 = da2.values
        with np.errstate(invalid='ignore', divide='ignore'):
            dc = d1 / (d1 + d2)

        da_rmse = xr.DataArray(
            rmse(d, dc),
            name=f'{var}_rmse',
            dims=['latitude', 'longitude'],
            coords=dict(
                latitude=da1.latitude,
                longitude=da1.longitude,
            ),
        )
        da_integral_diff = xr.DataArray(
            integral_diff(d, dc, 1),
            name=f'{var}_integral_diff',
            dims=['latitude', 'longitude'],
            coords=dict(
                latitude=da1.latitude,
                longitude=da1.longitude,
            ),
        )
        dataarrays.extend([da_rmse, da_integral_diff])
    return xr.merge([cu.xr_add_cyclic_point(da) for da in dataarrays])


def plot_global_rmse_bias(ds, var):
    fig, axes = plt.subplots(2, 1, subplot_kw=dict(projection=ccrs.PlateCarree()))
    fig.set_size_inches(20, 10)

    mean_rmse = np.nanmean(ds[f'{var}_rmse'].values)
    axes[0].set_title(f'Mean RMSE={mean_rmse}')
    im0 = axes[0].contourf(ds.longitude, ds.latitude, ds[f'{var}_rmse'], levels=np.linspace(0, 0.8, 9))
    if var == 'cape':
        levels = np.linspace(-50, 50, 11)
    elif var == 'tcwv':
        levels = np.linspace(-30, 30, 7)
    else:
        absmax = np.abs(ds[f'{var}_integral_diff'].values).max()
        levels = np.linspace(-absmax, absmax, 11)

    plt.colorbar(im0, ax=axes[0])

    mean_bias = np.nanmean(ds[f'{var}_integral_diff'].values)
    axes[1].set_title(f'Mean bias={mean_bias}')
    im1 = axes[1].contourf(ds.longitude, ds.latitude, ds[f'{var}_integral_diff'], levels=levels, cmap='bwr')
    plt.colorbar(im1, ax=axes[1])

    for ax in axes:
        ax.coastlines()


class PlotGridpointGlobal(TaskRule):
    enabled = False
    @staticmethod
    def rule_inputs(year):
        inputs = {
            f'hist_{year}': fmtp(cu.FMT_PATH_COMBINED_COND_HIST_GRIDPOINT, year=year),
            'ERA5_land_sea_mask': cu.PATH_ERA5_LAND_SEA_MASK,
        }
        return inputs

    rule_outputs = {
        f'fig_{var}': (
            PATHS['figdir'] / 'conditional_era5_histograms' / f'gridpoint_global_yearly_hist_{var}_{{year}}.png'
        )
        for var in cu.EXTENDED_ERA5VARS
    }

    depends_on = [
        gen_rmse_integral_diff,
        plot_global_rmse_bias,
    ]

    var_matrix = {'year': cu.YEARS}

    def rule_run(self):

        with xr.open_dataset(self.inputs[f'hist_{self.year}']) as ds:
            ds.load()
            ds_metrics = gen_rmse_integral_diff(ds, cu.EXTENDED_ERA5VARS)

            for var in cu.EXTENDED_ERA5VARS:
                plot_global_rmse_bias(ds_metrics, var)
                plt.savefig(self.outputs[f'fig_{var}'])


def plot_mcs_local_var(ds, var, title, mode='time_mean'):
    # ds.load()
    fig, (ax1, ax2, ax3) = plt.subplots(
        3,
        1,
        subplot_kw=dict(projection=ccrs.PlateCarree()),
    )
    fig.set_size_inches((20, 15))
    ax1.set_title(title)

    mask_sum = ds.dist_mask_sum.sum(dim='time')
    # mask_sum_masked = np.ma.masked_array(mask_sum.values, mask=mask_sum < 10)

    da_var = ds[f'mcs_local_{var}']
    da_mean = ds[var]

    vmax = max(np.max(ds[var].values), np.nanmax(ds[f'mcs_local_{var}'].values))
    if mode == 'time_mean':
        diff = da_var.mean(dim='time') - da_mean.mean(dim='time')
    elif mode == 'monthly':
        diff = (da_var - da_mean).mean(dim='time')

    absmax_diff = np.abs(diff).max()
    if var == 'cape':
        levels1 = np.linspace(0, 1000, 11)
        levels2 = np.linspace(-500, 500, 11)
    elif var == 'tcwv':
        levels1 = np.linspace(0, 60, 11)
        levels2 = np.linspace(-30, 30, 11)
    elif var.startswith('shear'):
        levels1 = np.linspace(0, 20, 11)
        levels2 = np.linspace(-10, 10, 11)
    else:
        levels1 = np.linspace(-0.0015, 0.0015, 11)
        levels2 = np.linspace(-0.0005, 0.0005, 11)

    # im1 = ax1.contourf(ds.longitude, ds.latitude, ds[var], levels=levels1, extend='both')
    # im2 = ax2.contourf(ds.longitude, ds.latitude, ds[f'mcs_local_{var}'], levels=levels1, extend='both')
    # im3 = ax3.contourf(ds.longitude, ds.latitude, diff, levels=levels2, cmap='bwr', extend='both')
    extent = (0, 360, -60, 60)
    # print(da_mean)
    im1 = ax1.imshow(
        np.ma.masked_array(da_mean.mean(dim='time').values, mask=mask_sum < 10),
        vmin=levels1[0],
        vmax=levels1[-1],
        extent=extent,
    )
    im2 = ax2.imshow(
        np.ma.masked_array(da_var.mean(dim='time').values, mask=mask_sum < 10),
        vmin=levels1[0],
        vmax=levels1[-1],
        extent=extent,
    )
    im3 = ax3.imshow(
        np.ma.masked_array(diff.values, mask=mask_sum < 10),
        vmin=levels2[0],
        vmax=levels2[-1],
        cmap='bwr',
        extent=extent,
    )

    plt.colorbar(im1, ax=[ax1, ax2], extend='max')
    plt.colorbar(im3, ax=ax3, extend='both')
    for ax in [ax1, ax2, ax3]:
        ax.coastlines()
        ax.set_ylim((-60, 60))


class PlotMcsLocalEnv(TaskRule):
    enabled = False
    @staticmethod
    def rule_inputs(year, mode):
        inputs = {
            f'mcs_local_env_{year}_{month}': fmtp(cu.FMT_PATH_COMBINE_MCS_LOCAL_ENV, year=year, month=month, mode=mode)
            for month in cu.MONTHS
        }
        return inputs

    rule_outputs = {
        f'fig_{radius}_{plot_type}_{var}': (
            PATHS['figdir'] / 'mcs_local_envs' / f'mcs_local_env_r{radius}km_{plot_type}_{var}_{{mode}}_{{year}}.png'
        )
        for var in cu.EXTENDED_ERA5VARS
        for plot_type in ['time_mean', 'monthly']
        for radius in cu.RADII
    }

    depends_on = [
        plot_mcs_local_var,
    ]

    var_matrix = {
        'year': cu.YEARS,
        'mode': ['init', 'lifetime'],
    }

    def rule_run(self):

        with xr.open_mfdataset(self.inputs.values()) as ds:
            ds.load()
            for var in cu.EXTENDED_ERA5VARS:
                for radius in cu.RADII:
                    title = f'r{radius}km - {var}'
                    print(title)
                    plot_mcs_local_var(ds.sel(radius=radius), var, title, 'time_mean')
                    plt.savefig(self.outputs[f'fig_{radius}_time_mean_{var}'])

                    plot_mcs_local_var(ds.sel(radius=radius), var, title, 'monthly')
                    plt.savefig(self.outputs[f'fig_{radius}_monthly_{var}'])


def plot_monthly_mcs_local_var(ax, ds, var):
    mask_sum = ds.dist_mask_sum.sum(dim='time')
    # mask_sum_masked = np.ma.masked_array(mask_sum.values, mask=mask_sum < 10)

    da_var = ds[f'mcs_local_{var}']
    da_mean = ds[var]

    vmax = max(np.max(ds[var].values), np.nanmax(ds[f'mcs_local_{var}'].values))
    diff = (da_var - da_mean).mean(dim='time')

    absmax_diff = np.abs(diff).max()
    if var == 'cape':
        levels1 = np.linspace(0, 1000, 11)
        levels2 = np.linspace(-500, 500, 11)
    elif var == 'tcwv':
        levels1 = np.linspace(0, 60, 11)
        levels2 = np.linspace(-30, 30, 11)
    elif var.startswith('shear'):
        levels1 = np.linspace(0, 20, 11)
        levels2 = np.linspace(-10, 10, 11)
    else:
        levels1 = np.linspace(-0.0015, 0.0015, 11) * 1e4
        levels2 = np.linspace(-0.0005, 0.0005, 11) * 1e4
        diff *= -1e4

    extent = (0, 360, -60, 60)
    im = ax.imshow(
        np.ma.masked_array(diff.values, mask=mask_sum < 10),
        vmin=levels2[0],
        vmax=levels2[-1],
        cmap='bwr',
        extent=extent,
    )

    if var == 'vertically_integrated_moisture_flux_div':
        label = 'MFC (10$^{-4}$ kg m$^{-2}$ s$^{-1}$)'
    else:
        label = get_labels(var)
    plt.colorbar(im, ax=ax, extend='both', label=label)
    ax.coastlines()
    ax.set_ylim((-60, 60))


class PlotCombinedMcsLocalEnv(TaskRule):
    enabled = False
    @staticmethod
    def rule_inputs(year, e5vars):
        inputs = {
            f'mcs_local_env_{year}_{month}': fmtp(
                cu.FMT_PATH_COMBINE_MCS_LOCAL_ENV, year=year, month=month, mode='init'
            )
            for month in cu.MONTHS
        }
        return inputs

    rule_outputs = {
        f'fig_{radius}': (
            PATHS['figdir'] / 'mcs_local_envs' / f'combined_mcs_local_env_r{radius}km_{{e5vars}}_init_{{year}}.png'
        )
        for radius in [500]
    }

    depends_on = [
        plot_monthly_mcs_local_var,
    ]

    var_matrix = {
        'year': cu.YEARS,
        'e5vars': [
            'cape-tcwv-shear_0-vertically_integrated_moisture_flux_div',
            # 'shear_3-theta_e_mid-RHlow-RHmid',
        ],
    }

    def rule_run(self):

        e5vars = self.e5vars.split('-')

        with xr.open_mfdataset(self.inputs.values()) as ds:
            ds.load()
            for radius in [500]:
                fig, axes = plt.subplots(2, 2, subplot_kw={'projection': ccrs.PlateCarree()})
                fig.set_size_inches((11, 4))
                for ax, var in zip(axes.flatten(), e5vars):
                    title = f'r{radius}km - {var}'
                    print(title)
                    plot_monthly_mcs_local_var(ax, ds.sel(radius=radius), var)

                plt.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.95, wspace=0.02, hspace=0.1)
                plt.savefig(self.outputs[f'fig_{radius}'])


def plot_precursor_mean_val(ds, var, radii, ax=None, N=73, colours=None, show_spread=False):
    if colours is None:
        colours = [None] * len(radii)
    ds[f'mean_{var}'].load()

    # fig, (ax1, ax2) = plt.subplots(2, 1)
    if ax is None:
        ax = plt.gca()
    for r, c in zip(radii, colours):
        print(f' plot {r}')
        if c:
            plot_kwargs = {'color': c}
        else:
            plot_kwargs = {}
        plot_data = ds[f'mean_{var}'].sel(radius=r).isel(times=slice(0, N)).mean(dim='tracks')
        if show_spread:
            data = ds[f'mean_{var}'].sel(radius=r).isel(times=slice(0, N)).values
            d25, d75 = np.nanpercentile(data, [25, 75], axis=0)
        if var == 'vertically_integrated_moisture_flux_div':
            p = ax.plot(range(-24, -24 + N), -plot_data * 1e4, label=f'{r} km', **plot_kwargs)
            if show_spread:
                spread_plot_kwargs = {**plot_kwargs, **{'linestyle': '--'}}
                if 'color' not in spread_plot_kwargs:
                    spread_plot_kwargs['color'] = p[0].get_color()
                print(spread_plot_kwargs)
                ax.plot(range(-24, -24 + N), -d25 * 1e4, **spread_plot_kwargs)
                ax.plot(range(-24, -24 + N), -d75 * 1e4, **spread_plot_kwargs)

            ylabel = 'MFC (10$^{-4}$ kg m$^{-2}$ s$^{-1}$)'
            ax.set_ylabel(ylabel)
        else:
            p = ax.plot(range(-24, -24 + N), plot_data, label=f'{r} km', **plot_kwargs)
            if show_spread:
                spread_plot_kwargs = {**plot_kwargs, **{'linestyle': '--'}}
                if 'color' not in spread_plot_kwargs:
                    spread_plot_kwargs['color'] = p[0].get_color()
                print(spread_plot_kwargs)
                ax.plot(range(-24, -24 + N), d25, **spread_plot_kwargs)
                ax.plot(range(-24, -24 + N), d75, **spread_plot_kwargs)
            ax.set_ylabel(get_labels(var))

    ax.axvline(x=0)

    # THIS IS SSSSLLLLOOOOOWWWW!!!!
    # print(f' load hist')
    # hist_data = ds[f'hist_{var}'].sel(radius=100).isel(times=slice(0, N)).load()
    # print(f' plot hist')
    # ax2.plot(range(-24, -24 + N), np.isnan(hist_data.values).sum(axis=(0, 1)))


class PlotCombinedMcsLocalEnvPrecursorMeanValue(TaskRule):
    @staticmethod
    def rule_inputs(years, e5vars):
        inputs = {
            f'mcs_local_env_{year}_{month}': fmtp(cu.FMT_PATH_LIFECYCLE_MCS_LOCAL_ENV, year=year, month=month)
            for year in years
            for month in cu.MONTHS
        }
        return inputs

    @staticmethod
    def rule_outputs(years, e5vars):
        ystr = f'{years[0]}-{years[-1]}'
        outputs = {
            'fig': (PATHS['figdir'] / 'mcs_env_cond_figs' /
                    f'mcs_local_env_precursor_mean_{e5vars}_{ystr}.png')
        }
        return outputs

    depends_on = [
        plot_precursor_mean_val,
    ]

    var_matrix = {
        'years': [
            cu.YEARS,
        ],
        'e5vars': [
            'cape-tcwv-shear_0-vertically_integrated_moisture_flux_div'
            'RHlow-RHmid-theta_e_mid',
        ],
    }

    def rule_run(self):

        fig, axes = plt.subplots(2, 2, sharex=True)
        fig.set_size_inches((10, 8))
        e5vars = self.e5vars.split('-')

        ds = xr.open_mfdataset(list(self.inputs.values()), combine='nested', concat_dim='tracks')
        ds['tracks'] = np.arange(0, ds.dims['tracks'], 1, dtype=int)
        for ax, var in zip(axes.flatten(), e5vars):
            print(var)
            plot_precursor_mean_val(ds, var, cu.RADII[1:], ax=ax, N=73)
            ax.set_xlim((-24, 48))

        axes[0, 0].legend()
        for ax in axes[1]:
            ax.set_xlabel('time from MCS initiation (hr)')
        plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95, wspace=0.22, hspace=0.1)
        plt.savefig(self.outputs[f'fig'])
        ds.close()


class PlotCombinedMcsLocalEnvPrecursorMeanValueSpread(TaskRule):
    @staticmethod
    def rule_inputs(years, e5vars):
        inputs = {
            f'mcs_local_env_{year}_{month}': fmtp(cu.FMT_PATH_LIFECYCLE_MCS_LOCAL_ENV, year=year, month=month)
            for year in years
            for month in cu.MONTHS
        }
        return inputs

    @staticmethod
    def rule_outputs(years, e5vars):
        ystr = f'{years[0]}-{years[-1]}'
        outputs = {
            'fig': (PATHS['figdir'] / 'mcs_env_cond_figs' /
                    f'mcs_local_env_precursor_mean_spread_{e5vars}_{ystr}.png')
        }
        return outputs

    depends_on = [
        plot_precursor_mean_val,
    ]

    var_matrix = {
        'years': [
            cu.YEARS,
        ],
        'e5vars': [
            'cape-tcwv-shear_0-vertically_integrated_moisture_flux_div'
        ],
    }

    def rule_run(self):

        fig, axes = plt.subplots(2, 2, sharex=True)
        fig.set_size_inches((10, 8))
        e5vars = self.e5vars.split('-')

        ds = xr.open_mfdataset(list(self.inputs.values()), combine='nested', concat_dim='tracks')
        ds['tracks'] = np.arange(0, ds.dims['tracks'], 1, dtype=int)
        for ax, var in zip(axes.flatten(), e5vars):
            print(var)
            plot_precursor_mean_val(ds, var, [cu.RADII[2], cu.RADII[4]], ax=ax, N=73, show_spread=True)
            ax.set_xlim((-24, 48))

        axes[0, 0].legend()
        for ax in axes[1]:
            ax.set_xlabel('time from MCS initiation (hr)')
        plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95, wspace=0.22, hspace=0.1)
        plt.savefig(self.outputs[f'fig'])
        ds.close()


class PlotCombinedMcsLocalEnvPrecursorMeanValueDCSpread(TaskRule):
    @staticmethod
    def rule_inputs(years, e5var, mode):
        inputs = {
            f'mcs_local_env_{year}_{month}': fmtp(cu.FMT_PATH_LIFECYCLE_MCS_LOCAL_ENV, year=year, month=month)
            for year in years
            for month in cu.MONTHS
        }
        for year in years:
            inputs[f'tracks_{year}'] = cu.fmt_mcs_stats_path(year)
        return inputs

    @staticmethod
    def rule_outputs(years, e5var, mode):
        ystr = f'{years[0]}-{years[-1]}'
        outputs = {
            'fig': (PATHS['figdir'] / 'mcs_env_cond_figs' /
                    f'mcs_local_env_precursor_mean_{e5var}_{ystr}_{mode}.png')
        }
        return outputs

    depends_on = [
        plot_precursor_mean_val,
    ]

    var_matrix = {
        'years': [
            # [2020],
            cu.YEARS,
        ],
        'e5var': cu.EXTENDED_ERA5VARS,
        'mode': ['diurnal_cycle', 'seasonal'],
    }

    def rule_run(self):

        tracks_keys = [k for k in self.inputs.keys() if k.startswith('tracks_')]
        tracks_paths = [self.inputs.pop(k) for k in tracks_keys]
        tracks = McsTracks.mfopen(tracks_paths, None)

        track_start_times = pd.DatetimeIndex(tracks.dstracks.start_basetime.values)
        if self.mode == 'diurnal_cycle':
            lst_offset = tracks.dstracks.meanlon.values[:, 0] / 360 * 24 * 3600 * 1e3  # in ms.
            lst_track_start_times = track_start_times + lst_offset.astype('timedelta64[ms]')
            N = 8
            hour_groups = [
                [0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11],
                [12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]
            ]
        elif self.mode == 'seasonal':
            seasons = [[12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
            N = 4

        ds = xr.open_mfdataset(list(self.inputs.values()), combine='nested', concat_dim='tracks')
        ds['tracks'] = np.arange(0, ds.dims['tracks'], 1, dtype=int)
        assert len(ds.tracks) == len(tracks.dstracks.tracks)

        fig, (ax, cax) = plt.subplots(2, 1, layout='constrained', gridspec_kw={'height_ratios':[1, 0.05]})
        fig.set_size_inches((10, 8))
        print(self.e5var)
        cmap = mpl.colormaps['twilight_shifted']
        for i in range(N):
            if self.mode == 'diurnal_cycle':
                hours = hour_groups[i]
                time_filter = lst_track_start_times.hour.isin(hours)
            elif self.mode == 'seasonal':
                months = seasons[i]
                time_filter = track_start_times.month.isin(months)
            c = cmap(i / N)
            print(i, time_filter.sum())
            plot_precursor_mean_val(ds.isel(tracks=time_filter), self.e5var, cu.RADII[2:3], ax=ax, N=73, colours=[c], show_spread=True)
            ax.set_xlim((-24, 48))

        # axes[0, 0].legend()
        # for ax in axes[-1]:
        #     ax.set_xlabel('time from MCS initiation (hr)')
        if self.mode == 'diurnal_cycle':
            norm = mpl.colors.Normalize(vmin=0, vmax=24)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            plt.colorbar(sm, cax=cax, orientation='horizontal',
                         ticks=np.arange(24), boundaries=np.linspace(-0.5, 23.5, 9))
        elif self.mode == 'seasonal':
            norm = mpl.colors.Normalize(vmin=0, vmax=4)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            labels = ['DJF', 'MAM', 'JJA', 'SON']
            cb = plt.colorbar(sm, cax=cax, orientation='horizontal',
                              ticks=np.arange(4), boundaries=np.linspace(-0.5, 3.5, 5))
            cb.set_ticklabels(labels)
        # plt.colorbar(sm, ticks=np.linspace(0,2,N),
        #              boundaries=np.arange(-0.05,2.1,.1))
        # plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95, wspace=0.22, hspace=0.1)
        plt.savefig(self.outputs[f'fig'])
        ds.close()
