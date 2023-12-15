from itertools import product
import string

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from mcs_prime import PATHS, mcs_mask_plotter, McsTracks
from remake import Remake, TaskRule
from remake.util import format_path as fmtp

import mcs_prime.mcs_prime_config_util as cu

slurm_config = {'account': 'short4hr', 'queue': 'short-serial-4hr', 'mem': 64000}
plotting = Remake(config=dict(slurm=slurm_config, content_checks=False))

YEARS = [2020]

SUBFIG_SQ_SIZE = 9  # cm

def cm_to_inch(*args):
    return [v / 2.54 for v in args]


def get_labels(var):
    labels = {
        'cape': 'CAPE (J kg$^{-1}$)',
        'cin': 'CIN (J kg$^{-1}$)',
        'tcwv': 'TCWV (mm)',
        'vertically_integrated_moisture_flux_div': 'VIMFD (kg m$^{-2}$ s$^{-1}$)',
        'shear_0': 'LLS (m s$^{-1}$)',
        'shear_1': 'L2MS (m s$^{-1}$)',
        'shear_2': 'M2HS (m s$^{-1}$)',
        'shear_3': 'DS (m s$^{-1}$)',
        'theta_e_mid': r'$\theta_e$mid (K)',
        'RHlow': 'RHlow (-)',
        'RHmid': 'RHmid (-)',
        'delta_3h_cape': r'$\Delta$ 3h CAPE (J kg$^{-1}$)',
        'delta_3h_tcwv': r'$\Delta$ 3h TCWV (mm)',
    }
    return labels[var]


def plot_hist(ds, ax=None, reg='all', var='cape', s=None, log=True):
    if s is None:
        s = slice(0, 101, None)
    if ax is None:
        plt.figure(figsize=(10, 10))
        ax = plt.gca()

    def _plot_hist(ds, ax, h, fmt, title):
        bins = ds[f'{var}_bins'].values
        width = bins[1] - bins[0]
        h_density = h / (h.sum() * width)
        if var == 'vertically_integrated_moisture_flux_div':
            ax.plot(-ds[f'{var}_hist_mids'].values[s] * 1e4, h_density[s], fmt, label=title)
        else:
            ax.plot(ds[f'{var}_hist_mids'].values[s], h_density[s], fmt, label=title)

    # ax.set_title(f'{var.upper()} distributions')
    _plot_hist(ds, ax, np.nansum(ds[f'{reg}_{var}_MCS_core'].values, axis=0), 'r-', 'MCS core')
    _plot_hist(ds, ax, np.nansum(ds[f'{reg}_{var}_MCS_shield'].values, axis=0), 'r--', 'MCS shield')
    _plot_hist(ds, ax, np.nansum(ds[f'{reg}_{var}_cloud_core'].values, axis=0), 'b-', 'non-MCS core')
    _plot_hist(ds, ax, np.nansum(ds[f'{reg}_{var}_cloud_shield'].values, axis=0), 'b--', 'non-MCS shield')
    _plot_hist(ds, ax, np.nansum(ds[f'{reg}_{var}_env'].values, axis=0), 'k-', 'env')
    # ax.legend()
    if log:
        ax.set_yscale('log')

    # ax.set_xlabel(get_labels(var))


def plot_hist_probs(ds, ax=None, reg='all', var='cape', s=None):
    if s is None:
        s = slice(0, 101, None)
    if ax is None:
        plt.figure(figsize=(10, 10))
        ax = plt.gca()

    counts = np.zeros((5, ds[f'{reg}_{var}_MCS_core'].shape[1]))
    counts[0] = np.nansum(ds[f'{reg}_{var}_MCS_core'].values, axis=0)
    counts[1] = np.nansum(ds[f'{reg}_{var}_MCS_shield'].values, axis=0)
    counts[2] = np.nansum(ds[f'{reg}_{var}_cloud_core'].values, axis=0)
    counts[3] = np.nansum(ds[f'{reg}_{var}_cloud_shield'].values, axis=0)
    counts[4] = np.nansum(ds[f'{reg}_{var}_env'].values, axis=0)
    probs = counts / counts.sum(axis=0)[None, :]

    if var == 'vertically_integrated_moisture_flux_div':
        x = -ds[f'{var}_hist_mids'].values[s] * 1e4
    else:
        x = ds[f'{var}_hist_mids'].values[s]
    # ax.set_title(f'{var.upper()} probabilities')
    ax.plot(x, probs[0][s], 'r-', label='MCS core')
    ax.plot(x, probs[1][s], 'r--', label='MCS shield')
    ax.plot(x, probs[2][s], 'b-', label='non-MCS core')
    ax.plot(x, probs[3][s], 'b--', label='non-MCS shield')
    ax.plot(x, probs[4][s], 'k-', label='env')
    # ax.legend()


def plot_hists_for_var(ds, var):
    xlim_ylim_title = {
        'cape': ((0, 2500), (0, 0.0014), 'CAPE {reg}'),
        'cin': ((None, None), (None, None), 'CIN {reg}'),
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
    fig.set_size_inches(cm_to_inch(SUBFIG_SQ_SIZE * 3, SUBFIG_SQ_SIZE * 1.5))

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


class PlotCombineConditionalERA5Hist(TaskRule):
    @staticmethod
    def rule_inputs(year, core_method):
        inputs = {
            f'hist_{year}_{month}': fmtp(cu.FMT_PATH_COND_HIST_HOURLY, year=year, month=month, core_method=core_method)
            for month in cu.MONTHS
        }
        return inputs

    rule_outputs = {
        f'fig_{var}': (
            PATHS['figdir'] / 'conditional_era5_histograms' / f'yearly_hist_{var}_{{year}}_{{core_method}}.png'
        )
        for var in cu.EXTENDED_ERA5VARS
    }

    depends_on = [get_labels, plot_hist, plot_hist_probs, plot_hists_for_var]

    var_matrix = {'year': YEARS, 'core_method': ['tb', 'precip']}

    def rule_run(self):
        with xr.open_mfdataset(self.inputs.values()) as ds:
            ds.load()
            for var in cu.EXTENDED_ERA5VARS:
                print(var)
                plot_hists_for_var(ds, var)
                plt.savefig(self.outputs[f'fig_{var}'])


def plot_combined_hists_for_var(ax0, ax1, ds, var):
    xlim_ylim_title = {
        'cape': ((0, 2500), (0, 0.0014), 'CAPE'),
        'cin': ((0, 500), (0, 0.14), 'CIN'),
        'tcwv': ((0, 80), (0, 0.08), 'TCWV'),
        'shear_0': ((0, 30), (0, 0.15), 'LLS'),
        'shear_1': ((0, 30), (0, 0.15), 'L2MS'),
        'shear_2': ((0, 30), (0, 0.15), 'M2HS'),
        'shear_3': ((0, 30), (0, 0.15), 'DS'),
        'RHlow': ((0, 1), (0, 8), 'RHlow'),
        'RHmid': ((0, 1), (0, 4), 'RHmid'),
        'theta_e_mid': ((300, 360), (0, None), r'$\theta_e$'),
        'vertically_integrated_moisture_flux_div': ((-12, 12), (0, None), 'VIMFD'),
        'delta_3h_cape': ((-300, 300), (0, None), r'\Delta 3h CAPE'),
        'delta_3h_tcwv': ((-20, 20), (0, None), r'\Delta 3h TCWV'),
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


class PlotCombineVarConditionalERA5Hist(TaskRule):
    @staticmethod
    def rule_inputs(year, e5vars):
        inputs = {
            f'hist_{year}_{month}': fmtp(cu.FMT_PATH_COND_HIST_HOURLY, year=year, month=month, core_method='tb')
            for month in cu.MONTHS
        }
        return inputs

    rule_outputs = {
        'fig': (PATHS['figdir'] / 'conditional_era5_histograms' / 'combined_yearly_hist_{e5vars}_{year}_tb.pdf')
    }

    depends_on = [get_labels, plot_hist, plot_hist_probs, plot_combined_hists_for_var]

    var_matrix = {
        'year': YEARS,
        'e5vars': ['all', 'tcwv-RHmid-vertically_integrated_moisture_flux_div'],
    }

    def rule_run(self):
        if self.e5vars == 'all':
            e5vars = [
                'cape',
                'cin',
                'tcwv',
                'shear_0',
                # 'shear_1', # Least interesting/quite close to LLS (shear_0)
                'shear_2',
                'shear_3',
                'RHlow',
                'RHmid',
                'vertically_integrated_moisture_flux_div',
                'delta_3h_cape',
                'delta_3h_tcwv',
                'theta_e_mid',
            ]
        else:
            e5vars = self.e5vars.split('-')

        nrows = (((len(e5vars) - 1) // 3) + 1) * 2  # trust me.
        sns.set_theme(palette='deep')
        sns.set_style('ticks')
        sns.set_context('paper')

        fig, axes = plt.subplots(nrows, 3, layout='constrained')
        fudge_factor = 0.8 if self.e5vars == 'all' else 1.
        fig.set_size_inches(cm_to_inch(SUBFIG_SQ_SIZE * 3, SUBFIG_SQ_SIZE * nrows / 2 * fudge_factor))

        with xr.open_mfdataset(self.inputs.values()) as ds:
            ds.load()
            for i, var in enumerate(e5vars):
                row_idx = (i // 3) * 2
                col_idx = (i % 3)
                print(var, row_idx, col_idx)
                ax0 = axes[row_idx, col_idx]
                ax1 = axes[row_idx + 1, col_idx]
                plot_combined_hists_for_var(ax0, ax1, ds, var)

                if var == 'vertically_integrated_moisture_flux_div':
                    label = 'MFC (10$^{-4}$ kg m$^{-2}$ s$^{-1}$)'
                else:
                    label = get_labels(var)
                c = string.ascii_lowercase[i]
                ax0.set_title(f'{c}) {label}', loc='left')

        if self.e5vars == 'all':
            axes[0, 0].legend(loc='lower left', bbox_to_anchor=(0.5, 0.3), framealpha=1)
        else:
            axes[0, -1].legend(loc='lower left', bbox_to_anchor=(0.6, 0.4), framealpha=1)

        for ax in axes[::2].flatten():
            ax.set_xticklabels([])
        plt.savefig(self.outputs['fig'])


def plot_convection_hourly_hists(ds, var, axes=None):
    xlim_ylim_title = {
        'cape': ((0, 2500), (0, 0.0014), 'CAPE'),
        'cin': ((0, 500), (0, 0.14), 'CIN'),
        'tcwv': ((0, 80), (0, 0.08), 'TCWV'),
        'shear_0': ((0, 30), (0, 0.15), 'LLS'),
        'shear_1': ((0, 30), (0, 0.15), 'L2MS'),
        'shear_2': ((0, 30), (0, 0.15), 'M2HS'),
        'shear_3': ((0, 30), (0, 0.15), 'DS'),
        'RHlow': ((0, 1), (0, 8), 'RHlow'),
        'RHmid': ((0, 1), (0, 4), 'RHmid'),
        'theta_e_mid': ((300, 360), (0, None), r'$\theta_e$'),
        'vertically_integrated_moisture_flux_div': ((-12, 12), (0, None), 'VIMFD'),
        'delta_3h_cape': ((-300, 300), (0, None), r'\Delta 3h CAPE'),
        'delta_3h_tcwv': ((-20, 20), (0, None), r'\Delta 3h TCWV'),
    }
    if axes is None:
        fig, (ax, ax2) = plt.subplots(2, 1, sharex=True)
    else:
        ax, ax2 = axes
    xlim, ylim, title = xlim_ylim_title[var]
    ax.set_xlim(xlim)
    ax2.set_xlim(xlim)
    # ax.set_ylim(ylim)

    # if var == 'vertically_integrated_moisture_flux_div':
    #     ax.set_title('MFC')
    # else:
    #     ax.set_title(var.upper())
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
        ax2.plot(x, dt / dt.sum(), label=lsreg, color=p[0].get_color(), linestyle='-')

        # if var == 'cape':
        #     plt.xlabel('J kg$^{-1}$')
        #     plt.xlim((0, 5000))
        # elif var == 'tcwv':
        #     plt.xlabel('mm')
        #     plt.xlim((0, 100))
        # elif var == 'vertically_integrated_moisture_flux_div':
        #     plt.xlabel('10$^{-4}$ kg m$^{-2}$ s$^{-1}$')
        #     plt.xlim((-0.002 * 1e4, 0.002 * 1e4))
        # elif var.startswith('shear'):
        #     plt.xlabel('m s$^{-1}$')
    ax.set_ylabel('p(MCS conv|conv)')
    ax2.set_ylabel('pdf')

    ax.set_ylim((0, 1))
    ax2.set_ylim((0, None))

    # ax.legend()


class PlotConvectionConditionalERA5Hist(TaskRule):
    @staticmethod
    def rule_inputs(year, core_method):
        inputs = {
            f'hist_{year}_{month}': fmtp(cu.FMT_PATH_COND_HIST_HOURLY, year=year, month=month, core_method=core_method)
            for month in cu.MONTHS
        }
        return inputs

    rule_outputs = {
        f'fig_{var}': (
            PATHS['figdir']
            / 'conditional_era5_histograms'
            / f'convection_yearly_hist_{var}_{{year}}_{{core_method}}.png'
        )
        for var in cu.EXTENDED_ERA5VARS
    }

    depends_on = [plot_convection_hourly_hists]

    var_matrix = {'year': YEARS, 'core_method': ['tb', 'precip']}

    def rule_run(self):
        with xr.open_mfdataset(self.inputs.values()) as ds:
            ds.load()

            for var in cu.EXTENDED_ERA5VARS:
                print(var)
                plot_convection_hourly_hists(ds, var)
                plt.savefig(self.outputs[f'fig_{var}'])


class PlotCombineConvectionConditionalERA5Hist(TaskRule):
    @staticmethod
    def rule_inputs(year, core_method, e5vars):
        inputs = {
            f'hist_{year}_{month}': fmtp(cu.FMT_PATH_COND_HIST_HOURLY, year=year, month=month, core_method=core_method)
            for month in cu.MONTHS
        }
        return inputs

    rule_outputs = {
        'fig': (
            PATHS['figdir']
            / 'conditional_era5_histograms'
            / 'combine_convection_yearly_hist_{e5vars}_{year}_{core_method}.pdf'
        )
    }

    depends_on = [plot_convection_hourly_hists]

    var_matrix = {
        'year': YEARS,
        'core_method': ['tb', 'precip'],
        'e5vars': ['all', 'tcwv-RHmid-vertically_integrated_moisture_flux_div'],
    }

    def rule_run(self):
        if self.e5vars == 'all':
            e5vars = [
                'cape',
                'cin',
                'tcwv',
                'shear_0',
                # 'shear_1', # Least interesting/quite close to LLS (shear_0)
                'shear_2',
                'shear_3',
                'RHlow',
                'RHmid',
                'vertically_integrated_moisture_flux_div',
                'delta_3h_cape',
                'delta_3h_tcwv',
                'theta_e_mid',
            ]
        else:
            e5vars = self.e5vars.split('-')

        nrows = (((len(e5vars) - 1) // 3) + 1) * 2  # trust me.
        sns.set_theme(palette='deep')
        sns.set_style('ticks')
        sns.set_context('paper')

        fig, axes = plt.subplots(nrows, 3, layout='constrained')
        fudge_factor = 0.8 if self.e5vars == 'all' else 1.
        fig.set_size_inches(cm_to_inch(SUBFIG_SQ_SIZE * 3, SUBFIG_SQ_SIZE * nrows / 2 * fudge_factor))

        with xr.open_mfdataset(self.inputs.values()) as ds:
            ds.load()

            for i, var in enumerate(e5vars):
                row_idx = (i // 3) * 2
                col_idx = (i % 3)
                print(var, row_idx, col_idx)
                ax0 = axes[row_idx, col_idx]
                ax1 = axes[row_idx + 1, col_idx]
                plot_convection_hourly_hists(ds, var, (ax0, ax1))

                if var == 'vertically_integrated_moisture_flux_div':
                    label = 'MFC (10$^{-4}$ kg m$^{-2}$ s$^{-1}$)'
                else:
                    label = get_labels(var)
                c = string.ascii_lowercase[i]
                ax0.set_title(f'{c}) {label}', loc='left')

        if self.e5vars == 'all':
            axes[0, 0].legend(loc='lower left', bbox_to_anchor=(0.8, -0.2), framealpha=1)
        else:
            axes[0, -1].legend(loc='lower left', bbox_to_anchor=(0.8, -0.2), framealpha=1)

        for ax in axes[::2].flatten():
            ax.set_xticklabels([])
        fig.align_ylabels(axes)
        plt.savefig(self.outputs['fig'])


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

    var_matrix = {'year': YEARS}

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

    var_matrix = {'year': YEARS}

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
        'year': YEARS,
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
    # Mask out grid points without much data.
    masked_diff = np.ma.masked_array(diff.values, mask=mask_sum < 10)
    if var == 'vertically_integrated_moisture_flux_div':
        masked_diff *= -1e4

    # Calculate max/min values so that they contain at least 90% of all points.
    contain = 90
    pmin = (100 - contain) / 2
    pmax = 100 - pmin
    percentiles = np.nanpercentile(masked_diff.compressed(), [pmin, pmax])
    vmax = np.abs(percentiles).max()
    vmin = -vmax

    # absmax_diff = np.abs(diff).max()
    # if var == 'cape':
    #     levels2 = np.linspace(-500, 500, 11)
    # elif var == 'tcwv':
    #     levels2 = np.linspace(-30, 30, 11)
    # elif var.startswith('shear'):
    #     levels2 = np.linspace(-10, 10, 11)
    # elif var == 'vertically_integrated_moisture_flux_div':
    #     levels2 = np.linspace(-0.0005, 0.0005, 11) * 1e4
    #     diff *= -1e4
    # else:
    #     levels2 = np.linspace(-absmax_diff / 3, absmax_diff / 3, 11)

    extent = (0, 360, -60, 60)
    # cmap = sns.color_palette('vlag', as_cmap=True)
    cmap = sns.color_palette('coolwarm', as_cmap=True)
    im = ax.imshow(
        masked_diff,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        extent=extent,
    )

    if var == 'vertically_integrated_moisture_flux_div':
        label = 'MFC (10$^{-4}$ kg m$^{-2}$ s$^{-1}$)'
    else:
        label = get_labels(var)
    plt.colorbar(im, ax=ax, extend='both')
    ax.set_title(label)
    ax.coastlines()
    ax.set_ylim((-60, 60))


class PlotCombinedMcsLocalEnv(TaskRule):
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
        for radius in [100, 200, 500]
    }

    depends_on = [
        plot_monthly_mcs_local_var,
    ]

    var_matrix = {
        'year': YEARS,
        'e5vars': [
            'cape-tcwv-shear_0-vertically_integrated_moisture_flux_div',
            # 'shear_3-theta_e_mid-RHlow-RHmid',
        ],
    }

    def rule_run(self):
        e5vars = self.e5vars.split('-')

        with xr.open_mfdataset(self.inputs.values()) as ds:
            ds.load()

            for radius in [100, 200, 500]:
                fig, axes = plt.subplots(2, 2, subplot_kw={'projection': ccrs.PlateCarree()})
                fig.set_size_inches((11, 4))
                for ax, var in zip(axes.flatten(), e5vars):
                    title = f'r{radius}km - {var}'
                    print(title)

                    plot_monthly_mcs_local_var(ax, ds.sel(radius=radius), var)

                plt.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.95, wspace=0.02, hspace=0.1)
            plt.savefig(self.outputs[f'fig_{radius}'])


class PlotAllCombinedMcsLocalEnv(TaskRule):
    @staticmethod
    def rule_inputs(year):
        inputs = {
            f'mcs_local_env_{year}_{month}': fmtp(
                cu.FMT_PATH_COMBINE_MCS_LOCAL_ENV, year=year, month=month, mode='init'
            )
            for month in cu.MONTHS
        }
        return inputs

    rule_outputs = {
        f'fig_{radius}': (
            PATHS['figdir'] / 'mcs_local_envs' / f'all_combined_mcs_local_env_r{radius}km_init_{{year}}.pdf'
        )
        # for radius in [100, 200, 500, 1000]
        for radius in [500]
    }

    depends_on = [
        plot_monthly_mcs_local_var,
    ]

    var_matrix = {
        'year': [2020],
    }

    def rule_run(self):
        e5vars = [
            'cape',
            'cin',
            'tcwv',
            'shear_0',
            # 'shear_1', # Least interesting/quite close to LLS (shear_0)
            'shear_2',
            'shear_3',
            'RHlow',
            'RHmid',
            'vertically_integrated_moisture_flux_div',
            'delta_3h_cape',
            'delta_3h_tcwv',
            'theta_e_mid',
        ]

        sns.set_theme(palette='deep')
        sns.set_style('ticks')
        sns.set_context('paper')

        with xr.open_mfdataset(self.inputs.values()) as ds:
            ds.load()

            # for radius in [100, 200, 500, 1000]:
            for radius in [500]:
                print(f'{radius}km')
                fig, axes = plt.subplots(4, 3, subplot_kw={'projection': ccrs.PlateCarree()}, layout='constrained')
                # Size of fig. Each one is 360x120, or aspect of 3.
                # Hence use /3 in height.
                # 1.1 is a fudge factor for colorbar, title...
                fig.set_size_inches(cm_to_inch(SUBFIG_SQ_SIZE * 3, SUBFIG_SQ_SIZE * 4 / 3 * 1.1))
                for ax, var in zip(axes.flatten(), e5vars):
                    title = f'-  {var}'
                    print(title)
                    plot_monthly_mcs_local_var(ax, ds.sel(radius=radius), var)

                # plt.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.95, wspace=0.02, hspace=0.1)
                plt.savefig(self.outputs[f'fig_{radius}'])


def plot_grouped_precursor_mean_val(grouped_data_dict, ax=None, show_spread=False):
    """grouped_data_dict key is the label for the data, and
    each entry is a dict containing:
    * xr.DataArray (req),
    * ylabel (req),
    * xvals (req),
    * plot_kwargs,
    * spread_plot_kwargs,
    """
    if ax is None:
        ax = plt.gca()

    xvals = grouped_data_dict.pop('xvals')
    for label, data_dict in grouped_data_dict.items():
        data_array = data_dict['data_array']
        ylabel = data_dict['ylabel']
        plot_kwargs = data_dict.get('plot_kwargs', {})

        plot_data = data_array.mean(dim='tracks')
        p = ax.plot(xvals, plot_data, label=label, **plot_kwargs)

        if show_spread:
            d25, d75 = np.nanpercentile(data_array.values, [25, 75], axis=0)
            spread_plot_kwargs = data_dict.get('spread_plot_kwargs', {})
            if not spread_plot_kwargs:
                spread_plot_kwargs = {**plot_kwargs, **{'linestyle': '--'}}
                if 'color' not in spread_plot_kwargs:
                    spread_plot_kwargs['color'] = p[0].get_color()
            ax.plot(xvals, d25, **spread_plot_kwargs)
            ax.plot(xvals, d75, **spread_plot_kwargs)
        # ax.set_ylabel(ylabel)
    # ax.set_xlabel('time from MCS initiation (hr)')
    # ax.legend()
    # ax.axvline(x=0)
    ax.set_xlim(xvals[0], xvals[-1])

def plot_precursor_mean_val_radii(ds, var, radii, ax=None, N=73):
    ds[f'mean_{var}'].load()

    # fig, (ax1, ax2) = plt.subplots(2, 1)
    if ax is None:
        ax = plt.gca()
    grouped_data_dict = {'xvals': range(-24, -24 + N)}
    for r in radii:
        data_array = ds[f'mean_{var}'].sel(radius=r).isel(times=slice(0, N))
        if var == 'vertically_integrated_moisture_flux_div':
            data_array = -data_array * 1e4
            ylabel = 'MFC (10$^{-4}$ kg m$^{-2}$ s$^{-1}$)'
        else:
            ylabel = get_labels(var)

        grouped_data_dict[f'{r} km'] = {
            'data_array': data_array,
            'ylabel': ylabel,
        }
    plot_grouped_precursor_mean_val(grouped_data_dict)
    #     if show_spread:
    #         data = ds[f'mean_{var}'].sel(radius=r).isel(times=slice(0, N)).values
    #         d25, d75 = np.nanpercentile(data, [25, 75], axis=0)
    #     if var == 'vertically_integrated_moisture_flux_div':
    #         p = ax.plot(range(-24, -24 + N), -plot_data * 1e4, label=f'{r} km', **plot_kwargs)
    #         if show_spread:
    #             spread_plot_kwargs = {**plot_kwargs, **{'linestyle': '--'}}
    #             if 'color' not in spread_plot_kwargs:
    #                 spread_plot_kwargs['color'] = p[0].get_color()
    #             print(spread_plot_kwargs)
    #             ax.plot(range(-24, -24 + N), -d25 * 1e4, **spread_plot_kwargs)
    #             ax.plot(range(-24, -24 + N), -d75 * 1e4, **spread_plot_kwargs)

    #         ylabel = 'MFC (10$^{-4}$ kg m$^{-2}$ s$^{-1}$)'
    #         ax.set_ylabel(ylabel)
    #     else:
    #         p = ax.plot(range(-24, -24 + N), plot_data, label=f'{r} km', **plot_kwargs)
    #         if show_spread:
    #             spread_plot_kwargs = {**plot_kwargs, **{'linestyle': '--'}}
    #             if 'color' not in spread_plot_kwargs:
    #                 spread_plot_kwargs['color'] = p[0].get_color()
    #             print(spread_plot_kwargs)
    #             ax.plot(range(-24, -24 + N), d25, **spread_plot_kwargs)
    #             ax.plot(range(-24, -24 + N), d75, **spread_plot_kwargs)
    #         ax.set_ylabel(get_labels(var))

    # ax.axvline(x=0)



class PlotMcsLocalEnvPrecursorMeanValue(TaskRule):
    @staticmethod
    def rule_inputs(year, N):
        inputs = {
            f'mcs_local_env_{year}_{month}': fmtp(cu.FMT_PATH_LIFECYCLE_MCS_LOCAL_ENV, year=year, month=month)
            for month in cu.MONTHS
        }
        # for month in [1]}
        return inputs

    rule_outputs = {
        f'fig_{var}': (PATHS['figdir'] / 'mcs_local_envs' / f'mcs_local_env_precursor_mean_{var}_{{year}}_{{N}}.png')
        for var in cu.EXTENDED_ERA5VARS
    }

    depends_on = [
        plot_precursor_mean_val_radii,
    ]

    var_matrix = {
        'year': YEARS,
        'N': [49, 73, 97, 424],
    }

    def rule_run(self):
        with xr.open_mfdataset(self.inputs.values()) as ds:
            for var in cu.EXTENDED_ERA5VARS:
                print(var)
                fig, ax = plt.subplots(1, 1, layout='constrained')
                fig.set_size_inches((15, 6))
                plot_precursor_mean_val_radii(ds, var, cu.RADII[1:], ax=ax, N=self.N)
                ax.legend()
                ax.set_xlabel('time from MCS initiation (hr)')
                plt.savefig(self.outputs[f'fig_{var}'])


class PlotMcsLocalEnvPrecursorMeanValueFilteredDecomp(TaskRule):
    @staticmethod
    def rule_inputs(year, filter, decomp_mode, show_spread):
        inputs = {
            f'mcs_local_env_{year}_{month}': fmtp(cu.FMT_PATH_LIFECYCLE_MCS_LOCAL_ENV, year=year, month=month)
            for month in cu.MONTHS
        }
        inputs[f'tracks_{year}'] = cu.fmt_mcs_stats_path(year)
        return inputs

    rule_outputs = {
        f'fig_{var}': (PATHS['figdir'] / 'mcs_local_envs' / f'filtered_decomp_mcs_local_env_precursor_mean_{var}_{{year}}_{{filter}}.decomp-{{decomp_mode}}.show_spread-{{show_spread}}.png')
        for var in cu.EXTENDED_ERA5VARS
    }

    depends_on = [
        plot_grouped_precursor_mean_val,
    ]

    var_matrix = {
        'year': YEARS,
        'filter': ['land-sea', 'equator-tropics-extratropics'],
        'decomp_mode': ['all', 'diurnal_cycle', 'seasonal'],
        'show_spread': [False],
    }

    def rule_run(self):
        ds_full = xr.open_mfdataset([p for k, p in self.inputs.items() if k.startswith('mcs_local_env_')],
                               combine='nested', concat_dim='tracks')
        ds_full['tracks'] = np.arange(0, ds_full.dims['tracks'], 1, dtype=int)

        tracks_paths = [p for k, p in self.inputs.items() if k.startswith('tracks_')]
        tracks = McsTracks.mfopen(tracks_paths, None)

        track_start_times = pd.DatetimeIndex(tracks.dstracks.start_basetime.values)

        if self.decomp_mode == 'all':
            n_time_filters = 1
            time_filters = [np.ones_like(tracks.dstracks.tracks, dtype=bool)]
        elif self.decomp_mode == 'diurnal_cycle':
            lst_offset = tracks.dstracks.meanlon.values[:, 0] / 360 * 24 * 3600 * 1e3  # in ms.
            lst_track_start_times = track_start_times + lst_offset.astype('timedelta64[ms]')
            hour_groups = [
                [0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11],
                [12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]
            ]
            n_time_filters = len(hour_groups)
            time_filters = [lst_track_start_times.hour.isin(hours)
                            for hours in hour_groups]
        elif self.decomp_mode == 'seasonal':
            seasons = [[12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
            n_time_filters = len(seasons)
            time_filters = [track_start_times.month.isin(months)
                            for months in seasons]

        natural = tracks.dstracks.start_split_cloudnumber.values == -9999
        filter_vals = {}
        if self.filter == 'land-sea':
            filter_groups = ['land', 'sea', 'transitional']
            mean_landfrac = np.nanmean(tracks.dstracks.pf_landfrac.values, axis=1)
            thresh_land = 0.9
            thresh_sea = 0.1
            for i in range(n_time_filters):
                filter_vals[('land', i)] = ds_full.isel(
                    tracks=(mean_landfrac > thresh_land) & natural & time_filters[i]
                )
                filter_vals[('sea', i)] = ds_full.isel(
                    tracks=(mean_landfrac < thresh_sea) & natural & time_filters[i]
                )
                filter_vals[('transitional', i)] = ds_full.isel(
                    tracks=(
                        (mean_landfrac >= thresh_sea) &
                        (mean_landfrac <= thresh_land) &
                        natural &
                        time_filters[i]
                    )
                )
        elif self.filter == 'equator-tropics-extratropics':
            filter_groups = ['NH extratropics', 'NH tropics', 'equatorial', 'SH tropics', 'SH extratropics']
            mean_lat = np.nanmean(tracks.dstracks.meanlat.values, axis=1)
            for i in range(n_time_filters):
                filter_vals[('NH extratropics', i)] = ds_full.isel(
                    tracks=(mean_lat > 30) & natural & time_filters[i]
                )
                filter_vals[('NH tropics', i)] = ds_full.isel(
                    tracks=((mean_lat <= 30) & (mean_lat > 10) & natural) & time_filters[i]
                )
                filter_vals[('equatorial', i)] = ds_full.isel(
                    tracks=((mean_lat <= 10) & (mean_lat >= -10) & natural) & time_filters[i]
                )
                filter_vals[('SH tropics', i)] = ds_full.isel(
                    tracks=((mean_lat < -10) & (mean_lat >= -30) & natural) & time_filters[i]
                )
                filter_vals[('SH extratropics', i)] = ds_full.isel(
                    tracks=(mean_lat < -30) & natural & time_filters[i]
                )

        # cmap = mpl.colormaps['hsv']
        sns.set_theme(palette='deep')
        sns.set_style('ticks')
        sns.set_context('paper')
        # cmap = mpl.colormaps['twilight_shifted']
        cmap = sns.color_palette('hls', as_cmap=True)
        N = 73
        for var in cu.EXTENDED_ERA5VARS:
            ds_full[f'mean_{var}'].sel(radius=200).isel(times=slice(0, N)).load()

            if self.decomp_mode == 'all':
                fig, ax = plt.subplots(1, 1, layout='constrained', sharey=True)
            else:
                fig, axes = plt.subplots(1, len(filter_groups), layout='constrained', sharey=True)

            fig.set_size_inches(cm_to_inch(len(filter_groups) * SUBFIG_SQ_SIZE, SUBFIG_SQ_SIZE))

            for i in range(len(filter_groups)):
                grouped_data_dict = {'xvals': range(-24, -24 + N)}
                if self.decomp_mode != 'all':
                    ax = axes[i]

                for j in range(n_time_filters):
                    k = (filter_groups[i], j)
                    print(var, k)
                    ds = filter_vals[k]
                    data_array = ds[f'mean_{var}'].sel(radius=200).isel(times=slice(0, N))
                    if var == 'vertically_integrated_moisture_flux_div':
                        data_array = -data_array * 1e4
                        ylabel = 'MFC (10$^{-4}$ kg m$^{-2}$ s$^{-1}$)'
                    else:
                        ylabel = get_labels(var)
                    if self.decomp_mode != 'all':
                        c = cmap(j / n_time_filters)
                        plot_kwargs = {'color': c}
                    else:
                        plot_kwargs = {}

                    grouped_data_dict[k] = {
                        'data_array': data_array,
                        'ylabel': ylabel,
                        'plot_kwargs': plot_kwargs,
                    }

                plot_grouped_precursor_mean_val(grouped_data_dict, ax=ax, show_spread=self.show_spread)
            plt.savefig(self.outputs[f'fig_{var}'])


class PlotCombinedMcsLocalEnvPrecursorMeanValueFilteredDecomp(TaskRule):
    @staticmethod
    def rule_inputs(year, decomp_mode, radius):
        inputs = {
            f'mcs_local_env_{year}_{month}': fmtp(cu.FMT_PATH_LIFECYCLE_MCS_LOCAL_ENV, year=year, month=month)
            for month in cu.MONTHS
        }
        inputs[f'tracks_{year}'] = cu.fmt_mcs_stats_path(year)
        return inputs

    rule_outputs = {
        'fig': (PATHS['figdir'] / 'mcs_local_envs' / 'combined_filtered_decomp_mcs_local_env_precursor_mean_{year}.decomp-{decomp_mode}.radius-{radius}.pdf')
    }

    depends_on = [
        plot_grouped_precursor_mean_val,
    ]

    var_matrix = {
        'year': YEARS,
        'decomp_mode': ['all'],
        'radius': [100, 200, 500, 1000],
    }

    def rule_run(self):
        # e5vars = cu.EXTENDED_ERA5VARS[:12]
        # default order.
        e5vars = [
            'cape',
            'cin',
            'tcwv',
            'shear_0',
            # 'shear_1', # Least interesting/quite close to LLS (shear_0)
            'shear_2',
            'shear_3',
            'RHlow',
            'RHmid',
            'vertically_integrated_moisture_flux_div',
            'delta_3h_cape',
            'delta_3h_tcwv',
            'theta_e_mid',
        ]
        print('Open datasets')
        ds_full = xr.open_mfdataset([p for k, p in self.inputs.items() if k.startswith('mcs_local_env_')],
                               combine='nested', concat_dim='tracks')
        ds_full['tracks'] = np.arange(0, ds_full.dims['tracks'], 1, dtype=int)

        tracks_paths = [p for k, p in self.inputs.items() if k.startswith('tracks_')]
        tracks = McsTracks.mfopen(tracks_paths, None)

        track_start_times = pd.DatetimeIndex(tracks.dstracks.start_basetime.values)

        print('Build filters')
        print('  natural')
        natural = tracks.dstracks.start_split_cloudnumber.values == -9999
        filter_vals = {'natural': {'natural': natural}}

        print('  equator-tropics-extratropics')
        mean_lat = np.nanmean(tracks.dstracks.meanlat.values, axis=1)
        # Not having 5 values for lat bands makes the figure much clearer.
        # filter_vals['equator-tropics-extratropics'] = {
        #     'NH extratropics': mean_lat > 30,
        #     'NH tropics': (mean_lat <= 30) & (mean_lat > 10),
        #     'equatorial': (mean_lat <= 10) & (mean_lat >= -10),
        #     'SH tropics': (mean_lat < -10) & (mean_lat >= -30),
        #     'SH extratropics': mean_lat < -30,
        # }
        filter_vals['equator-tropics-extratropics'] = {
            'equatorial': (mean_lat <= 10) & (mean_lat >= -10),
            'tropics': ((mean_lat <= 30) & (mean_lat > 10)) | ((mean_lat < -10) & (mean_lat >= -30)),
            'extratropics': (mean_lat > 30) | (mean_lat < -30),
        }

        print('  land-sea')
        mean_landfrac = np.nanmean(tracks.dstracks.pf_landfrac.values, axis=1)
        thresh_land = 0.5
        thresh_sea = 0.5
        filter_vals['land-sea'] = {
            'sea': mean_landfrac <= thresh_sea,
            'land': mean_landfrac > thresh_land,
        }

        filter_key_combinations = list(product(
            filter_vals['natural'].keys(),
            filter_vals['equator-tropics-extratropics'].keys(),
            filter_vals['land-sea'].keys(),
        ))

        N = 73
        sns.set_theme(palette='deep')
        sns.set_style('ticks')
        sns.set_context('paper')
        fig, axes = plt.subplots(4, 3, layout='constrained', sharex=True)

        fig.set_size_inches(cm_to_inch(SUBFIG_SQ_SIZE * 3, SUBFIG_SQ_SIZE * 4))

        colours = dict(zip(
            filter_vals['equator-tropics-extratropics'].keys(),
            plt.rcParams['axes.prop_cycle'].by_key()['color']
        ))
        linestyles = dict(zip(
            filter_vals['land-sea'].keys(),
            ['-', '--']
        ))

        for i, (ax, var) in enumerate(zip(axes.flatten(), e5vars)):
            print(var)
            # ds_full[f'mean_{var}'].sel(radius=200).isel(times=slice(0, N)).load()
            data_array = ds_full[f'mean_{var}'].sel(radius=self.radius).isel(times=slice(0, N)).load()

            if var == 'vertically_integrated_moisture_flux_div':
                data_array = -data_array * 1e4
                ylabel = 'MFC (10$^{-4}$ kg m$^{-2}$ s$^{-1}$)'
            else:
                ylabel = get_labels(var)

            grouped_data_dict = {'xvals': range(-24, -24 + N)}
            # print(filter_key_combinations)
            for filter_keys in filter_key_combinations:
                # print(filter_keys)
                full_filter = None
                for filter_name, key in zip(filter_vals.keys(), filter_keys):
                    if full_filter is None:
                        full_filter = filter_vals[filter_name][key]
                    else:
                        full_filter = full_filter & filter_vals[filter_name][key]

                plot_kwargs = {
                    'color': colours[filter_keys[1]],
                    'linestyle': linestyles[filter_keys[2]],
                }
                percentage = full_filter.sum() / natural.sum() * 100
                label = ' '.join(filter_keys[1:]) + f' ({percentage:.1f}%)'
                grouped_data_dict[label] = {
                    'data_array': data_array.isel(tracks=full_filter),
                    'ylabel': ylabel,
                    'plot_kwargs': plot_kwargs,
                }

            plot_grouped_precursor_mean_val(grouped_data_dict, ax=ax)

            plot_kwargs = {
                'color': 'k',
                'linestyle': ':',
            }
            grouped_data_dict = {'xvals': range(-24, -24 + N)}
            total = natural.sum()
            grouped_data_dict[f'all ({total} tracks)'] = {
                'data_array': data_array.isel(tracks=natural),
                'ylabel': ylabel,
                'plot_kwargs': plot_kwargs,
            }

            plot_grouped_precursor_mean_val(grouped_data_dict, ax=ax)

            c = string.ascii_lowercase[i]
            varname = ylabel[:ylabel.find('(')].strip()
            units = ylabel[ylabel.find('('):].strip()
            # ax.set_title(f'{c}) {varname}', loc='left')
            # ax.set_ylabel(units)
            # Saves space to put units in title, also avoids awkard problem of
            # positioning ylabel due to different widths of numbers for e.g. 1.5 vs 335
            ax.set_title(f'{c}) {ylabel}', loc='left')
            ax.axvline(x=0, color='k')

        axes[0, -1].legend(loc='lower left', bbox_to_anchor=(0.5, 0))
        axes[-1, 1].set_xlabel('time from MCS initiation (hr)')
        plt.savefig(self.outputs[f'fig'])


class PlotIndividualMcsLocalEnvPrecursorMeanValueFilteredDecomp(TaskRule):
    @staticmethod
    def rule_inputs(year, decomp_mode, radius, show_spread):
        inputs = {
            f'mcs_local_env_{year}_{month}': fmtp(cu.FMT_PATH_LIFECYCLE_MCS_LOCAL_ENV, year=year, month=month)
            for month in cu.MONTHS
        }
        inputs[f'tracks_{year}'] = cu.fmt_mcs_stats_path(year)
        return inputs

    rule_outputs = {
        f'fig_{var}': (PATHS['figdir'] / 'mcs_local_envs' / f'indiv_filtered_decomp_mcs_local_env_precursor_mean_{var}_{{year}}.decomp-{{decomp_mode}}.radius-{{radius}}.show_spread-{{show_spread}}.pdf')
        for var in cu.EXTENDED_ERA5VARS
    }

    depends_on = [
        plot_grouped_precursor_mean_val,
    ]

    var_matrix = {
        'year': YEARS,
        'decomp_mode': ['all', 'diurnal_cycle', 'seasonal'],
        'radius': [100, 200, 500, 1000],
        'show_spread': [False, True],
    }

    def rule_run(self):
        e5vars = cu.EXTENDED_ERA5VARS
        print('Open datasets')
        ds_full = xr.open_mfdataset([p for k, p in self.inputs.items() if k.startswith('mcs_local_env_')],
                               combine='nested', concat_dim='tracks')
        ds_full['tracks'] = np.arange(0, ds_full.dims['tracks'], 1, dtype=int)

        tracks_paths = [p for k, p in self.inputs.items() if k.startswith('tracks_')]
        tracks = McsTracks.mfopen(tracks_paths, None)

        track_start_times = pd.DatetimeIndex(tracks.dstracks.start_basetime.values)

        print('Build filters')
        print('  natural')
        natural = tracks.dstracks.start_split_cloudnumber.values == -9999
        filter_vals = {'natural': {'natural': natural}}

        print('  land-sea')
        mean_landfrac = np.nanmean(tracks.dstracks.pf_landfrac.values, axis=1)
        thresh_land = 0.5
        thresh_sea = 0.5
        filter_vals['land-sea'] = {
            'sea': mean_landfrac <= thresh_sea,
            'land': mean_landfrac > thresh_land,
        }

        print('  equator-tropics-extratropics')
        mean_lat = np.nanmean(tracks.dstracks.meanlat.values, axis=1)
        filter_vals['equator-tropics-extratropics'] = {
            'equatorial': (mean_lat <= 10) & (mean_lat >= -10),
            'tropics': ((mean_lat <= 30) & (mean_lat > 10)) | ((mean_lat < -10) & (mean_lat >= -30)),
            'extratropics': (mean_lat > 30) | (mean_lat < -30),
        }

        filter_key_combinations = list(product(
            filter_vals['natural'].keys(),
            filter_vals['land-sea'].keys(),
            filter_vals['equator-tropics-extratropics'].keys(),
        ))

        if self.decomp_mode == 'all':
            n_time_filters = 1
            decomp_filters = [np.ones_like(tracks.dstracks.tracks, dtype=bool)]
        elif self.decomp_mode == 'diurnal_cycle':
            lst_offset = tracks.dstracks.meanlon.values[:, 0] / 360 * 24 * 3600 * 1e3  # in ms.
            lst_track_start_times = track_start_times + lst_offset.astype('timedelta64[ms]')
            hour_groups = [
                [0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11],
                [12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]
            ]
            n_time_filters = len(hour_groups)
            decomp_filters = [lst_track_start_times.hour.isin(hours)
                              for hours in hour_groups]
        elif self.decomp_mode == 'seasonal':
            seasons = [[12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
            n_time_filters = len(seasons)
            decomp_filters = [track_start_times.month.isin(months)
                              for months in seasons]


        N = 73
        # cmap = mpl.colormaps['twilight_shifted']
        # cmap = mpl.colormaps['hsv']
        sns.set_theme(palette='deep')
        sns.set_style('ticks')
        sns.set_context('paper')
        # cmap = mpl.colormaps['viridis']
        # cmap = sns.color_palette('hls', as_cmap=True)
        cmap = mpl.colormaps['twilight_shifted']
        for var in e5vars:
            print(var)
            fig, axes = plt.subplots(2, 3, layout='constrained', sharex=True, sharey=True)

            fig.set_size_inches(cm_to_inch(SUBFIG_SQ_SIZE * 3, SUBFIG_SQ_SIZE * 1.3))

            data_array = ds_full[f'mean_{var}'].sel(radius=self.radius).isel(times=slice(0, N)).load()
            if var == 'vertically_integrated_moisture_flux_div':
                data_array = -data_array * 1e4
                ylabel = 'MFC (10$^{-4}$ kg m$^{-2}$ s$^{-1}$)'
            else:
                ylabel = get_labels(var)

            for i, (ax, filter_keys) in enumerate(zip(axes.flatten(), filter_key_combinations)):
                grouped_data_dict = {'xvals': range(-24, -24 + N)}
                # print(filter_key_combinations)
                # print(filter_keys)
                full_filter = None
                for filter_name, key in zip(filter_vals.keys(), filter_keys):
                    if full_filter is None:
                        full_filter = filter_vals[filter_name][key]
                    else:
                        full_filter = full_filter & filter_vals[filter_name][key]

                percentage = full_filter.sum() / natural.sum() * 100
                for j in range(n_time_filters):
                    # full_filter = full_filter & decomp_filters[j]
                    tracks_filter = full_filter & decomp_filters[j]

                    c = cmap(j / n_time_filters)
                    plot_kwargs = {
                        'color': c,
                    }
                    label = ' '.join(filter_keys[1:]) + f'{j} ({percentage:.1f}%)'
                    grouped_data_dict[label] = {
                        'data_array': data_array.isel(tracks=tracks_filter),
                        'ylabel': ylabel,
                        'plot_kwargs': plot_kwargs,
                    }

                plot_grouped_precursor_mean_val(grouped_data_dict, ax=ax, show_spread=self.show_spread)

                c = string.ascii_lowercase[i]
                varname = ylabel[:ylabel.find('(')].strip()
                units = ylabel[ylabel.find('('):].strip()
                # ax.set_title(f'{c}) {varname}', loc='left')
                # ax.set_ylabel(units)
                label = ' '.join(filter_keys[1:][::-1])
                # ax.set_title(f'{c}) {label} {ylabel} ({percentage:.1f}%)', loc='left')
                ax.set_title(f'{c}) {label} ({percentage:.1f}%)', loc='left')
                ax.axvline(x=0, color='k')
                ax.set_facecolor('silver')

                if var == 'vertically_integrated_moisture_flux_div' and self.radius in {100, 200}:
                    ax.set_ylim((-0.5, 5))

            for ax in axes[:, 0]:
                ax.set_ylabel(ylabel)
            # axes[0, -1].legend(loc='upper right')
            axes[-1, 1].set_xlabel('time from MCS initiation (hr)')
            if self.decomp_mode == 'diurnal_cycle':
                norm = mpl.colors.Normalize(vmin=0, vmax=24)
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                plt.colorbar(sm, ax=axes[:, -1], orientation='vertical',
                             ticks=np.arange(24), boundaries=np.linspace(-0.5, 23.5, 9))
            elif self.decomp_mode == 'seasonal':
                norm = mpl.colors.Normalize(vmin=0, vmax=4)
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                labels = ['DJF', 'MAM', 'JJA', 'SON']
                cb = plt.colorbar(sm, ax=axes[:, -1], orientation='vertical',
                                  ticks=np.arange(4), boundaries=np.linspace(-0.5, 3.5, 5))
                cb.set_ticklabels(labels)
                cb.ax.tick_params(rotation=90)
            plt.savefig(self.outputs[f'fig_{var}'])
            plt.close('all')


class PlotCombinedMcsLocalEnvPrecursorMeanValue(TaskRule):
    @staticmethod
    def rule_inputs(year, e5vars):
        inputs = {
            f'mcs_local_env_{year}_{month}': fmtp(cu.FMT_PATH_LIFECYCLE_MCS_LOCAL_ENV, year=year, month=month)
            for month in cu.MONTHS
        }
        return inputs

    rule_outputs = {
        'fig_{e5vars}': (PATHS['figdir'] / 'mcs_local_envs' / 'mcs_local_env_precursor_mean_{e5vars}_{year}.png')
    }

    depends_on = [
        plot_precursor_mean_val_radii,
    ]

    var_matrix = {
        'year': YEARS,
        'e5vars': ['cape-tcwv-shear_0-vertically_integrated_moisture_flux_div'],
    }

    def rule_run(self):
        fig, axes = plt.subplots(2, 2, sharex=True)
        fig.set_size_inches((10, 8))
        e5vars = self.e5vars.split('-')
        print(e5vars)

        with xr.open_mfdataset(self.inputs.values(), combine='nested', concat_dim='tracks') as ds:
            # tracks is monotonically increasing *within each year*. Apply a conversion to
            # monotonically increasing over all 20 years.
            ds['tracks'] = np.arange(0, ds.dims['tracks'], 1, dtype=int)
            for ax, var in zip(axes.flatten(), e5vars):
                print(var)
                plot_precursor_mean_val_radii(ds, var, cu.RADII[1:], ax=ax, N=73)
                ax.set_xlim((-24, 48))

        axes[0, 0].legend()
        for ax in axes[1]:
            ax.set_xlabel('time from MCS initiation (hr)')
        plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95, wspace=0.22, hspace=0.1)
        plt.savefig(self.outputs[f'fig_{self.e5vars}'])
