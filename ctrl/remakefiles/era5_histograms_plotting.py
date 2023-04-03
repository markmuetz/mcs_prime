import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


from remake import Remake, TaskRule
from remake.util import format_path as fmtp

from mcs_prime import PATHS

YEARS = list(range(2000, 2021))

slurm_config = {'account': 'short4hr', 'queue': 'short-serial-4hr', 'mem': 32000}
era5_histograms_plotting = Remake(config=dict(slurm=slurm_config, content_checks=False))


def plot_hist(ds, ax=None, reg='all', v='cape', s=None, log=True):
    if s is None:
        if v == 'cape':
            s = slice(0, 500, None)
        elif v == 'tcwv':
            s = slice(0, 101, None)
    if ax is None:
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
    def _plot_hist(ds, ax, h, fmt, title):
        bins = ds[f'{v}_bins'].values
        width = bins[1] - bins[0]
        h_density = h / (h.sum() * width)
        ax.plot(ds[f'{v}_hist_mids'].values[s], h_density[s], fmt, label=title);

    ax.set_title(f'{v.upper()} distributions')
    _plot_hist(ds, ax, np.nansum(ds[f'{reg}_{v}_MCS_core'].values, axis=0), 'r-', 'MCS core')
    _plot_hist(ds, ax, np.nansum(ds[f'{reg}_{v}_MCS_shield'].values, axis=0), 'r--', 'MCS shield')
    _plot_hist(ds, ax, np.nansum(ds[f'{reg}_{v}_cloud_core'].values, axis=0), 'b-', 'cloud core')
    _plot_hist(ds, ax, np.nansum(ds[f'{reg}_{v}_cloud_shield'].values, axis=0), 'b--', 'cloud shield')
    _plot_hist(ds, ax, np.nansum(ds[f'{reg}_{v}_env'].values, axis=0), 'k-', 'env')
    ax.legend()
    if log:
        ax.set_yscale('log')
    if v == 'cape':
        ax.set_xlabel('CAPE (J kg$^{-1}$)')
    elif v == 'tcwv':
        ax.set_xlabel('TCWV (mm)')


def plot_hist_probs(ds, ax=None, reg='all', v='cape', s=None):
    if s is None:
        if v == 'cape':
            s = slice(0, 500, None)
        elif v == 'tcwv':
            s = slice(0, 101, None)
    if ax is None:
        plt.figure(figsize=(10, 10))
        ax = plt.gca()

    counts = np.zeros((5, ds[f'{reg}_{v}_MCS_core'].shape[1]))
    counts[0] = np.nansum(ds[f'{reg}_{v}_MCS_core'].values, axis=0)
    counts[1] = np.nansum(ds[f'{reg}_{v}_MCS_shield'].values, axis=0)
    counts[2] = np.nansum(ds[f'{reg}_{v}_cloud_core'].values, axis=0)
    counts[3] = np.nansum(ds[f'{reg}_{v}_cloud_shield'].values, axis=0)
    counts[4] = np.nansum(ds[f'{reg}_{v}_env'].values, axis=0)
    probs = counts / counts.sum(axis=0)[None, :]

    ax.set_title(f'{v.upper()} probabilities')
    ax.plot(ds[f'{v}_hist_mids'].values[s], probs[0][s], 'r-', label='MCS core')
    ax.plot(ds[f'{v}_hist_mids'].values[s], probs[1][s], 'r--', label='MCS shield')
    ax.plot(ds[f'{v}_hist_mids'].values[s], probs[2][s], 'b-', label='cloud core')
    ax.plot(ds[f'{v}_hist_mids'].values[s], probs[3][s], 'b--', label='cloud shield')
    ax.plot(ds[f'{v}_hist_mids'].values[s], probs[4][s], 'k-', label='env')
    ax.legend()

    if v == 'cape':
        ax.set_xlabel('CAPE (J kg$^{-1}$)')
    elif v == 'tcwv':
        ax.set_xlabel('TCWV (mm)')


def plot_hists_for_var(ds, var):
    fig, axes = plt.subplots(2, 3, sharex=True)
    fig.set_size_inches((20, 10))
    for ax, reg in zip(axes[0], ['all', 'land', 'ocean']):
        plot_hist(ds, ax=ax, reg=reg, v=var, log=False)
        if var == 'cape':
            ax.set_xlim((0, 2500))
            ax.set_ylim((0, 0.0014))
            ax.set_title(f'CAPE {reg}')
        else:
            ax.set_ylim((0, 0.08))
            ax.set_title(f'TCWV {reg}')

    for ax, reg in zip(axes[1], ['all', 'land', 'ocean']):
        plot_hist_probs(ds, reg=reg, v=var, ax=ax)
        if var == 'cape':
            ax.set_xlim((0, 2500))
            ax.set_title(f'CAPE {reg}')
        else:
            ax.set_title(f'TCWV {reg}')
    return fig, axes


class PlotCombineConditionalERA5Hist(TaskRule):
    rule_inputs = {'hist_{year}': (PATHS['outdir'] / 'conditional_era5_histograms' /
                                   '{year}' /
                                   'yearly_hist_{year}.nc')
                   for year in YEARS}
    rule_outputs = {'cape': (PATHS['figdir'] / 'conditional_era5_histograms' /
                             'yearly_hist_cape.png'),
                    'tcwv': (PATHS['figdir'] / 'conditional_era5_histograms' /
                             'yearly_hist_tcwv.png')
    depends_on = [plot_hist, plot_hist_probs, plot_hists_for_var]

    def rule_run(self):
        with xr.open_mfdataset(self.inputs.values()) as ds:
            print(ds)

	    for var in ['cape', 'tcwv']:
                plot_hists_for_var(ds, var)
                plt.savefig(self.outputs[var])
