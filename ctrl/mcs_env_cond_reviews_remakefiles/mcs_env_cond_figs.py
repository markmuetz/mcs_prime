"""All figure plotting code for https://github.com/markmuetz/MCS_env_cond

"Environmental Conditions Affecting Global Mesoscale Convective System Occurrence"
JAS 2024.

Terminology:
* natural MCS: MCS that has not formed from a prev MCS split.
* MCS regions: 5 MCS regions: MCS core, MCS shield, non-MCS core, non-MCS shield, environment.

My biggest regret with this code is not splitting the processing from the plotting.
I.e. they are done together in one task. This means it can take a long time (hours) to produce one fig.
Not ideal! Unfortunately it is not straightforward to separate the processing/plotting code.
"""
from itertools import product
from pathlib import Path
import string

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import linregress
import xarray as xr

from remake import Remake, TaskRule
from remake.util import format_path as fmtp

from mcs_prime import PATHS, McsTracks, mcs_mask_plotter
import mcs_prime.mcs_prime_config_util as cu


slurm_config = {'account': 'short4hr', 'queue': 'short-serial-4hr', 'mem': 64000}
fig_plotting = Remake(config=dict(slurm=slurm_config, content_checks=False))

levels = [
    114,  # 850 hPa
    105,  # 700 hPa
    101,  # 600 hPa
]

DIV_ERA5VARS = (
    [f'div_ml{level}' for level in levels] +
    [f'vertically_integrated_div_ml{level}_surf' for level in levels]
)

FMT_PATH_MCS_ENV_COND_REVS_LIFECYCLE_MCS_LOCAL_ENV = (
    PATHS['outdir'] / 'mcs_env_cond_reviews' / 'mcs_local_envs' / '{year}' / '{month:02d}' / 'lifecycle_mcs_local_env_{year}_{month:02d}.{var}.nc'
)

SUBFIG_SQ_SIZE = 9  # cm

# This are the years 2001-2020, skipping 2003-6 (as in Feng et al. 2021).
YEARS = cu.YEARS

STANDARD_E5VARS = [
    'cape',
    'cin',
    'tcwv',
    'RHlow',
    'RHmid',
    'vertically_integrated_moisture_flux_div',
    *DIV_ERA5VARS,
]
print(STANDARD_E5VARS)

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
        'div_ml114': 'conv. at 850 hPa (10$^{-5}$ s$^{-1}$)',
        'div_ml105': 'conv. at 700 hPa (10$^{-5}$ s$^{-1}$)',
        'div_ml101': 'conv. at 600 hPa (10$^{-5}$ s$^{-1}$)',
        'vertically_integrated_div_ml114_surf': 'VI conv. surf. to 850 hPa (m s$^{-1}$)',
        'vertically_integrated_div_ml105_surf': 'VI conv. surf. to 700 hPa (m s$^{-1}$)',
        'vertically_integrated_div_ml101_surf': 'VI conv. surf. to 600 hPa (m s$^{-1}$)',
    }
    return labels[var]


def plot_grouped_precursor_mean_val(ax, grouped_data_dict, show_spread=False):
    """grouped_data_dict key is the label for the data, and
    each entry is a dict containing:
    * xvals (req), (top-level)
    * xr.DataArray (req),
    * plot_kwargs,
    * spread_plot_kwargs,
    """
    xvals = grouped_data_dict.pop('xvals')
    for label, data_dict in grouped_data_dict.items():
        data_array = data_dict['data_array']
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
    ax.set_xlim(xvals[0], xvals[-1])


def build_track_filters(tracks, nat_vs_all=False, combine_eq_tropics=False, thresh_land=0.5, thresh_sea=0.5):
    print('  natural')
    # Only include "natural" MCSs - those that do not form by splitting from an existing one.
    # Note, the number of natural MCSs is taken as the baseline for working out
    # how many MCSs are filtered into a particular group (natural.sum()).
    # NOTE there is a difference between tracks that were loaded with xr.open_mfdataset (as used here)
    # and xr.open_dataset (in e.g., notebooks). Former sets this to -9999, latter to np.nan.
    natural = tracks.dstracks.start_split_cloudnumber.values == -9999
    if nat_vs_all:
        filter_vals = {'natural': {'all': np.ones_like(natural, dtype=bool), 'natural': natural}}
    else:
        filter_vals = {'natural': {'natural': natural}}

    mean_lat = np.nanmean(tracks.dstracks.meanlat.values, axis=1)
    if combine_eq_tropics:
        print('  equator-tropics')
        mean_lat = np.nanmean(tracks.dstracks.meanlat.values, axis=1)
        # Just apply analysis over equator-tropics.
        filter_vals['equator-tropics-extratropics'] = {
            'equatorial-tropics': (mean_lat <= 30) & (mean_lat >= -30),
        }
    else:
        print('  equator-tropics-extratropics')
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
    filter_vals['land-sea'] = {
        'sea': mean_landfrac <= thresh_sea,
        'land': mean_landfrac > thresh_land,
    }

    # Build combinartorial product of filters.
    # E.g. first and second items could be:
    # (natural, equator, land),
    # (natural, equator, sea),
    # ...
    filter_key_combinations = list(
        product(
            filter_vals['natural'].keys(),
            filter_vals['equator-tropics-extratropics'].keys(),
            filter_vals['land-sea'].keys(),
        )
    )
    # Typical construct is to loop over filter_key_combinations, then apply each filter using gen_full_filter.
    return filter_vals, filter_key_combinations, natural


def gen_full_filter(filter_vals, filter_keys):
    full_filter = None
    for filter_name, key in zip(filter_vals.keys(), filter_keys):
        if full_filter is None:
            full_filter = filter_vals[filter_name][key]
        else:
            full_filter = full_filter & filter_vals[filter_name][key]
    return full_filter


class PlotCombinedMcsLocalEnvPrecursorMeanValueFiltered(TaskRule):
    """Used for fig02.pdf, supp_fig02.pdf

    Plot the composite local envs for MCSs over their lifetimes.
    i.e. from 24 h before MCS init (technically DCI), to 72 h after.
    Only natural MCSs.
    Split by lat band, land-sea.
    Use standard vars."""

    @staticmethod
    def rule_inputs(years, radius, mode):
        inputs = {}
        for year in years:
            inputs.update(
                {
                    f'mcs_local_env_{year}_{month}': fmtp(cu.FMT_PATH_LIFECYCLE_MCS_LOCAL_ENV, year=year, month=month)
                    for month in cu.MONTHS
                }
            )
            inputs.update(
                {
                    f'mcs_local_env_{year}_{month}_{var}': fmtp(FMT_PATH_MCS_ENV_COND_REVS_LIFECYCLE_MCS_LOCAL_ENV, year=year, month=month, var=var)
                    for month in cu.MONTHS
                    for var in DIV_ERA5VARS
                }
            )
            inputs[f'tracks_{year}'] = cu.fmt_mcs_stats_path(year)
        return inputs

    @staticmethod
    def rule_outputs(years, radius, mode):
        ystr = cu.fmt_ystr(years)
        if mode == 'natural':
            return {
                'fig': (
                    PATHS['figdir']
                    / 'mcs_env_cond_reviews' / 'mcs_env_cond_figs'
                    / f'combined_filtered_mcs_local_env_precursor_mean_{ystr}.radius-{radius}.pdf'
                )
            }
        else:
            return {
                'fig': (
                    PATHS['figdir']
                    / 'mcs_env_cond_reviews' / 'mcs_env_cond_figs'
                    / f'combined_filtered_mcs_local_env_precursor_mean_{ystr}.radius-{radius}.{mode}.pdf'
                )
            }

    depends_on = [
        plot_grouped_precursor_mean_val,
    ]

    var_matrix = {
        'years': [[2020]],
        'radius': [200],
        'mode': ['natural'],
    }

    def rule_run(self):
        print('Open datasets')
        ds_full = xr.open_mfdataset(
            [p for k, p in self.inputs.items() if k.startswith('mcs_local_env_')],
            # I used this originally, because it was used in Zhe Feng's code.
            # BUT it causes an exception here, and I don't think it's necessary
            # (I've tested in a notebook and am happy with this).
            # combine='nested', concat_dim='tracks'
        )
        ds_full['tracks'] = np.arange(0, ds_full.dims['tracks'], 1, dtype=int)
        print(ds_full)

        tracks_paths = [p for k, p in self.inputs.items() if k.startswith('tracks_')]
        tracks = McsTracks.mfopen(tracks_paths, None)
        print(tracks.dstracks)

        # Build filters. Each is determined by the MCS tracks dataset and applied
        # to the data in ds_full.
        print('Build filters')
        nat_vs_all = self.mode == 'nat_vs_all'
        filter_vals, filter_key_combinations, natural = build_track_filters(tracks, nat_vs_all=nat_vs_all)

        n_hours = 73
        # Set up the figure (using seaborn theming).
        sns.set_theme(palette='deep')
        sns.set_style('ticks')
        sns.set_context('paper')
        fig, axes = plt.subplots(3, 3, layout='constrained', sharex=True)
        fig.set_size_inches(cm_to_inch(SUBFIG_SQ_SIZE * 3, 0.6 * SUBFIG_SQ_SIZE * 4))
        for ax in axes.flatten()[[1, 2]]:
            ax.axis('off')
        flat_axes = axes.flatten()[[0, 3, 4, 5, 6, 7, 8]]  # 7 ax.

        # N.B. works even when only one key, as just gets 1 from 2nd list.
        linewidths = dict(zip(filter_vals['natural'].keys(), [1, 2.5]))
        # Colours determined by eq-trop-ET.
        colours = dict(
            zip(filter_vals['equator-tropics-extratropics'].keys(), plt.rcParams['axes.prop_cycle'].by_key()['color'])
        )
        # Linestyles determined by land-sea.
        linestyles = dict(zip(filter_vals['land-sea'].keys(), ['-', '--']))

        for i, (ax, var) in enumerate(zip(flat_axes, ['vertically_integrated_moisture_flux_div'] + DIV_ERA5VARS)):
            print(var)
            data_array = ds_full[f'mean_{var}'].sel(radius=self.radius).isel(times=slice(0, n_hours)).load()

            if var == 'vertically_integrated_moisture_flux_div':
                data_array = -data_array * 1e4
                ylabel = 'MFC (10$^{-4}$ kg m$^{-2}$ s$^{-1}$)'
            elif var in DIV_ERA5VARS:
                if var.startswith('div'):
                    data_array = -data_array * 1e5
                else:
                    data_array = -data_array
                ylabel = get_labels(var)
            else:
                ylabel = get_labels(var)

            grouped_data_dict = {'xvals': range(-24, -24 + n_hours)}
            if not nat_vs_all:
                # Apply filters.
                for filter_keys in filter_key_combinations:
                    full_filter = gen_full_filter(filter_vals, filter_keys)

                    plot_kwargs = {
                        'linewidth': linewidths[filter_keys[0]],
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

                plot_grouped_precursor_mean_val(ax, grouped_data_dict)

            # Plot the full data.
            plot_kwargs = {
                'color': 'k',
                'linestyle': ':',
            }
            grouped_data_dict = {'xvals': range(-24, -24 + n_hours)}
            total = natural.sum()
            if nat_vs_all:
                all_name = f'natural ({total} tracks)'
            else:
                all_name = f'all ({total} tracks)'
            grouped_data_dict[all_name] = {
                'data_array': data_array.isel(tracks=natural),
                'ylabel': ylabel,
                'plot_kwargs': plot_kwargs,
            }
            if nat_vs_all:
                plot_kwargs = {
                    'color': 'k',
                    'linestyle': '--',
                }
                total = (~natural).sum()
                grouped_data_dict[f'split ({total} tracks)'] = {
                    'data_array': data_array.isel(tracks=~natural),
                    'ylabel': ylabel,
                    'plot_kwargs': plot_kwargs,
                }
                plot_kwargs = {
                    'color': 'k',
                    'linestyle': '-',
                }
                total = len(natural)
                grouped_data_dict[f'all ({total} tracks)'] = {
                    'data_array': data_array,
                    'ylabel': ylabel,
                    'plot_kwargs': plot_kwargs,
                }

            plot_grouped_precursor_mean_val(ax, grouped_data_dict)

            # Apply a nice title for each ax.
            c = string.ascii_lowercase[i]
            # Saves space to put units in title, also avoids awkard problem of
            # positioning ylabel due to different widths of numbers for e.g. 1.5 vs 335
            ax.set_title(f'{c}) {ylabel}', loc='left')
            ax.axvline(x=0, color='k')
            ax.grid(ls='--', lw=0.5)

        # Set some figure-wide text.
        # ncols = 3 if nat_vs_all else 2
        ncol = 1
        axes[1, -1].legend(loc='lower left', bbox_to_anchor=(0.5, 1.02), framealpha=1, ncol=ncol)
        axes[-1, 1].set_xlabel('time from MCS initiation (hr)')
        # handles, labels = axes[0, 0].get_legend_handles_labels()
        # fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.0, 1.0), framealpha=1)
        plt.savefig(self.outputs[f'fig'])


class PlotCombinedMcsLocalEnvPrecursorMeanValueFilteredRadius(TaskRule):
    """Used for fig03.pdf

    Similar to fig02.pdf, but combine equatorial/tropical MCSs, and show different spatial scales in one fig."""
    # enabled = False
    @staticmethod
    def rule_inputs(years):
        # WARNING: It's important to get the ordering right here.
        # I need to loop over years first, then months, so that order of index matches tracks dataset.
        inputs = {
            f'mcs_local_env_{year}_{month}': fmtp(cu.FMT_PATH_LIFECYCLE_MCS_LOCAL_ENV, year=year, month=month)
            for year in years
            for month in cu.MONTHS
        }
        inputs.update(
            {
                f'mcs_local_env_{year}_{month}_{var}': fmtp(FMT_PATH_MCS_ENV_COND_REVS_LIFECYCLE_MCS_LOCAL_ENV, year=year, month=month, var=var)
                for year in years
                for month in cu.MONTHS
                for var in DIV_ERA5VARS
            }
        )
        for year in years:
            inputs[f'tracks_{year}'] = cu.fmt_mcs_stats_path(year)
        return inputs

    @staticmethod
    def rule_outputs(years):
        ystr = cu.fmt_ystr(years)
        return {
            'fig': (
                PATHS['figdir'] / 'mcs_env_cond_reviews' / 'mcs_env_cond_figs' /
                f'combined_filtered_radius_mcs_local_env_precursor_mean_{ystr}.pdf'
            )
        }

    depends_on = [
        plot_grouped_precursor_mean_val,
    ]

    var_matrix = {
        'years': [[2020]],
    }
    # Running out of time on 4h queue.
    # config = {'slurm': {'queue': 'short-serial', 'mem': 64000, 'max_runtime': '24:00:00', 'account': None}}

    def rule_run(self):
        print('Open datasets')
        ds_full = xr.open_mfdataset(
            [p for k, p in self.inputs.items() if k.startswith('mcs_local_env_')],
            # I used this originally, because it was used in Zhe Feng's code.
            # BUT it causes an exception here, and I don't think it's necessary
            # (I've tested in a notebook and am happy with this).
            # combine='nested', concat_dim='tracks'
        )
        ds_full['tracks'] = np.arange(0, ds_full.dims['tracks'], 1, dtype=int)
        print(ds_full)

        tracks_paths = [p for k, p in self.inputs.items() if k.startswith('tracks_')]
        tracks = McsTracks.mfopen(tracks_paths, None)

        track_start_times = pd.DatetimeIndex(tracks.dstracks.start_basetime.values)

        print('Build filters')
        filter_vals, filter_key_combinations, natural = build_track_filters(tracks, combine_eq_tropics=True)

        N = 73
        nrows = 3
        sns.set_theme(palette='deep')
        sns.set_style('ticks')
        sns.set_context('paper')

        fig, axes = plt.subplots(nrows, 3, layout='constrained', sharex=True)
        fudge_factor = 0.6
        fig.set_size_inches(cm_to_inch(SUBFIG_SQ_SIZE * 3, SUBFIG_SQ_SIZE * nrows * fudge_factor))
        for ax in axes.flatten()[[1, 2]]:
            ax.axis('off')
        flat_axes = axes.flatten()[[0, 3, 4, 5, 6, 7, 8]]  # 7 ax.

        radii = [100, 200, 500, 1000]
        colour_vals = plt.rcParams['axes.prop_cycle'].by_key()['color']
        bgor = [colour_vals[i] for i in [0, 2, 1, 3]]  # blue, green, orange, red for seaborn default colours.
        colours = dict(zip(radii, bgor))
        linestyles = dict(zip(filter_vals['land-sea'].keys(), ['-', '--']))

        for i, (ax, var) in enumerate(zip(flat_axes, ['vertically_integrated_moisture_flux_div'] + DIV_ERA5VARS)):
            print(var)
            grouped_data_dict = {'xvals': range(-24, -24 + N)}
            for filter_keys in filter_key_combinations:
                full_filter = gen_full_filter(filter_vals, filter_keys)

                percentage = full_filter.sum() / natural.sum() * 100
                if i == 0:
                    print(f'{filter_keys[2]} ({percentage:.1f}%)')

                for radius in radii:
                    data_array = ds_full[f'mean_{var}'].sel(radius=radius).isel(times=slice(0, N)).load()

                    label = f'{filter_keys[2]}: {radius} km'

                    if var == 'vertically_integrated_moisture_flux_div':
                        data_array = -data_array * 1e4
                        ylabel = 'MFC (10$^{-4}$ kg m$^{-2}$ s$^{-1}$)'
                    elif var in DIV_ERA5VARS:
                        if var.startswith('div'):
                            data_array = -data_array * 1e5
                        else:
                            data_array = -data_array
                        ylabel = get_labels(var)
                    else:
                        ylabel = get_labels(var)

                    plot_kwargs = {
                        'color': colours[radius],
                        'linestyle': linestyles[filter_keys[2]],
                    }
                    grouped_data_dict[label] = {
                        'data_array': data_array.isel(tracks=full_filter),
                        'ylabel': ylabel,
                        'plot_kwargs': plot_kwargs,
                    }

            plot_grouped_precursor_mean_val(ax, grouped_data_dict)

            c = string.ascii_lowercase[i]
            varname = ylabel[: ylabel.find('(')].strip()
            units = ylabel[ylabel.find('(') :].strip()
            ax.set_title(f'{c}) {ylabel}', loc='left')
            ax.axvline(x=0, color='k')
            ax.grid(ls='--', lw=0.5)

        axes[1, -1].legend(loc='lower left', bbox_to_anchor=(0.8, 0), framealpha=1)
        axes[-1, 1].set_xlabel('time from MCS initiation (hr)')
        plt.savefig(self.outputs[f'fig'])


CORR_TIMES = [-24, -10, -5, 0, 5, 10]
class PlotCorrelationMcsLocalEnvPrecursorMeanValueFilteredDecomp(TaskRule):
    """Used for supp_fig01.pdf"""

    @staticmethod
    def rule_inputs(year, decomp_mode, radius):
        inputs = {
            f'mcs_local_env_{year}_{month}': fmtp(cu.FMT_PATH_LIFECYCLE_MCS_LOCAL_ENV, year=year, month=month)
            for month in cu.MONTHS
        }
        inputs.update(
            {
                f'mcs_local_env_{year}_{month}_{var}': fmtp(FMT_PATH_MCS_ENV_COND_REVS_LIFECYCLE_MCS_LOCAL_ENV, year=year, month=month, var=var)
                for month in cu.MONTHS
                for var in DIV_ERA5VARS
            }
        )
        inputs[f'tracks_{year}'] = cu.fmt_mcs_stats_path(year)
        return inputs

    rule_outputs = {
        **{f'fig_{t}': (
            PATHS['figdir']
            / 'mcs_env_cond_reviews' / 'mcs_env_cond_figs'
            / f'corr_mcs_local_env_precursor_mean_{{year}}.decomp-{{decomp_mode}}.radius-{{radius}}.t={t}.pdf'
        )
        for t in CORR_TIMES},
        **{f'fig2_{t}': (
            PATHS['figdir']
            / 'mcs_env_cond_reviews' / 'mcs_env_cond_figs'
            / f'sel_corr_mcs_local_env_precursor_mean_{{year}}.decomp-{{decomp_mode}}.radius-{{radius}}.t={t}.pdf'
        )
        for t in CORR_TIMES},
        **{f'fig3_{t}': (
            PATHS['figdir']
            / 'mcs_env_cond_reviews' / 'mcs_env_cond_figs'
            / f'corr2_mcs_local_env_precursor_mean_{{year}}.decomp-{{decomp_mode}}.radius-{{radius}}.t={t}.pdf'
        )
        for t in CORR_TIMES}
    }

    depends_on = [
        plot_grouped_precursor_mean_val,
    ]

    var_matrix = {
        'year': [2020],
        'decomp_mode': ['all'],
        # 'radius': [100, 200, 500, 1000],
        'radius': [200],
    }

    def rule_run(self):
        e5vars = cu.EXTENDED_ERA5VARS + DIV_ERA5VARS
        print('Open datasets')
        ds_full = xr.open_mfdataset(
            [p for k, p in self.inputs.items() if k.startswith('mcs_local_env_')],
            # I used this originally, because it was used in Zhe Feng's code.
            # BUT it causes an exception here, and I don't think it's necessary
            # (I've tested in a notebook and am happy with this).
            # combine='nested', concat_dim='tracks'
        )
        ds_full['tracks'] = np.arange(0, ds_full.dims['tracks'], 1, dtype=int)

        tracks_paths = [p for k, p in self.inputs.items() if k.startswith('tracks_')]
        tracks = McsTracks.mfopen(tracks_paths, None)

        track_start_times = pd.DatetimeIndex(tracks.dstracks.start_basetime.values)

        print('Build filters')
        filter_vals, filter_key_combinations, natural = build_track_filters(tracks)

        N = 73
        sns.set_theme(palette='deep')
        sns.set_style('ticks')
        sns.set_context('paper')

        data_dict = {}
        # JUST do for all.
        filter_key_combinations = ['all']
        # filter_key_combinations.append('all')
        filter_keys = 'all'
        for var in e5vars:
            print(var)
            ds_full[f'mean_{var}'].sel(radius=self.radius).load()

        for t in CORR_TIMES:
            print(t, filter_keys)
            # fig.set_size_inches(cm_to_inch(SUBFIG_SQ_SIZE * 4 + 3, SUBFIG_SQ_SIZE * 4))

            data_dict[filter_keys] = {}

            if filter_keys != 'all':
                full_filter = gen_full_filter(filter_vals, filter_keys)
            else:
                full_filter = natural
            percentage = full_filter.sum() / natural.sum() * 100

            for var in e5vars:
                print(var)
                data_array = ds_full[f'mean_{var}'].sel(radius=self.radius, times=t).isel(tracks=full_filter)

                if var == 'vertically_integrated_moisture_flux_div':
                    data_array = -data_array * 1e4
                    label = 'MFC (10$^{-4}$ kg m$^{-2}$ s$^{-1}$)'
                elif var in DIV_ERA5VARS:
                    if var.startswith('div'):
                        data_array = -data_array * 1e5
                    else:
                        data_array = -data_array
                    label = get_labels(var)
                else:
                    label = get_labels(var)
                varname = label[: label.find('(')].strip()

                data_dict[filter_keys][varname] = data_array.values

            df = pd.DataFrame(data_dict[filter_keys])
            for corr_vars in ['all', 'selected']:
                fig, ax = plt.subplots(1, 1, layout='constrained')
                if corr_vars == 'all':
                    fig.set_size_inches(cm_to_inch(SUBFIG_SQ_SIZE * 3 + 3, SUBFIG_SQ_SIZE * 3))
                    corr = df.corr()
                else:
                    fig.set_size_inches(cm_to_inch(SUBFIG_SQ_SIZE * 1.5 + 3, SUBFIG_SQ_SIZE * 1.5))
                    div_vars = [
                        f'{method} {plev} hPa'
                        for method in ['conv. at', 'VI conv. surf. to']
                        for plev in [850, 700, 600]
                    ]

                    corr = df[['MFC'] + div_vars].corr()
                print(corr)
                sns.heatmap(corr, ax=ax, cmap='coolwarm', annot=True, fmt='.2f', vmin=-1, vmax=1)
                ax.set_aspect(1)
                if filter_keys != 'all':
                    ax.set_title(' '.join(filter_keys[1:]) + f' ({percentage:.1f}%)')
                else:
                    ax.set_title(f'all ({natural.sum()} tracks)')
                if corr_vars == 'all':
                    plt.savefig(self.outputs[f'fig_{t}'])
                else:
                    plt.savefig(self.outputs[f'fig2_{t}'])

            g = sns.PairGrid(df, y_vars=['MFC'], x_vars=['conv. at 850 hPa', 'VI conv. surf. to 700 hPa'])
            def hexbin(x, y, color, max_series=None, min_series=None, **kwargs):
                # cmap = sns.light_palette(color, as_cmap=True)
                ax = plt.gca()
                xmin, xmax = min_series[x.name], max_series[x.name]
                ymin, ymax = min_series[y.name], max_series[y.name]
                plt.hexbin(x, y, gridsize=25, extent=[xmin, xmax, ymin, ymax], **kwargs)

            g.map(hexbin, max_series=df.max(), min_series=df.min(), norm=LogNorm())
            plt.savefig(self.outputs[f'fig3_{t}'])


day_range = [
    pd.date_range(f'2020-{m:02d}-01 00:00', f'2020-{m:02d}-01 23:00', freq='H')
    for m in range(1, 13)
]

dates = pd.DatetimeIndex(pd.concat([pd.Series(dti) for dti in day_range]))

class PlotERA5Correlation(TaskRule):
    @staticmethod
    def rule_inputs(cov):
        basedir = f'/gws/nopw/j04/mcs_prime/mmuetz/data/mcs_prime_output/era5_processed/'
        if cov == 'full':
            sel_dates = dates
        elif cov == 'quick':
            sel_dates = dates[:2]

        inputs = {
            f'era5_{v}_{date}': Path(
                basedir +
                f'{date.year}/{date.month:02d}/01/' +
                f'ecmwf-era5_oper_an_ml_{date.year}{date.month:02d}01{date.hour:02d}00.proc_{v}.nc'
            )
            for v in ['div', 'vimfd']
            for date in sel_dates
        }
        return inputs

    @staticmethod
    def rule_outputs(cov):
        return {
            'fig1': (
                PATHS['figdir']
                / 'mcs_env_cond_reviews' / 'mcs_env_cond_figs'
                / f'fig_era5_corr1.{cov}.png'
            ),
            'fig2': (
                PATHS['figdir']
                / 'mcs_env_cond_reviews' / 'mcs_env_cond_figs'
                / f'fig_era5_corr2.{cov}.png'
            ),
        }
        return outputs

    var_matrix = {
        'cov': ['quick', 'full'],
    }


    def rule_run(self):
        ds = xr.open_mfdataset(self.inputs.values()).load()
        print(ds)
        df_data = {}
        for var in ['vertically_integrated_moisture_flux_div'] + DIV_ERA5VARS:
            data_array = ds[var]
            if var == 'vertically_integrated_moisture_flux_div':
                data_array = -data_array * 1e4
                label = 'MFC (10$^{-4}$ kg m$^{-2}$ s$^{-1}$)'
            elif var in DIV_ERA5VARS:
                if var.startswith('div'):
                    data_array = -data_array * 1e5
                else:
                    data_array = -data_array
                label = get_labels(var)
            varname = label[: label.find('(')].strip()
            df_data[varname] = data_array.values.flatten()
        df = pd.DataFrame(df_data)
        print(df)

        div_vars = [
            f'{method} {plev} hPa'
            for method in ['conv. at', 'VI conv. surf. to']
            for plev in [850, 700, 600]
        ]
        g = sns.PairGrid(df, y_vars=['MFC'], x_vars=['conv. at 850 hPa', 'VI conv. surf. to 700 hPa'])
        def hexbin(x, y, color, max_series=None, min_series=None, **kwargs):
            # cmap = sns.light_palette(color, as_cmap=True)
            ax = plt.gca()
            lr = linregress(x, y)
            xmin, xmax = min_series[x.name], max_series[x.name]
            ymin, ymax = min_series[y.name], max_series[y.name]
            lrx = np.array([xmin, xmax])
            lry = lr.slope * lrx + lr.intercept

            plt.hexbin(x, y, gridsize=25, extent=[xmin, xmax, ymin, ymax], **kwargs)
            label = f'slope: {lr.slope:.2f}\nintercept: {lr.intercept:.2f}\nr$^2$: {lr.rvalue**2:.2f}\np: {lr.pvalue:.2f}'
            plt.plot(lrx, lry, 'k--', label=label)
            #print(x.name, y.name)
            #print(label)
            # ax.legend() # <- does nothing for some reason.

        # g.map_diag(sns.histplot, element='poly', log_scale=(None, True))
        # g.map_offdiag(hexbin, max_series=df.max(), min_series=df.min(), norm=LogNorm())
        g.map(hexbin, max_series=df.quantile(0.0001), min_series=df.quantile(0.9999), norm=LogNorm())
        plt.show()
        plt.savefig(self.outputs['fig1'])

        corr = df[['MFC'] + div_vars].corr()
        print(corr)

        fig, ax = plt.subplots(1, 1, layout='constrained')
        sns.heatmap(corr, ax=ax, cmap='coolwarm', annot=True, fmt='.2f', vmin=-1, vmax=1)
        ax.set_aspect(1)
        plt.savefig(self.outputs['fig2'])

