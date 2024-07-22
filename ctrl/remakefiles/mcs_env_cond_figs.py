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
import string

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr

from remake import Remake, TaskRule
from remake.util import format_path as fmtp

from mcs_prime import PATHS, McsTracks, mcs_mask_plotter
import mcs_prime.mcs_prime_config_util as cu


slurm_config = {'account': 'short4hr', 'queue': 'short-serial-4hr', 'mem': 64000}
fig_plotting = Remake(config=dict(slurm=slurm_config, content_checks=False))

SUBFIG_SQ_SIZE = 9  # cm

# This are the years 2001-2020, skipping 2003-6 (as in Feng et al. 2021).
YEARS = cu.YEARS

# These are the 12 standard ERA5(-derived) variables that appear in figures.
STANDARD_E5VARS = [
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


mcs_masks_settings_list = [
    {
        'name': 'tcwv1',
        'time': pd.Timestamp(2020, 1, 1, 6, 30),
        'plot_kwargs': dict(
            var='tcwv',
            extent=(-25, -5, -2.5, 7.5),
            grid_x=[-20, -15, -10],
            grid_y=[-2, 0, 2, 4, 6],
            cbar_kwargs=dict(orientation='horizontal'),
        ),
    },
]
mcs_masks_settings_dict = dict([(s['name'], s) for s in mcs_masks_settings_list])


class PlotMcsMasks(TaskRule):
    """Used for fig01.pdf

    Plot the TCWV field on top of the various MCS/non-MCS masks (Tb = 241, 225K)
    Includes precip field.
    Settings as in mcs_masks_settings_list.
    Basically a wrapper round McsMaskPlotter."""

    @staticmethod
    def rule_inputs(settings_name):
        settings = mcs_masks_settings_dict[settings_name]
        time = settings['time']
        pdata = mcs_mask_plotter.McsMaskPlotterData(pd.DatetimeIndex([time]), [settings['plot_kwargs']['var']])
        inputs = {
            **pdata.tracks_inputs,
            **pdata.e5inputs,
            **pdata.pixel_on_e5_inputs,
        }

        return inputs

    @staticmethod
    def rule_outputs(settings_name):
        settings = mcs_masks_settings_dict[settings_name]
        filename = f'MCS_masks_and_centroids_{settings["time"]}_{settings_name}.pdf'.replace(' ', '_').replace(':', '')
        return {f'fig_{settings_name}': PATHS['figdir'] / 'mcs_env_cond_figs' / filename}

    depends_on = [
        mcs_mask_plotter.McsMaskPlotterData,
        mcs_mask_plotter.McsMaskPlotter,
    ]

    var_matrix = {
        'settings_name': list(mcs_masks_settings_dict.keys()),
    }

    def rule_run(self):
        settings = mcs_masks_settings_dict[self.settings_name]
        time = settings['time']
        print(settings)

        pdata = mcs_mask_plotter.McsMaskPlotterData(pd.DatetimeIndex([time]), [settings['plot_kwargs']['var']])
        pdata.load()

        sns.set_theme(palette='deep')
        sns.set_style('ticks')
        sns.set_context('paper')

        fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, layout='constrained')
        xsize = SUBFIG_SQ_SIZE * 1.8
        ysize = xsize * 2 / 3
        fig.set_size_inches(cm_to_inch(xsize, ysize))

        plotter = mcs_mask_plotter.McsMaskPlotter(pdata)
        plotter.plot(ax, time, **settings['plot_kwargs'])

        plt.savefig(self.outputs[f'fig_{self.settings_name}'])


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
            inputs[f'tracks_{year}'] = cu.fmt_mcs_stats_path(year)
        return inputs

    @staticmethod
    def rule_outputs(years, radius, mode):
        ystr = cu.fmt_ystr(years)
        if mode == 'natural':
            return {
                'fig': (
                    PATHS['figdir']
                    / 'mcs_env_cond_figs'
                    / f'combined_filtered_mcs_local_env_precursor_mean_{ystr}.radius-{radius}.pdf'
                )
            }
        else:
            return {
                'fig': (
                    PATHS['figdir']
                    / 'mcs_env_cond_figs'
                    / f'combined_filtered_mcs_local_env_precursor_mean_{ystr}.radius-{radius}.{mode}.pdf'
                )
            }

    depends_on = [
        plot_grouped_precursor_mean_val,
    ]

    var_matrix = {
        'years': [[2020], YEARS],
        'radius': [100, 200, 500, 1000],
        'mode': ['natural', 'nat_vs_all'],
    }

    def rule_run(self):
        print('Open datasets')
        ds_full = xr.open_mfdataset(
            [p for k, p in self.inputs.items() if k.startswith('mcs_local_env_')], combine='nested', concat_dim='tracks'
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
        fig, axes = plt.subplots(4, 3, layout='constrained', sharex=True)
        fig.set_size_inches(cm_to_inch(SUBFIG_SQ_SIZE * 3, 0.6 * SUBFIG_SQ_SIZE * 4))

        # N.B. works even when only one key, as just gets 1 from 2nd list.
        linewidths = dict(zip(filter_vals['natural'].keys(), [1, 2.5]))
        # Colours determined by eq-trop-ET.
        colours = dict(
            zip(filter_vals['equator-tropics-extratropics'].keys(), plt.rcParams['axes.prop_cycle'].by_key()['color'])
        )
        # Linestyles determined by land-sea.
        linestyles = dict(zip(filter_vals['land-sea'].keys(), ['-', '--']))

        for i, (ax, var) in enumerate(zip(axes.flatten(), STANDARD_E5VARS)):
            print(var)
            data_array = ds_full[f'mean_{var}'].sel(radius=self.radius).isel(times=slice(0, n_hours)).load()

            if var == 'vertically_integrated_moisture_flux_div':
                data_array = -data_array * 1e4
                ylabel = 'MFC (10$^{-4}$ kg m$^{-2}$ s$^{-1}$)'
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
        axes[0, -1].legend(loc='lower left', bbox_to_anchor=(0.5, 1.02), framealpha=1, ncol=ncol)
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
        for year in years:
            inputs[f'tracks_{year}'] = cu.fmt_mcs_stats_path(year)
        return inputs

    @staticmethod
    def rule_outputs(years):
        ystr = cu.fmt_ystr(years)
        return {
            'fig': (
                PATHS['figdir'] / 'mcs_env_cond_figs' / f'combined_filtered_radius_mcs_local_env_precursor_mean_{ystr}.pdf'
            )
        }

    depends_on = [
        plot_grouped_precursor_mean_val,
    ]

    var_matrix = {
        'years': [cu.YEARS, [2020]],
    }
    # Running out of time on 4h queue.
    config = {'slurm': {'queue': 'short-serial', 'mem': 64000, 'max_runtime': '24:00:00', 'account': None}}

    def rule_run(self):
        print('Open datasets')
        ds_full = xr.open_mfdataset(
            [p for k, p in self.inputs.items() if k.startswith('mcs_local_env_')], combine='nested', concat_dim='tracks'
        )
        ds_full['tracks'] = np.arange(0, ds_full.dims['tracks'], 1, dtype=int)

        tracks_paths = [p for k, p in self.inputs.items() if k.startswith('tracks_')]
        tracks = McsTracks.mfopen(tracks_paths, None)

        track_start_times = pd.DatetimeIndex(tracks.dstracks.start_basetime.values)

        print('Build filters')
        filter_vals, filter_key_combinations, natural = build_track_filters(tracks, combine_eq_tropics=True)

        N = 73
        nrows = ((len(STANDARD_E5VARS) - 1) // 3) + 1  # trust me.
        sns.set_theme(palette='deep')
        sns.set_style('ticks')
        sns.set_context('paper')

        fig, axes = plt.subplots(nrows, 3, layout='constrained', sharex=True)
        fudge_factor = 0.6
        fig.set_size_inches(cm_to_inch(SUBFIG_SQ_SIZE * 3, SUBFIG_SQ_SIZE * nrows * fudge_factor))

        radii = [100, 200, 500, 1000]
        colour_vals = plt.rcParams['axes.prop_cycle'].by_key()['color']
        bgor = [colour_vals[i] for i in [0, 2, 1, 3]]  # blue, green, orange, red for seaborn default colours.
        colours = dict(zip(radii, bgor))
        linestyles = dict(zip(filter_vals['land-sea'].keys(), ['-', '--']))

        for i, (ax, var) in enumerate(zip(axes.flatten(), STANDARD_E5VARS)):
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

        axes[0, -1].legend(loc='lower left', bbox_to_anchor=(0.8, 0), framealpha=1)
        axes[-1, 1].set_xlabel('time from MCS initiation (hr)')
        plt.savefig(self.outputs[f'fig'])


class PlotCombinedMcsLocalEnvPrecursorMeanValueFilteredRadiusRatio(TaskRule):
    """Not used for a figure.

    Like PlotCombinedMcsLocalEnvPrecursorMeanValueFilteredRadius/fig03.pdf but shows 200, 500, 1000km diff
    from 100km, scaled by the appropriate amount."""
    @staticmethod
    def rule_inputs(years):
        # WARNING: It's important to get the ordering right here.
        # I need to loop over years first, then months, so that order of index matches tracks dataset.
        inputs = {
            f'mcs_local_env_{year}_{month}': fmtp(cu.FMT_PATH_LIFECYCLE_MCS_LOCAL_ENV, year=year, month=month)
            for year in years
            for month in cu.MONTHS
        }
        for year in years:
            inputs[f'tracks_{year}'] = cu.fmt_mcs_stats_path(year)
        return inputs

    @staticmethod
    def rule_outputs(years):
        ystr = cu.fmt_ystr(years)
        return {
            'fig': (
                PATHS['figdir'] / 'mcs_env_cond_figs' / f'combined_filtered_radius_mcs_local_env_precursor_mean_{ystr}_ratio.pdf'
            )
        }

    depends_on = [
        plot_grouped_precursor_mean_val,
    ]

    var_matrix = {
        'years': [cu.YEARS, [2020]],
    }
    config = {'slurm': {'queue': 'short-serial', 'mem': 64000, 'max_runtime': '24:00:00', 'account': None}}

    def rule_run(self):
        print('Open datasets')
        ds_full = xr.open_mfdataset(
            [p for k, p in self.inputs.items() if k.startswith('mcs_local_env_')], combine='nested', concat_dim='tracks'
        )
        ds_full['tracks'] = np.arange(0, ds_full.dims['tracks'], 1, dtype=int)

        tracks_paths = [p for k, p in self.inputs.items() if k.startswith('tracks_')]
        tracks = McsTracks.mfopen(tracks_paths, None)

        track_start_times = pd.DatetimeIndex(tracks.dstracks.start_basetime.values)

        print('Build filters')
        filter_vals, filter_key_combinations, natural = build_track_filters(tracks, combine_eq_tropics=True)

        N = 73
        nrows = ((len(STANDARD_E5VARS) - 1) // 3) + 1  # trust me.
        sns.set_theme(palette='deep')
        sns.set_style('ticks')
        sns.set_context('paper')

        fig, axes = plt.subplots(nrows, 3, layout='constrained', sharex=True)
        fudge_factor = 0.6
        fig.set_size_inches(cm_to_inch(SUBFIG_SQ_SIZE * 3, SUBFIG_SQ_SIZE * nrows * fudge_factor))

        radii = [200, 500, 1000]
        colour_vals = plt.rcParams['axes.prop_cycle'].by_key()['color']
        bgor = [colour_vals[i] for i in [0, 2, 1, 3]]  # blue, green, orange, red for seaborn default colours.
        colours = dict(zip(radii, bgor[1:]))
        linestyles = dict(zip(filter_vals['land-sea'].keys(), ['-', '--']))

        for i, (ax, var) in enumerate(zip(axes.flatten(), STANDARD_E5VARS)):
            print(var)
            grouped_data_dict = {'xvals': range(-24, -24 + N)}
            for filter_keys in filter_key_combinations:
                full_filter = gen_full_filter(filter_vals, filter_keys)

                percentage = full_filter.sum() / natural.sum() * 100
                if i == 0:
                    print(f'{filter_keys[2]} ({percentage:.1f}%)')

                data_array_100_full = ds_full[f'mean_{var}'].sel(radius=100).isel(times=slice(0, N)).load()
                data_array_100 = data_array_100_full.isel(tracks=full_filter)
                if var == 'vertically_integrated_moisture_flux_div':
                    data_array_100 = -data_array_100 * 1e4

                for radius in radii:
                    data_array = ds_full[f'mean_{var}'].sel(radius=radius).isel(times=slice(0, N)).load()

                    label = f'{filter_keys[2]}: {radius} km'

                    if var == 'vertically_integrated_moisture_flux_div':
                        data_array = -data_array * 1e4
                        ylabel = 'MFC (10$^{-4}$ kg m$^{-2}$ s$^{-1}$)'
                    else:
                        ylabel = get_labels(var)

                    plot_kwargs = {
                        'color': colours[radius],
                        'linestyle': linestyles[filter_keys[2]],
                    }
                    ratio = radius / 100
                    grouped_data_dict[label] = {
                        'data_array': (data_array.isel(tracks=full_filter) - data_array_100) / ratio,
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

        axes[0, -1].legend(loc='lower left', bbox_to_anchor=(0.8, 0), framealpha=1)
        axes[-1, 1].set_xlabel('time from MCS initiation (hr)')
        plt.savefig(self.outputs[f'fig'])


class PlotIndividualMcsLocalEnvPrecursorMeanValueFilteredDecomp(TaskRule):
    """Used for fig04.pdf, fig05.pdf, supp_fig04.pdf, supp_fig05.pdf, supp_fig06.pdf

    As fig02.pdf, fig03.pdf, but just for one (each) variable at a time.
    Each combination of lat band/land-sea gets its own ax.
    """

    @staticmethod
    def rule_inputs(years, decomp_mode, radius, show_spread):
        # WARNING: It's important to get the ordering right here.
        # I need to loop over years first, then months, so that order of index matches tracks dataset.
        inputs = {
            f'mcs_local_env_{year}_{month}': fmtp(cu.FMT_PATH_LIFECYCLE_MCS_LOCAL_ENV, year=year, month=month)
            for year in years
            for month in cu.MONTHS
        }
        for year in years:
            inputs[f'tracks_{year}'] = cu.fmt_mcs_stats_path(year)
        return inputs

    @staticmethod
    def rule_outputs(years, decomp_mode, radius, show_spread):
        ystr = cu.fmt_ystr(years)
        return {
            f'fig_{var}': (
                PATHS['figdir']
                / 'mcs_env_cond_figs'
                / 'indiv'
                / var
                / f'indiv_filtered_decomp_mcs_local_env_precursor_mean_{var}_{ystr}.decomp-{decomp_mode}.radius-{radius}.show_spread-{show_spread}.pdf'
            )
            for var in cu.EXTENDED_ERA5VARS
        }

    depends_on = [
        plot_grouped_precursor_mean_val,
    ]

    var_matrix = {
        # 'years': [cu.YEARS],
        'years': [cu.YEARS, [2020]],
        # 'years': [[2020]],
        'decomp_mode': ['all', 'diurnal_cycle', 'seasonal'],
        'radius': [100, 200, 500, 1000],
        'show_spread': [False, True],
    }

    def rule_run(self):
        e5vars = cu.EXTENDED_ERA5VARS
        print('Open datasets')
        ds_full = xr.open_mfdataset(
            [p for k, p in self.inputs.items() if k.startswith('mcs_local_env_')], combine='nested', concat_dim='tracks'
        )
        ds_full['tracks'] = np.arange(0, ds_full.dims['tracks'], 1, dtype=int)

        tracks_paths = [p for k, p in self.inputs.items() if k.startswith('tracks_')]
        tracks = McsTracks.mfopen(tracks_paths, None)

        track_start_times = pd.DatetimeIndex(tracks.dstracks.start_basetime.values)

        print('Build filters')
        filter_vals, filter_key_combinations, natural = build_track_filters(tracks)

        if self.decomp_mode == 'all':
            n_time_filters = 1
            decomp_filters = [np.ones_like(tracks.dstracks.tracks, dtype=bool)]
        elif self.decomp_mode == 'diurnal_cycle':
            lst_offset = tracks.dstracks.meanlon.values[:, 0] / 360 * 24 * 3600 * 1e3  # in ms.
            lst_track_start_times = track_start_times + lst_offset.astype('timedelta64[ms]')
            hour_groups = [
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8],
                [9, 10, 11],
                [12, 13, 14],
                [15, 16, 17],
                [18, 19, 20],
                [21, 22, 23],
            ]
            n_time_filters = len(hour_groups)
            decomp_filters = [lst_track_start_times.hour.isin(hours) for hours in hour_groups]
        elif self.decomp_mode == 'seasonal':
            seasons = [[12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
            n_time_filters = len(seasons)
            decomp_filters = [track_start_times.month.isin(months) for months in seasons]

        N = 73
        sns.set_theme(palette='deep')
        sns.set_style('ticks')
        sns.set_context('paper')
        # Quite a lot of experimentation here.
        # In the end, I decided on twilight_shifted cyclical colormap,
        # with the compromise being that I set the bg to silver (light grey).
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

            fig_chars = np.array([string.ascii_lowercase[i] for i in range(6)]).reshape(2, 3)
            for (fig_char, ax, filter_keys) in zip(fig_chars.T.flatten(), axes.T.flatten(), filter_key_combinations):
                grouped_data_dict = {'xvals': range(-24, -24 + N)}
                full_filter = gen_full_filter(filter_vals, filter_keys)

                percentage = full_filter.sum() / natural.sum() * 100
                for j in range(n_time_filters):
                    tracks_filter = full_filter & decomp_filters[j]
                    print('  ', var, ' '.join(filter_keys[1:]), tracks_filter.sum(), len(tracks_filter))

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

                plot_grouped_precursor_mean_val(ax, grouped_data_dict, show_spread=self.show_spread)

                # fig_char = string.ascii_lowercase[i]
                varname = ylabel[: ylabel.find('(')].strip()
                units = ylabel[ylabel.find('(') :].strip()
                label = ' '.join(filter_keys[1:])
                ax.set_title(f'{fig_char}) {label} ({percentage:.1f}%)', loc='left')
                ax.axvline(x=0, color='k')
                ax.set_facecolor('silver')
                ax.grid(ls='--', lw=0.5)

                if var == 'vertically_integrated_moisture_flux_div' and self.radius in {100, 200}:
                    ax.set_ylim((-0.5, 5))

            for ax in axes[:, 0]:
                ax.set_ylabel(ylabel)
            axes[-1, 1].set_xlabel('time from MCS initiation (hr)')
            if self.decomp_mode == 'diurnal_cycle':
                norm = mpl.colors.Normalize(vmin=0, vmax=24)
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                plt.colorbar(
                    sm,
                    ax=axes[:, -1],
                    orientation='vertical',
                    ticks=np.arange(24),
                    boundaries=np.linspace(-0.5, 23.5, 9),
                )
            elif self.decomp_mode == 'seasonal':
                norm = mpl.colors.Normalize(vmin=0, vmax=4)
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                labels = ['DJF', 'MAM', 'JJA', 'SON']
                cb = plt.colorbar(
                    sm, ax=axes[:, -1], orientation='vertical', ticks=np.arange(4), boundaries=np.linspace(-0.5, 3.5, 5)
                )
                cb.set_ticklabels(labels)
                cb.ax.tick_params(rotation=90)
            plt.savefig(self.outputs[f'fig_{var}'])
            plt.close('all')


class PlotAllCombinedMcsLocalEnv(TaskRule):
    """Used for fig06.pdf

    This plot is a bit different. Plot the composite 2D local env for each MCS init point.
    Subtract the monthly mean for each field. Idea is to produce a composite, like is done regionally,
    but globally.
    """

    @staticmethod
    def rule_inputs(years, radius):
        inputs = {
            f'mcs_local_env_{year}_{month}': fmtp(
                cu.FMT_PATH_COMBINE_MCS_LOCAL_ENV, year=year, month=month, mode='init'
            )
            for year in years
            for month in cu.MONTHS
        }
        return inputs

    @staticmethod
    def rule_outputs(years, radius):
        ystr = cu.fmt_ystr(years)
        outputs = {
            'fig': (
                PATHS['figdir'] / 'mcs_env_cond_figs' / f'all_combined_mcs_local_env_r{radius}km_init_{ystr}.pdf'
            ),
            'data': (
                PATHS['figdir'] / 'mcs_env_cond_figs' / f'all_combined_mcs_local_env_r{radius}km_init_{ystr}.csv'
            )
        }
        return outputs

    var_matrix = {
        'years': [[2020], cu.YEARS],
        'radius': [100, 200, 500, 1000],
    }

    # Possibly running out of mem?
    # config = {'slurm': {'mem': 512000, 'partition': 'high-mem', 'max_runtime': '24:00:00', 'account': None}}

    def rule_run(self):
        sns.set_theme(palette='deep')
        sns.set_style('ticks')
        sns.set_context('paper')
        radius = self.radius

        data = []

        with xr.open_mfdataset(self.inputs.values()) as ds:
            ds.sel(radius=radius).load()
            ds_rad = ds.sel(radius=radius)
            mask_sum = ds_rad.dist_mask_sum.sum(dim='time')

            print(f'{radius}km')
            fig, axes = plt.subplots(4, 3, subplot_kw={'projection': ccrs.PlateCarree()}, layout='constrained')
            # Size of fig. Each one is 360x120, or aspect of 3.
            # Hence use /3 in height.
            # 1.1 is a fudge factor for colorbar, title...
            fig.set_size_inches(cm_to_inch(SUBFIG_SQ_SIZE * 3, SUBFIG_SQ_SIZE * 4 / 3 * 1.1))
            for i, (ax, var) in enumerate(zip(axes.flatten(), STANDARD_E5VARS)):
                title = f'-  {var}'
                print(title)

                da_var = ds_rad[f'mcs_local_{var}']
                da_mean = ds_rad[var]

                vmax = max(np.max(ds_rad[var].values), np.nanmax(ds_rad[f'mcs_local_{var}'].values))
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

                extent = (0, 360, -60, 60)
                cmap = sns.color_palette('coolwarm', as_cmap=True)
                im = ax.imshow(
                    masked_diff,
                    vmin=vmin,
                    vmax=vmax,
                    cmap=cmap,
                    extent=extent,
                )

                plt.colorbar(im, ax=ax, extend='both')
                ax.coastlines()
                ax.set_ylim((-60, 60))
                if var == 'vertically_integrated_moisture_flux_div':
                    label = 'MFC (10$^{-4}$ kg m$^{-2}$ s$^{-1}$)'
                else:
                    label = get_labels(var)

                gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, color='gray')
                gl.xlocator = mticker.FixedLocator([])
                gl.ylocator = mticker.FixedLocator([-30, -10, 10, 30])
                c = string.ascii_lowercase[i]
                ax.set_title(f'{c}) {label}', loc='left')

                # equality takes into account mask.
                # Number of valid points is given by denom.
                positive_diff = (masked_diff > 0).sum() / (~masked_diff.mask).sum()
                negative_diff = (masked_diff < 0).sum() / (~masked_diff.mask).sum()
                data.append((radius, var, positive_diff, negative_diff))

            plt.savefig(self.outputs[f'fig'])
            df = pd.DataFrame(columns=['radius', 'var', 'positive_diff', 'negative_diff'], data=data)
            df.to_csv(self.outputs['data'])


class PlotGeogNumPoints(TaskRule):
    """Used for supp_fig07.pdf

    Plot number of times each grid point is included at each radius from initiation points.
    """

    @staticmethod
    def rule_inputs(years, radius):
        inputs = {
            f'mcs_local_env_{year}_{month}': fmtp(
                cu.FMT_PATH_COMBINE_MCS_LOCAL_ENV, year=year, month=month, mode='init'
            )
            for year in years
            for month in cu.MONTHS
        }
        return inputs

    @staticmethod
    def rule_outputs(years, radius):
        ystr = cu.fmt_ystr(years)
        outputs = {
            'fig': (
                PATHS['figdir'] / 'mcs_env_cond_figs' / f'num_points_r{radius}km_init_{ystr}.pdf'
            )
        }
        return outputs

    var_matrix = {
        'years': [[2020], cu.YEARS],
        'radius': [100, 200, 500, 1000],
    }

    def rule_run(self):
        sns.set_theme(palette='deep')
        sns.set_style('ticks')
        sns.set_context('paper')

        radius = self.radius

        with xr.open_mfdataset(self.inputs.values()) as ds:
            print(f'{radius}km')
            fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()}, layout='constrained')
            fig.set_size_inches(cm_to_inch(SUBFIG_SQ_SIZE * 3, SUBFIG_SQ_SIZE))
            # title = f'-  {var}'
            # print(title)
            mask_sum = ds.sel(radius=radius).dist_mask_sum.sum(dim='time')
            max_val = mask_sum.values.max()
            extent = (0, 360, -60, 60)

            # AKA the UK cash system levels!
            a = np.array([1, 2, 5])
            b = 10**np.arange(1, 6)
            # E.g. [10, 20, 50, 100...]
            levels = (a[None, :] * b[:, None]).flatten()
            idx_max = np.where(levels > max_val)[0][0]  # index of first value above max_val.
            # Include next value above max_val. E.g. max_val = 150, include 200 as max level.
            levels = levels[:idx_max + 1]
            print(max_val, levels)

            cmap = sns.color_palette('flare', as_cmap=True)
            colours = [cmap(i / len(levels)) for i in range(len(levels))]

            im = ax.contourf(
                mask_sum[::-1],
                levels=levels,
                colors=colours,
                extent=extent,
            )

            plt.colorbar(im, ax=ax, extend='both', label='number of counts')
            ax.coastlines()
            ax.set_ylim((-60, 60))

            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, color='gray')
            gl.xlocator = mticker.FixedLocator([])
            gl.ylocator = mticker.FixedLocator([-30, -10, 10, 30])

            plt.savefig(self.outputs['fig'])


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

    d1 = np.nansum(ds[f'{reg}_{var}_MCS_core'].values, axis=0)
    d2 = np.nansum(ds[f'{reg}_{var}_MCS_shield'].values, axis=0)
    d3 = np.nansum(ds[f'{reg}_{var}_cloud_core'].values, axis=0)
    d4 = np.nansum(ds[f'{reg}_{var}_cloud_shield'].values, axis=0)
    d5 = np.nansum(ds[f'{reg}_{var}_env'].values, axis=0)
    dt = d1 + d2 + d3 + d4 + d5
    _plot_hist(ds, ax, d1, 'r-', 'MCS core')
    _plot_hist(ds, ax, d2, 'r--', 'MCS shield')
    _plot_hist(ds, ax, d3, 'b-', 'non-MCS core')
    _plot_hist(ds, ax, d4, 'b--', 'non-MCS shield')
    _plot_hist(ds, ax, d5, 'k-', 'env')
    _plot_hist(ds, ax, dt, 'k:', 'total')
    if log:
        ax.set_yscale('log')


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
    ax.plot(x, probs[0][s], 'r-', label='MCS core')
    ax.plot(x, probs[1][s], 'r--', label='MCS shield')
    ax.plot(x, probs[2][s], 'b-', label='non-MCS core')
    ax.plot(x, probs[3][s], 'b--', label='non-MCS shield')
    ax.plot(x, probs[4][s], 'k-', label='env')


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

    plot_hist_probs(ds, reg='all', var=var, ax=ax1)

    xlim, ylim, title = xlim_ylim_title[var]
    ax1.set_xlim(xlim)
    ax1.set_ylim((0, 1))


class PlotCombineVarConditionalERA5Hist(TaskRule):
    """Used for fig07.pdf, supp_fig08.pdf

    Plot the conitional PDFs and probabilites for each of the 5 MCS regions.
    """

    @staticmethod
    def rule_inputs(years, e5vars):
        inputs = {
            f'hist_{year}_{month}': fmtp(cu.FMT_PATH_COND_HIST_HOURLY, year=year, month=month, core_method='tb')
            for year in years
            for month in cu.MONTHS
        }
        return inputs

    @staticmethod
    def rule_outputs(years, e5vars):
        ystr = cu.fmt_ystr(years)
        outputs = {
            'fig': (PATHS['figdir'] / 'mcs_env_cond_figs' / f'combined_yearly_hist_{e5vars}_{ystr}_tb.pdf')
        }
        return outputs

    depends_on = [get_labels, plot_hist, plot_hist_probs, plot_combined_hists_for_var]

    var_matrix = {
        'years': [cu.YEARS, [2020]],
        'e5vars': ['all', 'tcwv-RHmid-vertically_integrated_moisture_flux_div'],
    }

    # Running out of time on 4h queue.
    config = {'slurm': {'queue': 'short-serial', 'mem': 64000, 'max_runtime': '24:00:00', 'account': None}}

    def rule_run(self):
        if self.e5vars == 'all':
            e5vars = STANDARD_E5VARS
        else:
            e5vars = self.e5vars.split('-')

        nrows = (((len(e5vars) - 1) // 3) + 1) * 2  # trust me.
        sns.set_theme(palette='deep')
        sns.set_style('ticks')
        sns.set_context('paper')

        fig, axes = plt.subplots(nrows, 3, layout='constrained')
        fudge_factor = 0.8 if self.e5vars == 'all' else 1.0
        fig.set_size_inches(cm_to_inch(SUBFIG_SQ_SIZE * 3, SUBFIG_SQ_SIZE * nrows / 2 * fudge_factor))

        with xr.open_mfdataset(self.inputs.values()) as ds:
            ds.load()
            for i, var in enumerate(e5vars):
                row_idx = (i // 3) * 2
                col_idx = i % 3
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
                ax0.grid(ls='--', lw=0.5)
                ax1.grid(ls='--', lw=0.5)

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

    ax.set_ylim((0, 1))
    ax2.set_ylim((0, None))


class PlotCombineConvectionConditionalERA5Hist(TaskRule):
    """Used for fig08.pdf, supp_fig09.pdf, supp_fig11.pdf

    As fig07.pdf, but this time just the probability of MCS-type convection given ANY convection."""

    @staticmethod
    def rule_inputs(years, core_method, e5vars):
        inputs = {
            f'hist_{year}_{month}': fmtp(cu.FMT_PATH_COND_HIST_HOURLY, year=year, month=month, core_method=core_method)
            for year in years
            for month in cu.MONTHS
        }
        return inputs

    @staticmethod
    def rule_outputs(years, core_method, e5vars):
        ystr = cu.fmt_ystr(years)
        outputs = {
            'fig': (
                PATHS['figdir']
                / 'mcs_env_cond_figs'
                / f'combine_convection_yearly_hist_{e5vars}_{ystr}_{core_method}.pdf'
            )
        }
        return outputs

    depends_on = [plot_convection_hourly_hists]

    var_matrix = {
        'years': [cu.YEARS, [2020]],
        'core_method': ['tb', 'precip'],
        'e5vars': ['all', 'tcwv-RHmid-vertically_integrated_moisture_flux_div'],
    }

    # Running out of time on 4h queue.
    config = {'slurm': {'queue': 'short-serial', 'mem': 64000, 'max_runtime': '24:00:00', 'account': None}}

    def rule_run(self):
        if self.e5vars == 'all':
            e5vars = STANDARD_E5VARS
        else:
            e5vars = self.e5vars.split('-')

        nrows = (((len(e5vars) - 1) // 3) + 1) * 2  # trust me.
        sns.set_theme(palette='deep')
        sns.set_style('ticks')
        sns.set_context('paper')

        fig, axes = plt.subplots(nrows, 3, layout='constrained')
        fudge_factor = 0.8 if self.e5vars == 'all' else 1.0
        fig.set_size_inches(cm_to_inch(SUBFIG_SQ_SIZE * 3, SUBFIG_SQ_SIZE * nrows / 2 * fudge_factor))

        with xr.open_mfdataset(self.inputs.values()) as ds:
            ds.load()

            for i, var in enumerate(e5vars):
                row_idx = (i // 3) * 2
                col_idx = i % 3
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
                ax0.grid(ls='--', lw=0.5)
                ax1.grid(ls='--', lw=0.5)

        if self.e5vars == 'all':
            axes[0, 0].legend(loc='lower left', bbox_to_anchor=(0.8, -0.2), framealpha=1)
        else:
            axes[0, -1].legend(loc='lower left', bbox_to_anchor=(0.8, -0.2), framealpha=1)

        for ax in axes[::2].flatten():
            ax.set_xticklabels([])
        axes[0, 0].set_ylabel('p(MCS conv|conv)')
        axes[1, 0].set_ylabel('pdf')

        fig.align_ylabels(axes)
        plt.savefig(self.outputs['fig'])


def plot_gridpoint_3x_lat_band_hist(ax, ds, var):
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
    xlim, ylim, title = xlim_ylim_title[var]
    ax.set_xlim(xlim)

    if var == 'vertically_integrated_moisture_flux_div':
        x = ds[f'{var}_hist_mids'].values * -1e4
    else:
        x = ds[f'{var}_hist_mids'].values
    lat = ds.latitude.values
    lat_idxs = {
        'equatorial': (lat <= 10) & (lat >= -10),
        'tropics': ((lat <= 30) & (lat > 10)) | ((lat < -10) & (lat >= -30)),
        'extratropics': (lat > 30) | (lat < -30),
    }
    for key, lat_idx in lat_idxs.items():
        d1 = ds[f'{var}_MCS_core'].isel(latitude=lat_idx).values.sum(axis=(0, 1))
        d2 = ds[f'{var}_cloud_core'].isel(latitude=lat_idx).values.sum(axis=(0, 1))
        with np.errstate(invalid='ignore', divide='ignore'):
            d = d1 / (d1 + d2)
        ax.plot(x, d, label=f'{key}')


class PlotCombineConvectionConditionalLatBandERA5Hist(TaskRule):
    """Used for fig09.pdf

    As fig08.pdf, but separately for each lat band (using equator, tropics, extratropics).
    """

    @staticmethod
    def rule_inputs(years, e5vars):
        inputs = {f'hist_{year}': fmtp(cu.FMT_PATH_COMBINED_COND_HIST_GRIDPOINT, year=year) for year in years}
        # inputs['ERA5_land_sea_mask'] = cu.PATH_ERA5_LAND_SEA_MASK
        return inputs

    @staticmethod
    def rule_outputs(years, e5vars):
        ystr = cu.fmt_ystr(years)
        outputs = {
            'fig': (
                PATHS['figdir']
                / 'mcs_env_cond_figs'
                / f'combine_convection_yearly_hist_lat_band_{e5vars}_{ystr}.pdf'
            )
        }
        return outputs

    depends_on = [plot_gridpoint_3x_lat_band_hist]

    var_matrix = {
        # 'years': [[2020], cu.YEARS],
        'years': [[2020], [2018, 2019, 2020]],
        'years': [[2020], [2018, 2019, 2020], cu.YEARS],
        # 'e5vars': ['all', 'tcwv-RHmid-vertically_integrated_moisture_flux_div'],
        'e5vars': ['tcwv-RHmid-vertically_integrated_moisture_flux_div'],
    }

    # Running out of time on 4h queue.
    # Running out of mem as well!
    # config = {'slurm': {'queue': 'short-serial', 'mem': 64000, 'max_runtime': '24:00:00', 'account': None}}
    config = {'slurm': {'mem': 512000, 'partition': 'high-mem', 'max_runtime': '24:00:00', 'account': None}}

    def rule_run(self):
        if self.e5vars == 'all':
            e5vars = STANDARD_E5VARS
        else:
            e5vars = self.e5vars.split('-')

        nrows = ((len(e5vars) - 1) // 3) + 1  # trust me.
        sns.set_theme(palette='deep')
        sns.set_style('ticks')
        sns.set_context('paper')

        fig, axes = plt.subplots(nrows, 3, layout='constrained', sharey=True)
        fudge_factor = 0.8 if self.e5vars == 'all' else 1.0
        fig.set_size_inches(cm_to_inch(SUBFIG_SQ_SIZE * 3, SUBFIG_SQ_SIZE * nrows / 2 * fudge_factor))

        with xr.open_mfdataset(self.inputs.values(), combine='nested', concat_dim='time') as ds:
            for var in e5vars:
                ds[f'{var}_MCS_core'].load()
                ds[f'{var}_cloud_core'].load()

            for i, var in enumerate(e5vars):
                row_idx = i // 3
                col_idx = i % 3
                print(var, row_idx, col_idx)
                if nrows == 1:
                    ax = axes[col_idx]
                else:
                    ax = axes[row_idx, col_idx]
                plot_gridpoint_3x_lat_band_hist(ax, ds.sum(dim='time'), var)
                ax.set_ylim((0, 1))

                if var == 'vertically_integrated_moisture_flux_div':
                    label = 'MFC (10$^{-4}$ kg m$^{-2}$ s$^{-1}$)'
                else:
                    label = get_labels(var)
                c = string.ascii_lowercase[i]
                ax.set_title(f'{c}) {label}', loc='left')
                ax.grid(ls='--', lw=0.5)

        if self.e5vars == 'all':
            axes[0, 0].legend(loc='lower left', bbox_to_anchor=(0.8, -0.2), framealpha=1)
        else:
            axes[0].set_ylabel('p(MCS conv|conv)')
            axes[-1].legend(loc='lower left', bbox_to_anchor=(0.8, 0.02), framealpha=1)

        fig.align_ylabels(axes)
        plt.savefig(self.outputs['fig'])


class PlotCorrelationMcsLocalEnvPrecursorMeanValueFilteredDecomp(TaskRule):
    """Used for supp_fig01.pdf"""

    @staticmethod
    def rule_inputs(year, decomp_mode, radius):
        inputs = {
            f'mcs_local_env_{year}_{month}': fmtp(cu.FMT_PATH_LIFECYCLE_MCS_LOCAL_ENV, year=year, month=month)
            for month in cu.MONTHS
        }
        inputs[f'tracks_{year}'] = cu.fmt_mcs_stats_path(year)
        return inputs

    rule_outputs = {
        f'fig_{i}': (
            PATHS['figdir']
            / 'mcs_env_cond_figs'
            / f'corr_mcs_local_env_precursor_mean_{{year}}.decomp-{{decomp_mode}}.radius-{{radius}}.{i}.pdf'
        )
        for i in range(7)
    }

    depends_on = [
        plot_grouped_precursor_mean_val,
    ]

    var_matrix = {
        'year': YEARS[-4:],
        'decomp_mode': ['all'],
        'radius': [100, 200, 500, 1000],
    }

    def rule_run(self):
        e5vars = cu.EXTENDED_ERA5VARS
        print('Open datasets')
        ds_full = xr.open_mfdataset(
            [p for k, p in self.inputs.items() if k.startswith('mcs_local_env_')], combine='nested', concat_dim='tracks'
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
        filter_key_combinations.append('all')
        for i, filter_keys in enumerate(filter_key_combinations):
            print(i, filter_keys)
            fig, ax = plt.subplots(1, 1, layout='constrained')
            fig.set_size_inches(cm_to_inch(SUBFIG_SQ_SIZE * 2 + 3, SUBFIG_SQ_SIZE * 2))

            data_dict[filter_keys] = {}

            if filter_keys != 'all':
                full_filter = gen_full_filter(filter_vals, filter_keys)
            else:
                full_filter = natural
            percentage = full_filter.sum() / natural.sum() * 100

            for var in e5vars:
                print(var)
                ds_full[f'mean_{var}'].sel(radius=self.radius, times=0).load()
                data_array = ds_full[f'mean_{var}'].sel(radius=self.radius, times=0).isel(tracks=full_filter)

                if var == 'vertically_integrated_moisture_flux_div':
                    data_array = -data_array * 1e4
                    label = 'MFC (10$^{-4}$ kg m$^{-2}$ s$^{-1}$)'
                else:
                    label = get_labels(var)
                varname = label[: label.find('(')].strip()

                data_dict[filter_keys][varname] = data_array.values

            df = pd.DataFrame(data_dict[filter_keys])
            corr = df.corr()
            sns.heatmap(corr, ax=ax, cmap='coolwarm', annot=True, fmt='.2f', vmin=-1, vmax=1)
            ax.set_aspect(1)
            if filter_keys != 'all':
                ax.set_title(' '.join(filter_keys[1:]) + f' ({percentage:.1f}%)')
            else:
                ax.set_title(f'all ({natural.sum()} tracks)')
            plt.savefig(self.outputs[f'fig_{i}'])


class GenDataForConvectionConditionalERA5HistSeasonalDC(TaskRule):
    """Data for supp_fig10.pdf

    Use the diurnal cycle data to produce MCS-type convection conditional on any convection data.
    Can also be used to generate seasonal data as well.
    Split DC into 8 groups, 0-2, 3-5...
    Seasonal into 4 seasons.
    """
    @staticmethod
    def rule_inputs(e5vars, years):
        # Note, the DC data can be used to generate seasonal data as well.
        # This is because it is the same as the hourly data when viewed over the course of
        # one complete day (or season).
        inputs = {
            f'hist_{year}_{month}': fmtp(cu.FMT_PATH_COND_HIST_DC, year=year, month=month, core_method='tb')
            for year in cu.YEARS
            for month in cu.MONTHS
        }
        return inputs

    @staticmethod
    def rule_outputs(e5vars, years):
        ystr = cu.fmt_ystr(years)
        outputs = {
            'season_dc_data': (PATHS['figdir'] / 'fig_data' /
                               f'convection_combined_hist_{e5vars}_{ystr}.season_dc.nc')
        }
        return outputs

    var_matrix = {
        'e5vars': [
            'tcwv-RHmid-vertically_integrated_moisture_flux_div',
        ],
        'years': [[2020], cu.YEARS],
    }

    def rule_run(self):

        e5vars = self.e5vars.split('-')
        mcs_regs = ['MCS_core', 'cloud_core']
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
        print(coords)
        coords['season'] = np.arange(4)
        coords['diurnal_cycle'] = np.arange(8)

        data_vars = {}
        for var in e5vars:
            blank_hist_season_data = np.zeros((4, 100))
            blank_hist_dc_data = np.zeros((8, 100))
            season_prob_key = f'all_{var}_MCS_conv_given_conv_prob_season_data'
            dc_prob_key = f'all_{var}_MCS_conv_given_conv_prob_diurnal_cycle_data'
            data_vars[season_prob_key] = (
                ('season', f'{var}_hist_mid'), blank_hist_season_data.copy()
            )
            data_vars[dc_prob_key] = (
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
        hour_groups = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [9, 10, 11],
            [12, 13, 14],
            [15, 16, 17],
            [18, 19, 20],
            [21, 22, 23],
        ]
        for i, season in enumerate(['djf', 'mam', 'jja', 'son']):
            time_filters[('season', i)] = times.month.isin(month_lookup[season])
        for i, hg in enumerate(hour_groups):
            time_filters[('diurnal_cycle', i)] = times.hour.isin(hg)

        for var in e5vars:
            for key, time_filter in time_filters.items():
                filter_type, filter_idx = key
                print(var, filter_type, filter_idx)
                # Apply filter.
                ds_filtered = ds.isel(time=time_filter)

                d1 = np.nansum(ds_filtered[f'all_{var}_MCS_core'].values, axis=0)
                d2 = np.nansum(ds_filtered[f'all_{var}_cloud_core'].values, axis=0)
                with np.errstate(invalid='ignore', divide='ignore'):
                    d = d1 / (d1 + d2)
                prob_key = f'all_{var}_MCS_conv_given_conv_prob_{filter_type}_data'
                dsout[prob_key][filter_idx] = d

        cu.to_netcdf_tmp_then_copy(dsout, self.outputs['season_dc_data'])


class FigPlotCombineVarConditionalERA5HistSeasonalDC(TaskRule):
    """Used for supp_fig10.pdf

    Heavy lifting is done by GenDataForConvectionConditionalERA5HistSeasonalDC"""
    rule_inputs = GenDataForConvectionConditionalERA5HistSeasonalDC.rule_outputs
    @staticmethod
    def rule_outputs(e5vars, years):
        ystr = cu.fmt_ystr(years)
        outputs = {
            'fig_season_dc': (PATHS['figdir'] / 'mcs_env_cond_figs' /
                              f'conv_conditional_combined_hist_{e5vars}_{ystr}_season_dc.pdf'),
        }
        return outputs

    var_matrix = {
        'e5vars': [
            'tcwv-RHmid-vertically_integrated_moisture_flux_div',
        ],
        'years': [[2020], cu.YEARS],
    }


    def rule_run(self):
        e5vars = self.e5vars.split('-')

        xlims = {
            'cape': (0, 2500),
            'cin': (0, 500),
            'tcwv': (0, 80),
            'shear_0': (0, 30),
            'shear_1': (0, 30),
            'shear_2': (0, 30),
            'shear_3': (0, 30),
            'RHlow': (0, 1),
            'RHmid': (0, 1),
            'theta_e_mid': (300, 360),
            'vertically_integrated_moisture_flux_div': (-12, 12),
            'delta_3h_cape': (-300, 300),
            'delta_3h_tcwv': (-20, 20),
        }

        with xr.open_mfdataset(self.inputs['season_dc_data']) as ds:
            ds.load()

        fig, axes = plt.subplots(2, 3, sharex='col', sharey=True, layout='constrained')
        fig.set_size_inches(cm_to_inch(SUBFIG_SQ_SIZE * 3, 0.9 * SUBFIG_SQ_SIZE * 2))

        sns.set_theme(palette='deep')
        sns.set_style('ticks')
        sns.set_context('paper')
        cmap = mpl.colormaps['twilight_shifted']

        for axrow, mode in zip(axes, ['season', 'diurnal_cycle']):
            indices = np.arange(len(ds[mode]))

            for i in indices:
                c = cmap(i / len(indices))
                for ax, var in zip(axrow, e5vars):
                    print(var, mode, i)
                    kwargs = {mode: i}
                    ds_filtered = ds.isel(**kwargs)

                    prob_key = f'all_{var}_MCS_conv_given_conv_prob_{mode}_data'
                    if var == 'vertically_integrated_moisture_flux_div':
                        x = ds[f'{var}_hist_mids'].values * -1e4
                    else:
                        x = ds[f'{var}_hist_mids'].values
                    p = ax.plot(x, ds_filtered[prob_key].values, color=c)

                    xlim = xlims[var]
                    ax.set_xlim(xlim)
                    ax.set_ylim((0, 1))

                    ax.set_facecolor('silver')
                    ax.grid(ls='--', lw=0.5)

            if mode == 'diurnal_cycle':
                norm = mpl.colors.Normalize(vmin=0, vmax=24)
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                plt.colorbar(
                    sm,
                    ax=axrow,
                    orientation='vertical',
                    ticks=np.arange(24),
                    boundaries=np.linspace(-0.5, 23.5, 9),
                )
            elif mode == 'season':
                norm = mpl.colors.Normalize(vmin=0, vmax=4)
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                labels = ['DJF', 'MAM', 'JJA', 'SON']
                cb = plt.colorbar(
                    sm, ax=axrow, orientation='vertical', ticks=np.arange(4), boundaries=np.linspace(-0.5, 3.5, 5)
                )
                cb.set_ticklabels(labels)
                cb.ax.tick_params(rotation=90)

        axcount = 0
        for axrow, mode in zip(axes, ['season', 'diurnal_cycle']):
            for ax, var in zip(axrow, e5vars):
                figchar = string.ascii_lowercase[axcount]
                axcount += 1
                if var == 'vertically_integrated_moisture_flux_div':
                    label = 'MFC (10$^{-4}$ kg m$^{-2}$ s$^{-1}$)'
                else:
                    label = get_labels(var)
                if mode == 'season':
                    label = 'seasonal ' + label
                elif mode == 'diurnal_cycle':
                    label = 'diurnal cycle ' + label
                ax.set_title(f'{figchar}) {label}', loc='left')

        plt.savefig(self.outputs[f'fig_season_dc'])


class FigPlotMcsLifetimes(TaskRule):
    """Used for supp_fig03.pdf

    Plot the number of MCSs surviving to a given duration, for each lat band/land-sea, and all."""
    @staticmethod
    def rule_inputs(years):
        inputs = {}
        for year in years:
            inputs[f'tracks_{year}'] = cu.fmt_mcs_stats_path(year)
        return inputs

    @staticmethod
    def rule_outputs(years):
        ystr = cu.fmt_ystr(years)
        outputs = {
            'fig': (PATHS['figdir'] / 'mcs_env_cond_figs' /
                    f'mcs_lifetimes_{ystr}.pdf'),
        }
        return outputs

    var_matrix = {
        'years': [[2020], cu.YEARS],
    }


    def rule_run(self):
        tracks_paths = [p for k, p in self.inputs.items() if k.startswith('tracks_')]
        tracks = McsTracks.mfopen(tracks_paths, None)
        print(tracks.dstracks)

        filter_vals, filter_key_combinations, natural = build_track_filters(tracks)

        # Colours determined by lat band.
        colours = dict(zip(
            filter_vals['equator-tropics-extratropics'].keys(),
            plt.rcParams['axes.prop_cycle'].by_key()['color']
        ))
        # Linestyles determined by land-sea.
        linestyles = dict(zip(
            filter_vals['land-sea'].keys(),
            ['-', '--']
        ))
        grouped_data_dict = {}
        # Apply filters.
        for filter_keys in filter_key_combinations:
            full_filter = gen_full_filter(filter_vals, filter_keys)

            plot_kwargs = {
                'color': colours[filter_keys[1]],
                'linestyle': linestyles[filter_keys[2]],
            }
            percentage = full_filter.sum() / natural.sum() * 100
            label = ' '.join(filter_keys[1:]) + f' ({percentage:.1f}%)'
            grouped_data_dict[label] = {
                'ds': tracks.dstracks.isel(tracks=full_filter),
                'label': label,
                'plot_kwargs': plot_kwargs,
            }

        grouped_data_dict[f'all ({natural.sum()} tracks)'] = {
            'ds': tracks.dstracks,
            'label': '',
            'plot_kwargs': {'color': 'k', 'linestyle': ':'},
        }

        sns.set_theme(palette='deep')
        sns.set_style('ticks')
        sns.set_context('paper')

        plt.figure(figsize=(10, 6), layout='constrained')
        for k, v in grouped_data_dict.items():
            c, ls = v['plot_kwargs']['color'], v['plot_kwargs']['linestyle']
            mean_duration = v['ds'].track_duration.values.mean()
            label = k + f' - mean {mean_duration:.1f}h'
            plt.hist(
                v['ds'].track_duration.values,
                bins=np.linspace(0.5, 100.5, 101),
                density=True,
                cumulative=-1,
                histtype='step',
                label=label,
                color=c,
                linestyle=ls
            )
            plt.axvline(x=mean_duration, color=c, ls=ls)
            duration_gt40_frac = (v['ds'].track_duration.values > 40).sum() / len(v['ds'].track_duration.values)
            print(label, f'gt40h: {duration_gt40_frac * 100:.1f}%')

        plt.legend()
        plt.xlim((0, 100))
        plt.axvline(x=4, ls='-', lw=1, color='k')
        plt.grid(ls='--', lw=0.5)
        plt.xlabel('time (h)')
        plt.ylabel('MCS surviving fraction')

        plt.savefig(self.outputs['fig'])

