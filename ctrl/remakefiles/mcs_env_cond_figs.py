from itertools import product
import string

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
# import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr

from remake import Remake, TaskRule
from remake.util import format_path as fmtp

from mcs_prime import PATHS, McsTracks, mcs_mask_plotter
import mcs_prime.mcs_prime_config_util as cu


slurm_config = {'account': 'short4hr', 'queue': 'short-serial-4hr', 'mem': 64000}
plotting = Remake(config=dict(slurm=slurm_config, content_checks=False))

SUBFIG_SQ_SIZE = 9  # cm

# YEARS = cu.YEARS
YEARS = [2020]


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


settings_list = [
    {
        'name': 'tcwv1',
        'time': pd.Timestamp(2020, 1, 1, 6, 30),
        'plot_kwargs': dict(var='tcwv', extent=(-25, -5, -2.5, 7.5), grid_x=[-20, -15, -10], grid_y=[-2, 0, 2, 4, 6], cbar_kwargs=dict(orientation='horizontal')),
    },
]
settings_dict = dict([(s['name'], s) for s in settings_list])


class PlotMcsMasks(TaskRule):
    @staticmethod
    def rule_inputs(settings_name):
        settings = settings_dict[settings_name]
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
        settings = settings_dict[settings_name]
        filename = f'MCS_masks_and_centroids_{settings["time"]}_{settings_name}.pdf'.replace(' ', '_').replace(':', '')
        return {f'fig_{settings_name}': PATHS['figdir'] / 'mcs_env_cond_figs' / filename}

    depends_on = [
        mcs_mask_plotter.McsMaskPlotterData,
        mcs_mask_plotter.McsMaskPlotter,
    ]

    var_matrix = {
        'settings_name': list(settings_dict.keys()),
    }

    def rule_run(self):
        settings = settings_dict[self.settings_name]
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


class PlotCombinedMcsLocalEnvPrecursorMeanValueFiltered(TaskRule):
    @staticmethod
    def rule_inputs(years, radius):
        inputs = {}
        for year in years:
            inputs.update({
                f'mcs_local_env_{year}_{month}': fmtp(cu.FMT_PATH_LIFECYCLE_MCS_LOCAL_ENV, year=year, month=month)
                for month in cu.MONTHS
            })
            inputs[f'tracks_{year}'] = cu.fmt_mcs_stats_path(year)
        return inputs

    @staticmethod
    def rule_outputs(years, radius):
        ystr = cu.fmt_ystr(years)
        return {'fig': (PATHS['figdir'] / 'mcs_env_cond_figs' / f'combined_filtered_mcs_local_env_precursor_mean_{ystr}.radius-{radius}.pdf')}

    depends_on = [
        plot_grouped_precursor_mean_val,
    ]

    var_matrix = {
        'years': [YEARS],
        'radius': [100, 200, 500, 1000],
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
        print('Open datasets')
        ds_full = xr.open_mfdataset([p for k, p in self.inputs.items() if k.startswith('mcs_local_env_')],
                                     combine='nested', concat_dim='tracks')
        ds_full['tracks'] = np.arange(0, ds_full.dims['tracks'], 1, dtype=int)
        print(ds_full)

        tracks_paths = [p for k, p in self.inputs.items() if k.startswith('tracks_')]
        tracks = McsTracks.mfopen(tracks_paths, None)
        print(tracks.dstracks)

        # Build filters. Each is determined by the MCS tracks dataset and applied
        # to the data in ds_full.
        print('Build filters')
        print('  natural')
        # Only include "natural" MCSs - those that do not form by splitting from an existing one.
        # Note, the number of natural MCSs is taken as the baseline for working out
        # how many MCSs are filtered into a particular group (natural.sum()).
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

        # Build combinartorial product of filters.
        filter_key_combinations = list(product(
            filter_vals['natural'].keys(),
            filter_vals['equator-tropics-extratropics'].keys(),
            filter_vals['land-sea'].keys(),
        ))

        n_hours = 73
        # Set up the figure (using seaborn theming).
        sns.set_theme(palette='deep')
        sns.set_style('ticks')
        sns.set_context('paper')
        fig, axes = plt.subplots(4, 3, layout='constrained', sharex=True)
        fig.set_size_inches(cm_to_inch(SUBFIG_SQ_SIZE * 3, 0.6 * SUBFIG_SQ_SIZE * 4))

        # Colours determined by eq-trop-ET.
        colours = dict(zip(
            filter_vals['equator-tropics-extratropics'].keys(),
            plt.rcParams['axes.prop_cycle'].by_key()['color']
        ))
        # Linestyles determined by land-sea.
        linestyles = dict(zip(
            filter_vals['land-sea'].keys(),
            ['-', '--']
        ))

        for i, (ax, var) in enumerate(zip(axes.flatten(), e5vars)):
            print(var)
            data_array = ds_full[f'mean_{var}'].sel(radius=self.radius).isel(times=slice(0, n_hours)).load()

            if var == 'vertically_integrated_moisture_flux_div':
                data_array = -data_array * 1e4
                ylabel = 'MFC (10$^{-4}$ kg m$^{-2}$ s$^{-1}$)'
            else:
                ylabel = get_labels(var)

            grouped_data_dict = {'xvals': range(-24, -24 + n_hours)}
            # Apply filters.
            for filter_keys in filter_key_combinations:
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

            plot_grouped_precursor_mean_val(ax, grouped_data_dict)

            # Plot the full data.
            plot_kwargs = {
                'color': 'k',
                'linestyle': ':',
            }
            grouped_data_dict = {'xvals': range(-24, -24 + n_hours)}
            total = natural.sum()
            grouped_data_dict[f'all ({total} tracks)'] = {
                'data_array': data_array.isel(tracks=natural),
                'ylabel': ylabel,
                'plot_kwargs': plot_kwargs,
            }

            plot_grouped_precursor_mean_val(ax, grouped_data_dict)

            # Apply a nice title for each ax.
            c = string.ascii_lowercase[i]
            # varname = ylabel[:ylabel.find('(')].strip()
            # units = ylabel[ylabel.find('('):].strip()
            # ax.set_title(f'{c}) {varname}', loc='left')
            # ax.set_ylabel(units)
            # Saves space to put units in title, also avoids awkard problem of
            # positioning ylabel due to different widths of numbers for e.g. 1.5 vs 335
            ax.set_title(f'{c}) {ylabel}', loc='left')
            ax.axvline(x=0, color='k')

        # Set some figure-wide text.
        axes[0, -1].legend(loc='lower left', bbox_to_anchor=(0.5, -0.2))
        axes[-1, 1].set_xlabel('time from MCS initiation (hr)')

        plt.savefig(self.outputs[f'fig'])

