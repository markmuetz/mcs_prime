import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from mcs_prime import PATHS, mcs_mask_plotter
from remake import Remake, TaskRule
from remake.util import format_path as fmtp

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
    _plot_hist(ds, ax, np.nansum(ds[f'{reg}_{var}_cloud_core'].values, axis=0), 'b-', 'cloud core')
    _plot_hist(ds, ax, np.nansum(ds[f'{reg}_{var}_cloud_shield'].values, axis=0), 'b--', 'cloud shield')
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
    ax.plot(x, probs[2][s], 'b-', label='cloud core')
    ax.plot(x, probs[3][s], 'b--', label='cloud shield')
    ax.plot(x, probs[4][s], 'k-', label='env')
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


class PlotCombineConditionalERA5Hist(TaskRule):
    @staticmethod
    def rule_inputs(year, core_method):
        inputs = {
            f'hist_{year}_{month}': fmtp(cu.FMT_PATH_COND_HIST_HOURLY, year=year, month=month, core_method=core_method)
            for month in cu.MONTHS
        }
        return inputs

    rule_outputs = {
        f'fig_{var}': (PATHS['figdir'] / 'conditional_era5_histograms' /
                       f'yearly_hist_{var}_{{year}}_{{core_method}}.png')
        for var in cu.EXTENDED_ERA5VARS
    }

    depends_on = [get_labels, plot_hist, plot_hist_probs, plot_hists_for_var]

    var_matrix = {'year': cu.YEARS, 'core_method': ['tb', 'precip']}

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


class PlotCombineVarConditionalERA5Hist(TaskRule):
    @staticmethod
    def rule_inputs(year, e5vars):
        inputs = {
            f'hist_{year}_{month}': fmtp(cu.FMT_PATH_COND_HIST_HOURLY, year=year, month=month, core_method='tb')
            for month in cu.MONTHS
        }
        return inputs

    rule_outputs = {
        'fig': (PATHS['figdir'] / 'conditional_era5_histograms' /
                'combined_yearly_hist_{e5vars}_{year}_tb.png')
    }

    depends_on = [get_labels, plot_hist, plot_hist_probs, plot_combined_hists_for_var]

    var_matrix = {
        'year': cu.YEARS,
        'e5vars': [
            'cape-tcwv-vertically_integrated_moisture_flux_div',
            'shear_0-shear_1-shear_2',
            'shear_0-shear_1-shear_3',
            'RHlow-RHmid-theta_e_mid',
        ],
    }

    def rule_run(self):

        fig, axes = plt.subplots(2, 3, sharex='col')
        fig.set_size_inches((15, 8))
        e5vars = self.e5vars.split('-')

        with xr.open_mfdataset(self.inputs.values()) as ds:
            ds.load()
            for (ax0, ax1), var in zip(axes.T, e5vars):
                print(var)
                plot_combined_hists_for_var(ax0, ax1, ds, var)

                if var == 'vertically_integrated_moisture_flux_div':
                    ax1.set_xlabel('MFC (10$^{-4}$ kg m$^{-2}$ s$^{-1}$)')
                else:
                    ax1.set_xlabel(get_labels(var))

        axes[0, 0].legend()
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


class PlotCombineConvectionConditionalERA5Hist(TaskRule):
    @staticmethod
    def rule_inputs(year, core_method):
        inputs = {
            f'hist_{year}_{month}': fmtp(cu.FMT_PATH_COND_HIST_HOURLY, year=year, month=month, core_method=core_method)
            for month in cu.MONTHS
        }
        return inputs

    rule_outputs = {
        f'fig_{var}': (PATHS['figdir'] / 'conditional_era5_histograms' /
                       f'convection_yearly_hist_{var}_{{year}}_{{core_method}}.png')
        for var in cu.EXTENDED_ERA5VARS
    }

    depends_on = [plot_convection_hourly_hists]

    var_matrix = {'year': cu.YEARS, 'core_method': ['tb', 'precip']}

    def rule_run(self):
        with xr.open_mfdataset(self.inputs.values()) as ds:
            ds.load()
            for var in cu.EXTENDED_ERA5VARS:
                print(var)
                plot_convection_hourly_hists(ds, var)
                plt.savefig(self.outputs[f'fig_{var}'])


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
        aspect = 1/300
    elif var.startswith('shear'):
        aspect = 100
    else:
        aspect = 100
    plt.imshow(h.T, origin='lower', extent=(ds[f'{var}_hist_mids'].values[0], ds[f'{var}_hist_mids'].values[-1], 0, 1), aspect=aspect)
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
    @staticmethod
    def rule_inputs(year):
        inputs = {
            f'hist_{year}': fmtp(cu.FMT_PATH_COMBINED_COND_HIST_GRIDPOINT, year=year),
            'ERA5_land_sea_mask': cu.PATH_ERA5_LAND_SEA_MASK,
        }
        return inputs

    rule_outputs = {
        f'fig_{output}_{var}': (PATHS['figdir'] / 'conditional_era5_histograms' /
                                f'gridpoint_{output}_yearly_hist_{var}_{{year}}.png')
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
    return np.sqrt(np.nanmean((a[None, None, :] - b)**2, axis=2))


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
            )
        )
        da_integral_diff = xr.DataArray(
            integral_diff(d, dc, 1),
            name=f'{var}_integral_diff',
            dims=['latitude', 'longitude'],
            coords=dict(
                latitude=da1.latitude,
                longitude=da1.longitude,
            )
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
    im1 = axes[1].contourf(ds.longitude, ds.latitude, ds[f'{var}_integral_diff'],
                           levels=levels, cmap='bwr')
    plt.colorbar(im1, ax=axes[1])

    for ax in axes:
        ax.coastlines()


class PlotGridpointGlobal(TaskRule):
    @staticmethod
    def rule_inputs(year):
        inputs = {
            f'hist_{year}': fmtp(cu.FMT_PATH_COMBINED_COND_HIST_GRIDPOINT, year=year),
            'ERA5_land_sea_mask': cu.PATH_ERA5_LAND_SEA_MASK,
        }
        return inputs

    rule_outputs = {
        f'fig_{var}': (PATHS['figdir'] / 'conditional_era5_histograms' /
                       f'gridpoint_global_yearly_hist_{var}_{{year}}.png')
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
        3, 1,
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
        diff = (da_var.mean(dim='time') - da_mean.mean(dim='time'))
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
    im1 = ax1.imshow(np.ma.masked_array(da_mean.mean(dim='time').values, mask=mask_sum < 10), vmin=levels1[0], vmax=levels1[-1], extent=extent)
    im2 = ax2.imshow(np.ma.masked_array(da_var.mean(dim='time').values, mask=mask_sum < 10), vmin=levels1[0], vmax=levels1[-1], extent=extent)
    im3 = ax3.imshow(np.ma.masked_array(diff.values, mask=mask_sum < 10), vmin=levels2[0], vmax=levels2[-1], cmap='bwr', extent=extent)

    plt.colorbar(im1, ax=[ax1, ax2], extend='max')
    plt.colorbar(im3, ax=ax3, extend='both')
    for ax in [ax1, ax2, ax3]:
        ax.coastlines()
        ax.set_ylim((-60, 60))


class PlotMcsLocalEnv(TaskRule):
    @staticmethod
    def rule_inputs(year, mode):
        inputs = {f'mcs_local_env_{year}_{month}': fmtp(cu.FMT_PATH_COMBINE_MCS_LOCAL_ENV,
                                                        year=year,
                                                        month=month,
                                                        mode=mode)
                  for month in cu.MONTHS}
        return inputs

    rule_outputs = {
        f'fig_{radius}_{plot_type}_{var}': (PATHS['figdir'] / 'mcs_local_envs' /
                                            f'mcs_local_env_r{radius}km_{plot_type}_{var}_{{mode}}_{{year}}.png')
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
    im = ax.imshow(np.ma.masked_array(diff.values, mask=mask_sum < 10), vmin=levels2[0], vmax=levels2[-1], cmap='bwr', extent=extent)

    if var == 'vertically_integrated_moisture_flux_div':
        label = 'MFC (10$^{-4}$ kg m$^{-2}$ s$^{-1}$)'
    else:
        label = get_labels(var)
    plt.colorbar(im, ax=ax, extend='both', label=label)
    ax.coastlines()
    ax.set_ylim((-60, 60))


class PlotCombinedMcsLocalEnv(TaskRule):
    @staticmethod
    def rule_inputs(year, e5vars):
        inputs = {f'mcs_local_env_{year}_{month}': fmtp(cu.FMT_PATH_COMBINE_MCS_LOCAL_ENV,
                                                        year=year,
                                                        month=month,
                                                        mode='init')
                  for month in cu.MONTHS}
        return inputs

    rule_outputs = {
        f'fig_{radius}': (PATHS['figdir'] / 'mcs_local_envs' /
                          f'combined_mcs_local_env_r{radius}km_{{e5vars}}_init_{{year}}.png')
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


def plot_precursor_mean_val(ds, var, radii, ax=None, N=73):
    ds[f'mean_{var}'].load()

    # fig, (ax1, ax2) = plt.subplots(2, 1)
    if ax is None:
        ax = plt.gca()
    for r in radii:
        print(f' plot {r}')
        data = ds[f'mean_{var}'].sel(radius=r).isel(times=slice(0, N)).mean(dim='tracks')
        if var == 'vertically_integrated_moisture_flux_div':
            ax.plot(range(-24, -24 + N), -data * 1e4, label=f'{r} km')
            ylabel = 'MFC (10$^{-4}$ kg m$^{-2}$ s$^{-1}$)'
            ax.set_ylabel(ylabel)
        else:
            ax.plot(range(-24, -24 + N), data, label=f'{r} km')
            ax.set_ylabel(get_labels(var))

    ax.axvline(x=0)

    # THIS IS SSSSLLLLOOOOOWWWW!!!!
    # print(f' load hist')
    # hist_data = ds[f'hist_{var}'].sel(radius=100).isel(times=slice(0, N)).load()
    # print(f' plot hist')
    # ax2.plot(range(-24, -24 + N), np.isnan(hist_data.values).sum(axis=(0, 1)))


class PlotMcsLocalEnvPrecursorMeanValue(TaskRule):
    @staticmethod
    def rule_inputs(year, N):
        inputs = {f'mcs_local_env_{year}_{month}': fmtp(cu.FMT_PATH_LIFECYCLE_MCS_LOCAL_ENV,
                                                        year=year,
                                                        month=month)
                  for month in cu.MONTHS}
                  # for month in [1]}
        return inputs

    rule_outputs = {
        f'fig_{var}': (PATHS['figdir'] / 'mcs_local_envs' /
                       f'mcs_local_env_precursor_mean_{var}_{{year}}_{{N}}.png')
        for var in cu.EXTENDED_ERA5VARS
    }

    depends_on = [
        plot_precursor_mean_val,
    ]

    var_matrix = {
        'year': cu.YEARS,
        'N': [49, 73, 97, 424],
    }

    def rule_run(self):
        with xr.open_mfdataset(self.inputs.values()) as ds:
            for var in cu.EXTENDED_ERA5VARS:
                print(var)
                fig, ax = plt.subplots(1, 1)
                fig.set_size_inches((20, 8))
                plot_precursor_mean_val(ds, var, cu.RADII[1:], ax=ax, N=self.N)
                ax.legend()
                ax.set_xlabel('time from MCS initiation (hr)')
                plt.savefig(self.outputs[f'fig_{var}'])


class PlotCombinedMcsLocalEnvPrecursorMeanValue(TaskRule):
    @staticmethod
    def rule_inputs(year, e5vars):
        inputs = {f'mcs_local_env_{year}_{month}': fmtp(cu.FMT_PATH_LIFECYCLE_MCS_LOCAL_ENV,
                                                        year=year,
                                                        month=month)
                  for month in cu.MONTHS}
        return inputs

    rule_outputs = {
        'fig_{e5vars}': (PATHS['figdir'] / 'mcs_local_envs' /
                         'mcs_local_env_precursor_mean_{e5vars}_{year}.png')
    }

    depends_on = [
        plot_precursor_mean_val,
    ]

    var_matrix = {
        'year': cu.YEARS,
        'e5vars': ['cape-tcwv-shear_0-vertically_integrated_moisture_flux_div'],
    }

    def rule_run(self):
        fig, axes = plt.subplots(2, 2, sharex=True)
        fig.set_size_inches((10, 8))
        e5vars = self.e5vars.split('-')

        with xr.open_mfdataset(self.inputs.values()) as ds:
            for ax, var in zip(axes.flatten(), e5vars):
                print(var)
                plot_precursor_mean_val(ds, var, cu.RADII[1:], ax=ax, N=73)
                ax.set_xlim((-24, 48))

        axes[0, 0].legend()
        for ax in axes[1]:
            ax.set_xlabel('time from MCS initiation (hr)')
        plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95, wspace=0.22, hspace=0.1)
        plt.savefig(self.outputs[f'fig_{self.e5vars}'])

