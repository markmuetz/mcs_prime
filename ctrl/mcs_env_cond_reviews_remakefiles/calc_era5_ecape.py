import sys
from pathlib import Path

import cartopy.crs as ccrs
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import metpy.calc as mpcalc
from metpy.plots import Hodograph, SkewT
from metpy.units import units
import numpy as np
import pandas as pd
from scipy.stats import linregress
import xarray as xr


from remake import Remake, TaskRule

import mcs_prime.mcs_prime_config_util as cu
from mcs_prime.era5_calc import ERA5Calc

sys.path.insert(0, '/home/users/mmuetz/projects/ecape_calc')
# My version of peters2023 code.
from ecape_calc import compute_CAPE_CIN, compute_NCAPE, compute_VSR, compute_ETILDE, lift_parcel_adiabatic


from consts import Rv, Rd
from params import T1, T2

# Capella github with minor mods.
from ecape.calc import calc_ecape



OUTDIR = cu.PATHS['outdir']
FIGDIR = cu.PATHS['figdir']

slurm_config = {'account': 'short4hr', 'queue': 'short-serial-4hr', 'mem': 64000}
# slurm_config = {'queue': 'short-serial', 'mem': 64000}
rmk = Remake(config=dict(slurm=slurm_config, content_checks=False))

# Pick first entire day for 1st of month for 2020.
day_range = [
    pd.date_range(f'2020-{m:02d}-01 00:00', f'2020-{m:02d}-01 23:00', freq='H')
    for m in range(1, 13)
]

dates = pd.DatetimeIndex(pd.concat([pd.Series(dti) for dti in day_range]))

Rd = 287.06
g = 9.80665
# As used by IFS: https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation#ERA5:datadocumentation-SpatialreferencesystemsandEarthmodel
Re = 6371229 # m

class Era5ComputeAlt:
    """Compute geopotential (z) from netcdf data, then compute altitude (alt)

    Code taken from existing code for doing the same for GRIB data"""
    def __init__(self, ds):
        self.ds = ds
        self.e5calc = ERA5Calc('/gws/nopw/j04/mcs_prime/mmuetz/data/ERA5/ERA5_L137_model_levels_table.csv')


    def run(self):
        T = self.ds.t.values
        q = self.ds.q.values
        p = self.e5calc.calc_pressure(self.ds.lnsp.values)
        self.p = p
        zsfc = self.ds.z.values

        # Get levels in ascending order of height (starts at 137)
        levels = self.ds.level.values[::-1]
        # print(levels)

        # 0.609133 = Rv/Rd - 1.
        # TODO: Why extra RD
        Tv = T * (1. + 0.609133 * q) * Rd
        z_h = zsfc

        z = np.zeros_like(p)
        for lev in levels:
            lev_idx = lev - 1
            # print(lev, lev_idx)
            z_h, z_f = self.compute_z_level(lev_idx, p, Tv, z_h)
            z[lev_idx] = z_f

        h = z / g
        alt = Re * h / (Re - h)

        self.z = z
        self.h = h
        self.alt = alt
        return z, h, alt

    def compute_z_level(self, lev_idx, p, Tv, z_h):
        '''Compute z at half & full level for the given level, based on T/q/sp'''
        # compute the pressures (on half-levels)
        # ph_lev, ph_levplusone = get_ph_levs(values, lev)
        ph_lev, ph_levplusone = p[lev_idx - 1], p[lev_idx]

        if lev_idx == 0:
            dlog_p = np.log(ph_levplusone / 0.1)
            alpha = np.log(2)
        else:
            dlog_p = np.log(ph_levplusone / ph_lev)
            alpha = 1. - ((ph_lev / (ph_levplusone - ph_lev)) * dlog_p)

        # z_f is the geopotential of this full level
        # integrate from previous (lower) half-level z_h to the
        # full level
        z_f = z_h + (Tv[lev_idx] * alpha)

        # z_h is the geopotential of 'half-levels'
        # integrate z_h to next half level
        z_h = z_h + (Tv[lev_idx] * dlog_p)

        return z_h, z_f


def peters23_compute_ECAPE_etc(ds, comp_alt, lon_idx, lat_idx):
    # ERA5 indexes levels from highest to lowest - switch with -1..
    levlatlon_idx = (slice(None, None, -1), lat_idx, lon_idx)
    # Pull out profiles for given lat/lon.
    T0 = ds.t.values[levlatlon_idx]
    q0 = ds.q.values[levlatlon_idx]
    u0 = ds.u.values[levlatlon_idx]
    v0 = ds.v.values[levlatlon_idx]
    p0 = comp_alt.p[levlatlon_idx]
    z0_msl = comp_alt.alt[levlatlon_idx]
    z0 = z0_msl - z0_msl[0]  # Convert into height above surface.
    Lmix = 120  # m Recommended value.

    CAPE, CIN, LFC, EL = compute_CAPE_CIN(T0, p0, q0, 0, 0, 0, z0, T1, T2)

    if not (np.isnan(LFC) or np.isnan(EL)) and CAPE > 100:
        NCAPE, MSE0_star, MSE0bar = compute_NCAPE(T0, p0, q0, z0, T1, T2, LFC, EL)

        # Get the 0-1 km mean storm-relative wind, estimated using bunkers2000 method for right-mover storm motion
        V_SR, C_x, C_y = compute_VSR(z0, u0, v0)

        # Get e_tilde, which is the ratio of ecape to cape.  also, varepsilon is the fractional entrainment rate, and radius is the theoretical upraft radius
        Etilde, varepsilon, radius = compute_ETILDE(CAPE, NCAPE, V_SR, EL, Lmix)
        ECAPE = CAPE * Etilde

        return dict(
            CAPE=CAPE,
            CIN=CIN,
            LFC=LFC,
            EL=EL,
            NCAPE=NCAPE,
            V_SR=V_SR,
            Etilde=Etilde,
            varepsilon=varepsilon,
            radius=radius,
            ECAPE=Etilde * CAPE,
        )
    else:
        return dict(
            CAPE=None,
            CIN=None,
            LFC=None,
            EL=None,
            NCAPE=None,
            V_SR=None,
            Etilde=None,
            varepsilon=None,
            radius=None,
            ECAPE=None,
        )


def capella_compute_ECAPE_etc(ds, comp_alt, lon_idx, lat_idx):
    # ERA5 indexes levels from highest to lowest.
    levlatlon_idx = (slice(None, None, -1), lat_idx, lon_idx)
    T0 = ds.t.values[levlatlon_idx]
    q0 = ds.q.values[levlatlon_idx]
    u0 = ds.u.values[levlatlon_idx]
    v0 = ds.v.values[levlatlon_idx]
    p0 = comp_alt.p[levlatlon_idx]
    z0_msl = comp_alt.alt[levlatlon_idx]
    z0 = z0_msl - z0_msl[0]  # Convert into height above surface.

    z0 = z0 * units('m')
    p0 = p0 * units('Pa')
    T0 = T0 * units('K')
    q0 = q0 * units('kg/kg')
    u0 = u0 * units('m/s')
    v0 = v0 * units('m/s')

    # TODO:
    # Appears to be a problem in the ODE solver for the LFC.
    # lsoda--  at t (=r1), too much accuracy requested
    #               for precision of machine..  see tolsf (=r2)
    #             in above,  r1 =  0.3448291218653D+02   r2 =                  NaN
    CAPE, CIN, LFC, EL, NCAPE, V_SR, ECAPE = calc_ecape(
        z0,
        p0,
        T0,
        q0,
        u0,
        v0,
        'most_unstable',
        # undiluted_cape=cape,
    )

    return dict(
        CAPE=CAPE.magnitude,
        CIN=CIN.magnitude,
        LFC=LFC.magnitude,
        EL=EL.magnitude,
        NCAPE=NCAPE.magnitude,
        V_SR=V_SR.magnitude,
        Etilde=ECAPE.magnitude / CAPE.magnitude,
        varepsilon=0,
        radius=0,
        ECAPE=ECAPE.magnitude,
    )


ECAPE_KEYS = [
    'CAPE',
    'CIN',
    'LFC',
    'EL',
    'NCAPE',
    'V_SR',
    'Etilde',
    'varepsilon',
    'radius',
    'ECAPE',
]

def plot_profile(ds, comp_alt, lon_idx, lat_idx, compute_ECAPE_etc):
    # ERA5 indexes levels from highest to lowest.
    levlatlon_idx = (slice(None, None, -1), lat_idx, lon_idx)
    T0 = ds.t.values[levlatlon_idx]
    q0 = ds.q.values[levlatlon_idx]
    u0 = ds.u.values[levlatlon_idx]
    v0 = ds.v.values[levlatlon_idx]
    p0 = comp_alt.p[levlatlon_idx]
    z0_msl = comp_alt.alt[levlatlon_idx]
    z0 = z0_msl - z0_msl[0]  # Convert into height above surface.

    ecape_dict = compute_ECAPE_etc(ds, comp_alt, lon_idx, lat_idx)
    Etilde = ecape_dict['Etilde']
    CAPE = ecape_dict['CAPE']
    varepsilon = ecape_dict['varepsilon']
    radius = ecape_dict['radius']

    # Compute dewpoint and temperature, assign units in accordance with metpy vernacular
    T = T0 * units.degK
    p = p0 * units.Pa
    e = mpcalc.vapor_pressure(p, q0 * units('kg/kg'))
    Td = mpcalc.dewpoint(e)
    for iz in np.arange(0, Td.shape[0], 1):
        T[iz] = max(T[iz], Td[iz].to('K'))

    # Compute lifted parcel properties for an undiluted parcel
    fracent = 0
    T_lif, Qv_lif, Qt_lif, B_lif = lift_parcel_adiabatic(
        T.magnitude, p0, q0, 0, fracent, 0, z0, T1, T2
    )
    T_rho = T_lif * (1 + (Rv / Rd) * Qv_lif - Qt_lif)
    T_rho = T_rho * units('K')

    # Plot the skew-t skeleton
    # params = {
    #     "ytick.color": "black",
    #     "xtick.color": "black",
    #     "axes.labelcolor": "black",
    #     "axes.edgecolor": "black",
    #     "font.size": 12,
    #     #"text.usetex": True,
    # }
    # plt.rcParams.update(params)

    gs = gridspec.GridSpec(3, 3)
    fig = plt.figure(figsize=(9, 9))
    # axs = plt.subplots(1, 2, constrained_layout=True)

    skew = SkewT(fig, rotation=45, subplot=gs[:, :2])

    # Plot the data using normal plotting functions, in this case using
    # log scaling in Y, as dictated by the typical meteorological plot.
    skew.plot(p, T, 'r')
    skew.plot(p, Td, 'g')
    p_ = p[p >= 9000 * units.Pa]
    u0_ = u0[p >= 9000 * units.Pa]
    v0_ = v0[p >= 9000 * units.Pa]
    skew.plot_barbs(p_[0::5], u0_[0::5], v0_[0::5], x_clip_radius=0.1, y_clip_radius=0.08)
    skew.ax.set_ylim(1020, 100)
    skew.ax.set_xlim(-20, 40)

    lat = ds.latitude.values[lat_idx]
    lon = ds.longitude.values[lon_idx]
    skew.ax.text(
        -15,
        900,
        f'lat,lon: {lat:.1f}, {lon:.1f}\n'
        f'CAPE: {CAPE:.0f} J kg$^{{-1}}$\n'
        f'ECAPE: {Etilde * CAPE:.0f} J kg$^{{-1}}$\n'
        f'$\widetilde{{\mathrm{{E}}}}_A$: {Etilde * 100:.0f}%\n'
        f'R: {radius:.0f} m',
        ha="center",
        va="center",
        size=7,
        bbox=dict(boxstyle="square,pad=0.3", fc="lightblue", ec="steelblue", lw=2),
    )

    # Set some better labels than the default
    skew.ax.set_xlabel(f'Temperature (C)')
    skew.ax.set_ylabel(f'Pressure (hPa)')

    lcl_pressure, lcl_temperature = mpcalc.lcl(p[0], T[0], Td[0])
    skew.plot(lcl_pressure, lcl_temperature, 'ko', markerfacecolor='black')

    # Calculate full parcel profile and add to plot as black line
    # prof = mpcalc.parcel_profile(p, T[0], Td[0]).to('degC')
    prof = T_rho.to('degC')
    skew.plot(p, prof, 'k', linewidth=2)
    T_rho0 = T.magnitude * (1 + (Rv / Rd - 1) * q0)
    T_rho0 = T_rho0 * units('K')

    skew.plot(p, T_rho0, 'r', linewidth=0.5)
    # skew.plot(p,T_rho.to('degC'),'k',linewidth=2)

    # Shade areas of CAPE and CIN
    try:
        skew.shade_cin(p, T_rho0.to('degC'), prof, Td)
    except:
        print('NO CIN')
    try:
        skew.shade_cape(p, T_rho0.to('degC'), prof)
    except:
        print('NO CAPE')

    fracent = varepsilon
    # prate=3e-5
    T_lif, Qv_lif, Qt_lif, B_lif = lift_parcel_adiabatic(
        T.magnitude, p0, q0, 0, fracent, 0, z0, T1, T2
    )
    ECAPE, ECIN, ELFC, EEL = compute_CAPE_CIN(T0, p0, q0, 0, fracent, 0, z0, T1, T2)
    # Compute density temeprature for the lifted parcel and assign units
    T_rho = T_lif * (1 + (Rv / Rd) * Qv_lif - Qt_lif)
    T_rho = T_rho * units('K')

    skew.plot(p, T_rho, 'b--', linewidth=1)
    prof = T_rho.to('degC')
    try:
        skew.shade_cape(p, T_rho0.to('degC'), prof, facecolor=(0.5, 0.5, 0.5, 0.75))
    except:
        print('NO CAPE')

    # An example of a slanted line at constant T -- in this case the 0
    # isotherm
    skew.ax.axvline(0, color='c', linestyle='--', linewidth=2)

    # Add the relevant special lines
    skew.plot_dry_adiabats(linewidths=0.5)
    skew.plot_moist_adiabats(linewidths=0.5)
    skew.plot_mixing_lines(linewidths=0.5)

    ax_hod = inset_axes(skew.ax, '40%', '40%', loc=1)

    fplt = np.where(z0 <= 6000)[0]
    max_wind = np.sqrt(u0[fplt]**2 + v0[fplt]**2).max()
    limit = np.ceil(max_wind / 10) * 10
    h = Hodograph(ax_hod, component_range=limit + 5)
    h.add_grid(increment=limit / 2, linewidth=0.5)

    cmap = plt.get_cmap('autumn_r', len(fplt))
    h.plot_colormapped(
        u0[fplt],
        v0[fplt],
        np.floor(z0[fplt] / 1000),
        linewidth=2,
        cmap=plt.get_cmap('gist_rainbow_r', 6),
    )  # Plot a line colored by wind speed

    plt.xlim(-limit - 5, limit + 5)
    plt.ylim(-limit - 5, limit + 5)
    plt.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        left=False,
        top=False,  # ticks along the top edge are off
        labelleft=False,
        labelbottom=False,
    )  # labels along the bottom edge are off
    for mag in [limit / 2, limit]:
        plt.text(-5, -mag - 1, f'{mag:.0f} m s$^{{-1}}$', fontsize=8)
    # plt.text(-5, -41, 'm s$^{-1}$', fontsize=8)
    # TODO: what is this meant to plot?
    # storm relative motion I suspect.
    #plt.plot(C_x, C_y, 'ko', markersize=2.5)
    return skew, h


class Era5ECAPE(TaskRule):
    @staticmethod
    def rule_inputs(date, cov, method):
        basepath = Path(f'/badc/ecmwf-era5/data/oper/an_ml/' + date.strftime('%Y/%m/%d'))
        tstamp = date.strftime('%Y%m%d%H%M')
        inputs = {
            v: basepath / f'ecmwf-era5_oper_an_ml_{tstamp}.{v}.nc'
            for v in ['lnsp', 'z', 't', 'q', 'u', 'v']
            }
        return inputs

    @staticmethod
    def rule_outputs(date, cov, method):
        outdir = OUTDIR / 'mcs_env_cond_reviews/ecape' / date.strftime('%Y/%m/%d')
        tstamp = date.strftime('%Y%m%d%H%M')
        outputs = {'output': outdir / cov / method / f'ecmwf-era5_oper_an_ml_{tstamp}.{method}_ecape.{cov}.nc'}
        for k in ECAPE_KEYS:
            for i in range(11):
                outputs[f'fig_{k}_d{i}'] = (
                    outdir / cov / method / 'figs' /
                    f'skewT_{tstamp}.{method}_ecape.{k}.decile{i:02d}.{cov}.png'
                )
        return outputs

    var_matrix = {
        'date': dates,
        'cov': ['quick', 'full'],
        # 'cov': ['quick'],
        'method': ['peters2023'],
        # 'method': ['capella'],
    }

    depends_on = [peters23_compute_ECAPE_etc, capella_compute_ECAPE_etc, plot_profile]

    def rule_run(self):
        ds = xr.open_mfdataset(self.inputs.values()).isel(time=0).sel(latitude=slice(60, -60)).load()

        comp_alt = Era5ComputeAlt(ds)
        _ = comp_alt.run()

        if self.cov == 'quick':
            step = 10
        else:
            step = 1

        lons = list(range(0, 1440, step))
        lats = list(range(0, 481, step))
        output = np.full((10, len(lats), len(lons)), np.nan)

        if self.method == 'peters2023':
            compute_ECAPE_etc = peters23_compute_ECAPE_etc
        elif self.method == 'capella':
            compute_ECAPE_etc = capella_compute_ECAPE_etc
        for i, lon_idx in enumerate(lons):
            print(f'{i / len(lons) * 100:.1f}%')
            for j, lat_idx in enumerate(lats):
                try:
                    ecape_dict = compute_ECAPE_etc(ds, comp_alt, lon_idx, lat_idx)
                    for k, key in enumerate(ecape_dict.keys()):
                        output[k, j, i] = ecape_dict[key]
                except Exception as e:
                    # print(e)
                    pass

        lon = ds.longitude.values[lons]
        lat = ds.latitude.values[lats]
        keys = list(ecape_dict.keys())

        # Save these as attrs.
        import params
        import consts
        attrs = {}
        attrs.update({
            f'ECAPE param {pname}': getattr(params, pname)
            for pname in [pname for pname in dir(params) if not pname.startswith('__')]
        })
        attrs.update({
            f'ECAPE const {pname}': getattr(consts, pname)
            for pname in [pname for pname in dir(consts) if not pname.startswith('__')]
        })

        dsout = xr.Dataset(
            coords=dict(
                time=[pd.Timestamp(ds.time.values.item())],
                latitude=lat,
                longitude=lon,
            ),
            data_vars={
                keys[i]: (('latitude', 'longitude'), output[i, :, :])
                for i in range(10)
            },
            attrs=attrs,
        )
        cu.to_netcdf_tmp_then_copy(dsout, self.outputs['output'])

        for var in keys:
            print(f'Skew-T for {var}')
            data = dsout[var].values
            ptiles = np.nanpercentile(data, np.arange(0, 101, 10))
            for i, ptile_val in enumerate(ptiles):
                print(f'  decile{i:02d}')
                idx = np.nanargmin(np.abs(data - ptile_val))
                data_val = data.flat[idx]
                lat_idx, lon_idx = np.where(data == data_val)
                # print(ptile_val, idx, data_val)
                lat_idx *= step
                lon_idx *= step
                try:
                    skewT, hodo = plot_profile(ds, comp_alt, lon_idx[0], lat_idx[0], compute_ECAPE_etc)
                    skewT.ax.set_title(f'{var} percentile d{i * 10}')
                    plt.savefig(self.outputs[f'fig_{var}_d{i}'])
                    plt.close('all')
                except Exception as e:
                    print(lon_idx, lat_idx, e)
                    self.outputs[f'fig_{var}_d{i}'].write_text(str((lon_idx, lat_idx, e)))


ptiles = np.array([1, 5, 10, 25, 50, 75, 90, 95, 99])
class PlotEra5ECAPE(TaskRule):
    @staticmethod
    def rule_inputs(cov, method):
        inputs = {}
        for date in dates:
            outdir = OUTDIR / 'mcs_env_cond_reviews/ecape' / date.strftime('%Y/%m/%d')
            tstamp = date.strftime('%Y%m%d%H%M')
            inputs[f'ecape_{date}'] = (
                outdir / cov / method / f'ecmwf-era5_oper_an_ml_{tstamp}.{method}_ecape.{cov}.nc'
            )
        return inputs

    @staticmethod
    def rule_outputs(cov, method):
        figdir = FIGDIR / 'mcs_env_cond_reviews/ecape'
        outputs = {
            'spatial_mean': figdir / cov / method / f'fig_spatial_vars.{method}_ecape.{cov}.png',
            'cape_vs_ecape': figdir / cov / method / f'fig_cape_vs_ecape.{method}_ecape.{cov}.png',
        }
        return outputs

    var_matrix = {
        'cov': ['quick', 'full'],
        # 'cov': ['quick'],
        'method': ['peters2023'],
    }

    def rule_run(self):
        ds = xr.open_mfdataset(self.inputs.values())
        print(ds)

        print('Loading dataset')
        ds.load()
        print('Loaded')

        cu.print_mem_usage()

        fig, axes = plt.subplots(
            5, 2, sharex=True, sharey=True, layout='constrained',
            subplot_kw={'projection': ccrs.PlateCarree()}
        )
        dkeys = list(ds.data_vars.keys())
        units = {
            'CAPE': 'J kg$^{-1}$',
            'CIN': 'J kg$^{-1}$',
            'LFC': 'm',
            'EL': 'm',
            'NCAPE': 'J kg$^{-1}$',
            'V_SR': 'm s$^{-1}$',
            'Etilde': '-',
            'varepsilon': '-',
            'radius': 'm',
            'ECAPE': 'J kg$^{-1}$',
        }

        fig.set_size_inches(15, 12)
        for i, (key, ax) in enumerate(zip(dkeys, axes.flatten())):
            print(key)
            ax.set_title(key)
            im = ax.pcolormesh(ds.longitude, ds.latitude, ds[key].mean(dim='time').values)
            plt.colorbar(im, ax=ax, label=units[key])
            ax.coastlines()
        plt.savefig(self.outputs['spatial_mean'])

        plt.close('all')

        cape = ds.CAPE.values.flatten()
        ecape = ds.ECAPE.values.flatten()
        cape = cape[~np.isnan(ecape)]
        ecape = ecape[~np.isnan(ecape)]

        lr = linregress(cape, ecape)
        x = np.array([cape.min(), cape.max()])
        y = lr.slope * x + lr.intercept

        plt.figure(layout='constrained')
        im = plt.hexbin(cape, ecape, norm=LogNorm(), gridsize=40)
        plt.colorbar(im)
        plt.plot(x, y, 'k--', label=f'slope: {lr.slope:.2f}\nintercept: {lr.intercept:.2f}\nr$^2$: {lr.rvalue**2:.2f}')
        plt.xlabel('CAPE (J kg$^{-1}$)')
        plt.ylabel('ECAPE (J kg$^{-1}$)')
        plt.legend()
        plt.savefig(self.outputs['cape_vs_ecape'])
        plt.close('all')

