import sys
from itertools import product

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import metpy.calc as mpcalc
from metpy.plots import Hodograph, SkewT
from metpy.units import units
import numpy as np
import xarray as xr

from IPython.display import clear_output

# Contains previous code for calculating pressure from lnsp.
from mcs_prime.era5_calc import ERA5Calc

sys.path.insert(0, '/home/users/mmuetz/projects/ecape_calc')
from ecape_calc import compute_CAPE_CIN, compute_NCAPE, compute_VSR, compute_ETILDE
from params import T1, T2


basepath = '/badc/ecmwf-era5/data/oper/an_ml/2020/01/01/'

paths = [
    basepath + f'ecmwf-era5_oper_an_ml_202001010000.{v}.nc'
    for v in ['lnsp', 'z', 't', 'q', 'u', 'v']
]

ds = xr.open_mfdataset(paths).isel(time=0).sel(latitude=slice(60, -60)).load()

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


comp_alt = Era5ComputeAlt(ds)
_ = comp_alt.run()


import datetime as dt
from timeit import default_timer as timer

class LoopPercentageDisp:
    def __init__(self, L, step_percentage=10, fmt='{percentage:.1f}% ({i}/{L}): elapsed: {elapsed:.1f}s, est dur: {est_duration:.1f}s - est end: {est_end_time}'):
        self.L = L
        self.step_percentage = step_percentage
        self.disp_percentage = step_percentage
        self.fmt = fmt
        self.curr_percentage = 0
        self.start_time = None

    def __call__(self, i):
        L = self.L
        frac = i / L
        percentage = frac * 100

        if not self.start_time:
            self.start_time = dt.datetime.now()
            print(f'Start at: {self.start_time}')
            self.start = timer()
            elapsed = 0
        else:
            self.curr = timer()
            elapsed = self.curr - self.start

            est_duration = elapsed / frac
            est_end_time = self.start_time + dt.timedelta(seconds=elapsed)

        if percentage >= self.disp_percentage:
            print(self.fmt.format(
                percentage=percentage,
                i=i,
                L=L,
                elapsed=elapsed,
                est_duration=est_duration,
                est_end_time=est_end_time
            ))
            self.disp_percentage += self.step_percentage
        self.curr_percentage = percentage


def compute_ECAPE_etc(ds, comp_alt, lon_idx, lat_idx):
    # ERA5 indexes levels from highest to lowest.
    levlatlon_idx = (slice(None, None, -1), lat_idx, lon_idx)
    T0 = ds.t.values[levlatlon_idx]
    q0 = ds.q.values[levlatlon_idx]
    u0 = ds.u.values[levlatlon_idx]
    v0 = ds.v.values[levlatlon_idx]
    p0 = comp_alt.p[levlatlon_idx]
    z0_msl = comp_alt.alt[levlatlon_idx]
    z0 = z0_msl - z0_msl[0]  # Convert into height above surface.

    CAPE, CIN, LFC, EL = compute_CAPE_CIN(T0, p0, q0, 0, 0, 0, z0, T1, T2)

    if not (np.isnan(LFC) or np.isnan(EL)) and CAPE > 100:
        NCAPE, MSE0_star, MSE0bar = compute_NCAPE(T0, p0, q0, z0, T1, T2, LFC, EL)

        # Get the 0-1 km mean storm-relative wind, estimated using bunkers2000 method for right-mover storm motion
        V_SR, C_x, C_y = compute_VSR(z0, u0, v0)

        # Get e_tilde, which is the ratio of ecape to cape.  also, varepsilon is the fracitonal entrainment rate, and radius is the theoretical upraft radius
        Etilde, varepsilon, radius = compute_ETILDE(CAPE, NCAPE, V_SR, EL, 120)
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
            CAPE=CAPE,
            CIN=CIN,
            LFC=LFC,
            EL=EL,
            NCAPE=None,
            V_SR=None,
            Etilde=None,
            varepsilon=None,
            radius=None,
            ECAPE=None,
        )


lons = list(range(0, 1440, 10))
lats = list(range(0, 481, 10))
output = np.full((10, len(lats), len(lons)), np.nan)
lpd = LoopPercentageDisp(len(lons))
for i, lon_idx in enumerate(lons):
    lpd(i)
    for j, lat_idx in enumerate(lats):
        # print(lon_idx, lat_idx)
        try:
            ecape_dict = compute_ECAPE_etc(ds, comp_alt, lon_idx, lat_idx)
            for k, key in enumerate(ecape_dict.keys()):
                output[k, j, i] = ecape_dict[key]
        except Exception as e:
            # output[(lon_idx, lat_idx)] = e
            pass


lon = ds.longitude.values[lons]
lat = ds.latitude.values[lats]


from ecape_calc import (
    lift_parcel_adiabatic,
    compute_CAPE_CIN,
    compute_NCAPE,
    compute_VSR,
    compute_ETILDE,
)

from consts import *
from params import *


def plot_skewt(z0, p0, T0, q0, u0, v0, C_x, C_y, Etilde, CAPE, varepsilon, radius):
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
    #     "text.usetex": True,
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
    skew.plot_barbs(p[0::5], u0[0::5], v0[0::5], x_clip_radius=0.1, y_clip_radius=0.08)
    skew.ax.set_ylim(1020, 100)
    skew.ax.set_xlim(-20, 40)
    skew.ax.text(
        -15,
        900,
        f'CAPE: {CAPE:.0f} J kg$^{{-1}}$\n'
        f'ECAPE: {Etilde * CAPE:.0f} J kg$^{{-1}}$\n'
        f'$\widetilde{{\mathrm{{E}}}}_A$: {Etilde * 100:.0f}\%\n'
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
    h = Hodograph(ax_hod, component_range=55.0)
    h.add_grid(increment=20, linewidth=0.5)

    fplt = np.where(z0 <= 6000)[0]

    cmap = plt.get_cmap('autumn_r', len(fplt))
    h.plot_colormapped(
        u0[fplt],
        v0[fplt],
        np.floor(z0[fplt] / 1000),
        linewidth=2,
        cmap=plt.get_cmap('gist_rainbow_r', 6),
    )  # Plot a line colored by wind speed
    plt.xlim(-50, 50)
    plt.ylim(-50, 50)
    plt.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        left=False,
        top=False,  # ticks along the top edge are off
        labelleft=False,
        labelbottom=False,
    )  # labels along the bottom edge are off
    plt.text(-5, -21, '20 kt', fontsize=8)
    plt.text(-5, -41, '40 kt', fontsize=8)
    # TODO: what is this meant to plot?
    # storm relative motion I suspect.
    #plt.plot(C_x, C_y, 'ko', markersize=2.5)

    # plt.savefig('figs/sndfig.pdf')
    plt.show()


def plot_profile(ds, comp_alt, lon_idx, lat_idx):
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
    print(ecape_dict)
    Etilde = ecape_dict['Etilde']
    CAPE = ecape_dict['CAPE']
    varepsilon = ecape_dict['varepsilon']
    radius = ecape_dict['radius']
    plot_skewt(z0, p0, T0, q0, u0, v0, None, None, Etilde, CAPE, varepsilon, radius)


lat_idx, lon_idx = np.where(~np.isnan(output[-1]))
lat_idx *= 10
lon_idx *= 10
plot_profile(ds, comp_alt, lon_idx[0], lat_idx[0])
