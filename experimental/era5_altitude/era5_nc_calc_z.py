import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from mcs_prime.era5_calc import ERA5Calc

Rd = 287.06
g = 9.80665
Re = 6371222.9 # m

def compute_z_level(lev_idx, p, t_moist, z_h):
    '''Compute z at half & full level for the given level, based on t/q/sp'''
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
    z_f = z_h + (t_moist[lev_idx] * alpha)

    # z_h is the geopotential of 'half-levels'
    # integrate z_h to next half level
    z_h = z_h + (t_moist[lev_idx] * dlog_p)

    return z_h, z_f

if __name__ == '__main__':
    try:
        print(e5calc)
    except NameError:
        basepath = '/badc/ecmwf-era5/data/oper/an_ml/2020/01/01/'

        e5calc = ERA5Calc('/gws/nopw/j04/mcs_prime/mmuetz/data/ERA5/ERA5_L137_model_levels_table.csv')

        t = xr.load_dataarray(basepath + 'ecmwf-era5_oper_an_ml_202001010000.t.nc')[0].sel(latitude=slice(60, -60))
        q = xr.load_dataarray(basepath + 'ecmwf-era5_oper_an_ml_202001010000.q.nc')[0].sel(latitude=slice(60, -60))
        zsfc = xr.load_dataarray(basepath + 'ecmwf-era5_oper_an_ml_202001010000.z.nc')[0].sel(latitude=slice(60, -60))
        lnsp = xr.load_dataarray(basepath + 'ecmwf-era5_oper_an_ml_202001010000.lnsp.nc')[0].sel(latitude=slice(60, -60))
        p = e5calc.calc_pressure(lnsp.values)
        print(p.shape)

        # Get levels in ascending order of height (starts at 137)
        levels = t.level.values[::-1]
        print(levels)

        t_moist = t.values * (1. + 0.609133 * q.values) * Rd
        z_h = zsfc.values

        z = np.zeros_like(p)
        for lev in levels:
            lev_idx = lev - 1
            print(lev, lev_idx)
            z_h, z_f = compute_z_level(lev_idx, p, t_moist, z_h)
            z[lev_idx] = z_f

        h = z / g
        alt = Re * h / (Re - h)

    # plt.contourf(zsfc.longitude.values, zsfc.latitude.values, alt[-1], levels=np.arange(-500, 9000, 500))
    # plt.colorbar(label='altitude (m)')
    # plt.show()

    # TODO:
    # This *should be* the same as zsfc, but is not. WHY??
    dsz = xr.open_dataset('zlnsp_ml.grib', engine='cfgrib')

    # This is -180 to 180. Above is 0 to 360. Sigh.
    ds = xr.open_dataset('z_out.grib', engine='cfgrib')
    z2 = ds.z
    h2 = z2 / g
    alt2 = Re * h2 / (Re - h2)

    fig, axes = plt.subplots(3, 2)

    axes[0, 0].set_title('surf')
    im = axes[0, 0].contourf(ds.longitude.values, ds.latitude.values, np.roll(alt[-1], 720, axis=1))
    plt.colorbar(im, ax=axes[0, 0], label='altitude (m)')

    axes[1, 0].set_title('ml 10')
    im = axes[1, 0].contourf(ds.longitude.values, ds.latitude.values, np.roll(alt[-10], 720, axis=1))
    plt.colorbar(im, ax=axes[1, 0], label='altitude (m)')

    axes[2, 0].set_title('top')
    im = axes[2, 0].contourf(ds.longitude.values, ds.latitude.values, np.roll(alt[0], 720, axis=1))
    plt.colorbar(im, ax=axes[2, 0], label='altitude (m)')

    axes[0, 1].set_title('surf')
    im = axes[0, 1].contourf(ds.longitude.values, ds.latitude.values, alt2[-1])
    plt.colorbar(im, ax=axes[0, 1], label='altitude (m)')

    axes[1, 1].set_title('ml 10')
    im = axes[1, 1].contourf(ds.longitude.values, ds.latitude.values, alt2[-10])
    plt.colorbar(im, ax=axes[1, 1], label='altitude (m)')

    axes[2, 1].set_title('top')
    im = axes[2, 1].contourf(ds.longitude.values, ds.latitude.values, alt2[0])
    plt.colorbar(im, ax=axes[2, 1], label='altitude (m)')

    def rmse(a1, a2):
        return np.sqrt(((a1 - a2)**2).mean())

    rmses = np.zeros(alt.shape[0])
    for lev in levels:
        lev_idx = lev - 1
        rmses[lev_idx] = rmse(np.roll(alt[lev_idx], 720, axis=1), alt2[lev_idx])

    plt.figure()
    plt.subplot(121)
    plt.plot(alt2.mean(axis=(1, 2)), rmses)
    plt.xlim((0, 20000))
    plt.ylim((0, 250))
    plt.subplot(122)
    plt.plot(alt2.mean(axis=(1, 2)), rmses / alt2.mean(axis=(1, 2)) * 100)
    plt.xlim((0, 20000))
    plt.ylim((0, 4))
    plt.show()





    # TODO: It must be possible to do it like this, but the flipped signs and levels
    # is beyond me right now.
    # # if lev == 1:
    # #     dlog_p = np.log(ph_levplusone / 0.1)
    # #     alpha = np.log(2)
    # # else:
    # #     dlog_p = np.log(ph_levplusone / ph_lev)
    # #     alpha = 1. - ((ph_lev / (ph_levplusone - ph_lev)) * dlog_p)

    # dlog_p = np.zeros_like(p)
    # dlog_p[0] = np.log(p[0] / 0.1)
    # dlog_p[1:] = np.log(p[:-1] / p[1:])

    # alpha = np.zeros_like(p)
    # alpha[0] = np.log(2)
    # alpha[1:] = 1. - ((p[1:] / (p[:-1] - p[1:])) * dlog_p)




