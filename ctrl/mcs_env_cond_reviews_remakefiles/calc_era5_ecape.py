import sys
from pathlib import Path

from metpy.units import units
import numpy as np
import pandas as pd
import xarray as xr

from remake import Remake, TaskRule

import mcs_prime.mcs_prime_config_util as cu
from mcs_prime.era5_calc import ERA5Calc

sys.path.insert(0, '/home/users/mmuetz/projects/ecape_calc')
# My version of peters2023 code.
from ecape_calc import compute_CAPE_CIN, compute_NCAPE, compute_VSR, compute_ETILDE
from params import T1, T2

# Capella github with minor mods.
from ecape.calc import calc_ecape



OUTDIR = cu.PATHS['outdir']

# slurm_config = {'account': 'short4hr', 'queue': 'short-serial-4hr', 'mem': 64000}
slurm_config = {'queue': 'short-serial', 'mem': 64000}
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
        print(p[-1].mean())
        raise
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
    # ERA5 indexes levels from highest to lowest.
    levlatlon_idx = (slice(None, None, -1), lat_idx, lon_idx)
    T0 = ds.t.values[levlatlon_idx]
    q0 = ds.q.values[levlatlon_idx]
    u0 = ds.u.values[levlatlon_idx]
    v0 = ds.v.values[levlatlon_idx]
    p0 = comp_alt.p[levlatlon_idx]
    z0 = comp_alt.alt[levlatlon_idx]

    CAPE, CIN, LFC, EL = compute_CAPE_CIN(T0, p0, q0, 0, 0, 0, z0, T1, T2)

    if not (np.isnan(LFC) or np.isnan(EL)):
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
    z0 = comp_alt.alt[levlatlon_idx]

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
        if cov == 'quick':
            outputs = {'output': outdir / 'quick' / method / f'ecmwf-era5_oper_an_ml_{tstamp}.{method}_ecape.quick.nc'}
        else:
            outputs = {'output': outdir / 'full' / method / f'ecmwf-era5_oper_an_ml_{tstamp}.{method}_ecape.nc'}
        return outputs

    var_matrix = {
        'date': dates,
        # 'cov': ['quick', 'full'],
        'cov': ['quick'],
        # 'method': ['peters2023', 'capella'],
        'method': ['capella'],
    }

    depends_on = [peters23_compute_ECAPE_etc]

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
        successful = 0
        unsuccessful = 0
        for i, lon_idx in enumerate(lons):
            print(f'{i / len(lons) * 100:.1f}%')
            for j, lat_idx in enumerate(lats):
                try:
                    if self.method == 'peters2023':
                        ecape_dict = peters23_compute_ECAPE_etc(ds, comp_alt, lon_idx, lat_idx)
                    elif self.method == 'capella':
                        ecape_dict = capella_compute_ECAPE_etc(ds, comp_alt, lon_idx, lat_idx)

                    successful += 1
                    for k, key in enumerate(ecape_dict.keys()):
                        output[k, j, i] = ecape_dict[key]
                except Exception as e:
                    unsuccessful += 1
                    # print(e)
                    pass
            print(successful, unsuccessful)

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
