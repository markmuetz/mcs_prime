"""Remakefile to do processing of ERA5/MCS Pixel input data.

copied from mcs_prime/ctrl/remakefiles/era5_process.py

ERA5:
* Divergence
"""
import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe
from mcs_prime.era5_calc import ERA5Calc
from remake import Remake, TaskRule
from remake.util import format_path as fmtp

import mcs_prime.mcs_prime_config_util as cu

YEARS = [2020]


slurm_config = {'queue': 'short-serial', 'mem': 64000, 'max_runtime': '20:00:00'}
era5_process = Remake(config=dict(slurm=slurm_config, content_checks=False, no_check_input_exist=True))

# Extend years/months for shear, VIMFD because I need to have the first value preceding/following
# each year for interp to MCS times (ERA5 on the hour, MCS on the half hour).
EXTENDED_YEARS_MONTHS = []
for y in YEARS:
    EXTENDED_YEARS_MONTHS.extend([*[(y - 1, 12)], *[(y, m) for m in cu.MONTHS], *[(y + 1, 1)]])
EXTENDED_YEARS_MONTHS = sorted(set(EXTENDED_YEARS_MONTHS))


def calc_div(u, v, dx, dy):
    """Calculates divergence

    Switches back to original grid, but loses latitudinal extremes.
    Keeps longitudinal extremes due to biperiodic domain.
    """
    def calc_mid_u(var):
        return (var + np.roll(var, -1, axis=2)) / 2
    def calc_mid_v(var):
        s1 = (slice(None), slice(None, -1), slice(None))
        s2 = (slice(None), slice(1, None), slice(None))
        return (var[s1] + var[s2]) / 2


    u_mid = calc_mid_u(u)
    v_mid = calc_mid_v(v)

    du_dx = (u_mid - np.roll(u_mid, 1, axis=2)) / dx[None, :, None]
    # Note, these indices are not the wrong way round!
    # latitude decreases with increasing index, hence I want the opposite
    # to what you would expect.
    dv_dy = (v_mid[:, :-1, :] - v_mid[:, 1:, :]) / dy

    return du_dx[:, 1:-1] + dv_dy


class CalcERA5Div(TaskRule):
    """Calculate the divergence and its vertical integral at/up to different levels

    Borrows heavily from era5_process.py:CalcERA5VIMoistureFluxDiv

    This is quite an involved calculation:
    * Use ERA5 u, v, T, q, and lnsp as inputs.
    * Calculate rho from lnsp (by calc'ing p, Tv, then rho using ERA5Calc.
    * Calculate div(u, v)
    * Calculate divergence on original grid (loses lat extremes)
    * Do pressure-weighted integral over atmosphere.
    """
    @staticmethod
    def rule_inputs(year, month):
        e5times = cu.gen_era5_times_for_month(year, month)
        fmt = cu.FMT_PATH_ERA5_ML if year not in range(2000, 2007) else cu.FMT_PATH_ERA51_ML
        e5inputs = {
            f'era5_{t}_{var}': fmtp(fmt, year=t.year, month=t.month, day=t.day, hour=t.hour, var=var)
            for t in e5times
            for var in ['u', 'v', 't', 'q', 'lnsp']
        }
        e5inputs['model_levels'] = cu.PATH_ERA5_MODEL_LEVELS
        return e5inputs

    @staticmethod
    def rule_outputs(year, month):
        e5times = cu.gen_era5_times_for_month(year, month)
        outputs = {
            f'div_{t}': fmtp(cu.FMT_PATH_ERA5P_DIV, year=t.year, month=t.month, day=t.day, hour=t.hour)
            for t in e5times
        }
        return outputs

    var_matrix = {
        ('year', 'month'): EXTENDED_YEARS_MONTHS,
    }

    depends_on = [ERA5Calc, calc_div]

    def rule_run(self):
        # https://confluence.ecmwf.int/display/UDOC/L137+model+level+definitions
        levels = [
            114,  # 850 hPa
            105,  # 700 hPa
            101,  # 600 hPa
        ]
        approx_pressure_level = {
            114: '850 hPa',
            105: '700 hPa',
            101: '600 hPa',
        }
        Re = 6371e3  # Radius of Earth in m.
        g = 9.81  # accn due to gravity in m/s2.

        e5times = cu.gen_era5_times_for_month(self.year, self.month)
        e5calc = ERA5Calc(self.inputs['model_levels'])

        for i, time in enumerate(e5times):
            # Running out of memory doing full 4D calc: break into 24x31 3D calc.
            print(time)
            e5inputs = {(time, v): self.inputs[f'era5_{time}_{v}'] for v in ['u', 'v', 't', 'q', 'lnsp']}
            # Only need data up to (level 101=~100hPa) to save memory.
            with xr.open_mfdataset(e5inputs.values()) as ds:
                e5data = ds.isel(time=0).sel(latitude=slice(60.25, -60.25), level=slice(101, 137)).load()
                u, v = e5data.u.values, e5data.v.values
                q = e5data.q.values
                T = e5data.t.values
                lnsp = e5data.lnsp.values
                longitude = e5data.longitude
                latitude = e5data.latitude

            nlev = 137 - 101 + 2
            p = e5calc.calc_pressure(lnsp)[-nlev:]
            Tv = e5calc.calc_Tv(T, q)
            print(p.mean(axis=(1, 2)))
            print('p', p.shape)
            print('Tv', Tv.shape)
            rho = e5calc.calc_rho(p[1:], Tv)

            # Calc dx/dy.
            dx_deg = longitude.values[1] - longitude.values[0]
            dy_deg = latitude.values[0] - latitude.values[1]  # N.B. want positive so swap indices.

            dy = dy_deg / 360 * 2 * np.pi * Re
            dx = np.cos(latitude.values * np.pi / 180) * dx_deg / 360 * 2 * np.pi * Re

            # 3D field.
            div = calc_div(u, v, dx, dy)

            data_vars = {}
            press_level_url = 'https://confluence.ecmwf.int/display/UDOC/L137+model+level+definitions'

            # Store divergence at each model level.
            for level in levels:
                attrs = {
                    'description': 'horizontal divergence',
                    'model_level': level,
                    'approx_pressure_level': approx_pressure_level[level],
                    'see for pressure level': press_level_url,
                    'units': 's**-1',
                }
                data_vars[f'div_ml{level}'] = (('latitude', 'longitude'), div[level - 101], attrs)

            # Store vertical integral of divergence up to each model level.

            # Int_zs^zt(div_mf, dz) ==
            # Pressure-coord integral.
            # Int_ps^pt(1 / (rho g) div_mf, dp)
            dp = p[1:] - p[:-1]
            # Calc 3D integrand then subset for vertical integral.
            integrand = 1 / (rho[:, 1:-1, :] * g) * div * dp[:, 1:-1, :]
            for level in levels:
                vidiv = (integrand[level - 101:]).sum(axis=0)
                attrs = {
                    'description': 'vertical integral of horizontal divergence',
                    'model_levels': f'{level}-137',
                    'approx_pressure_levels': f'surface to {approx_pressure_level[level]}',
                    'see for pressure level': press_level_url,
                    'units': 'm * s**-1',
                }
                data_vars[f'vertically_integrated_div_ml{level}_surf'] = (
                    ('latitude', 'longitude'),
                    vidiv,
                    attrs
                )

            dsout = xr.Dataset(
                coords=dict(
                    time=[time],
                    latitude=latitude[1:-1],
                    longitude=longitude,
                ),
                data_vars=data_vars,
            )
            print(dsout)
            cu.to_netcdf_tmp_then_copy(dsout, self.outputs[f'div_{time}'])

