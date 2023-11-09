"""Remakefile to do processing of ERA5/MCS Pixel input data.

ERA5:
* Shear
* Vertically Integrated Moisture Flux Divergence (VIMFD)
* Deltas of CAPE, TCWV
* Layer means of RH, theta_e
* Monthly means

MCS Pixel (Feng et al. 2021)
* Regrid to ERA5 grid (i.e. coarsen)

ERA5 model levels count down - model level 0 is the highest, model level 137/-1 is the surface.
Note, they are 0 indexed: https://confluence.ecmwf.int/display/UDOC/L137+model+level+definitions

References:
* Feng et al. (2021), A global high-resolution mesoscale convective system database using satellite-derived cloud tops, surface precipitation, and tracking
* B. Chen, Chuntao Liu, and Mapes (2017), Relationships between large precipitating systems and atmospheric factors at a grid scale
* X. Chen et al. (2023), Environmental controls on MCS lifetime rainfall over tropical oceans
"""
import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe
from mcs_prime.era5_calc import ERA5Calc
from remake import Remake, TaskRule
from remake.util import format_path as fmtp

import mcs_prime.mcs_prime_config_util as cu


slurm_config = {'queue': 'short-serial', 'mem': 64000, 'max_runtime': '20:00:00'}
era5_process = Remake(config=dict(slurm=slurm_config, content_checks=False, no_check_input_exist=True))

pixel_inputs_cache = cu.PixelInputsCache()

# Extend years/months for shear, VIMFD because I need to have the first value preceding/following
# each year for interp to MCS times (ERA5 on the hour, MCS on the half hour).
EXTENDED_YEARS_MONTHS = []
for y in cu.YEARS:
    EXTENDED_YEARS_MONTHS.extend([*[(y - 1, 12)], *[(y, m) for m in cu.MONTHS], *[(y + 1, 1)]])
EXTENDED_YEARS_MONTHS = sorted(set(EXTENDED_YEARS_MONTHS))


class GenRegridder(TaskRule):
    """Generate a xesmf regridder for regridding/coarsening from MCS dataset to ERA5 grid."""

    rule_inputs = {
        'cape': fmtp(cu.FMT_PATH_ERA5_SFC, year=2020, month=1, day=1, hour=0, var='cape'),
        'pixel': cu.fmt_mcs_pixel_path(year=2020, month=1, day=1, hour=0),
    }
    rule_outputs = {'regridder': cu.PATH_REGRIDDER}

    def rule_run(self):
        e5cape = xr.open_dataarray(self.inputs['cape'])
        dspixel = xr.open_dataset(self.inputs['pixel'])

        e5cape = e5cape.sel(latitude=slice(60, -60)).isel(time=0)
        pixel_precip = dspixel.precipitation.isel(time=0)
        # Note, quite a challenging regrid:
        # * lat direction is different,
        # * ERA5 lon: 0--360, pixel lon: -180--180.
        # xesmf seems to manage without any issue though.
        # Uses bilinear and periodic regridding.
        regridder = xe.Regridder(pixel_precip, e5cape, 'bilinear', periodic=True)
        regridder.to_netcdf(self.outputs['regridder'])


class CalcERA5Shear(TaskRule):
    """Calculate shear from ERA5 u, v values at 3 levels

    Uses ERA5 model levels 136, 111, 100, 90 (pressure levels of approx. 1000, 800, 600, 400 hPa).
    Calculates a shear between each adjacent level, and between the surface and highest levels.
    """
    @staticmethod
    def rule_inputs(year, month):
        e5times = cu.gen_era5_times_for_month(year, month)
        fmt = cu.FMT_PATH_ERA5_ML if year not in range(2000, 2007) else cu.FMT_PATH_ERA51_ML
        e5inputs = {
            f'era5_{t}_{var}': fmtp(fmt, year=t.year, month=t.month, day=t.day, hour=t.hour, var=var)
            for t in e5times
            for var in ['u', 'v']
        }
        return e5inputs

    @staticmethod
    def rule_outputs(year, month):
        e5times = cu.gen_era5_times_for_month(year, month)
        outputs = {
            f'shear_{t}': fmtp(cu.FMT_PATH_ERA5P_SHEAR, year=t.year, month=t.month, day=t.day, hour=t.hour)
            for t in e5times
        }
        return outputs

    var_matrix = {
        ('year', 'month'): EXTENDED_YEARS_MONTHS,
    }

    def rule_run(self):
        # These are ERA5 model levels.
        # https://confluence.ecmwf.int/display/UDOC/L137+model+level+definitions
        levels = [
            136,  # 1010hPa/31m
            111,  # 804hPa/1911m
            100,  # 590hPa/4342m
            90,  # 399hPa/7214m
        ]
        e5times = cu.gen_era5_times_for_month(self.year, self.month)

        for i, time in enumerate(e5times):
            print(time)

            e5inputs = {(time, v): self.inputs[f'era5_{time}_{v}'] for v in ['u', 'v']}

            with xr.open_mfdataset(e5inputs.values()).sel(latitude=slice(60, -60), level=levels) as e5data:
                e5data = e5data.sel(time=time).load()
                u = e5data.u.values
                v = e5data.v.values

            # Calc LLS (1010-804), Low-to-mid shear (804-590), Mid-to-high (590-399), deep shear in one line.
            # LLS = shear[:, :, 0]
            # Low-to-Mid shear = shear[:, :, 1]
            # Mid-to-High = shear[:, :, 2]
            # Deep shear = shear[:, :, 3]
            # axis 1 is the level axis.
            shear = np.sqrt((np.roll(u, -1, axis=0) - u) ** 2 + (np.roll(v, -1, axis=0) - v) ** 2)
            attrs = {
                'shear_0': {
                    'description': 'Low-level shear: shear between surf and 800 hPa (ERA5 136-111)',
                    'units': 'm * s**-1',
                },
                'shear_1': {
                    'description': 'Low-to-mid shear: shear between 800 and 600 hPa (ERA5 111-100)',
                    'units': 'm * s**-1',
                },
                'shear_2': {
                    'description': 'Mid-to-high shear: shear between 600 and 400 hPa (ERA5 100-90)',
                    'units': 'm * s**-1',
                },
                'shear_3': {
                    'description': 'Deep shear: shear between surf and 400 hPa (ERA5 136-90)',
                    'units': 'm * s**-1',
                },
            }
            dsout = xr.Dataset(
                coords=dict(
                    time=[time],
                    latitude=e5data.latitude,
                    longitude=e5data.longitude,
                ),
                data_vars={
                    f'shear_{i}': (('latitude', 'longitude'), shear[i, :, :], attrs[f'shear_{i}'])
                    for i in range(len(levels))
                },
            )
            cu.to_netcdf_tmp_then_copy(dsout, self.outputs[f'shear_{time}'])


# Used for VIMFD calc.
def calc_mf_u(rho, q, u):
    """Calculates the x-component of density-weighted moisture flux on a c-grid"""

    def calc_mid(var):
        return (var + np.roll(var, -1, axis=2)) / 2

    return calc_mid(rho * q * u)


def calc_mf_v(rho, q, v):
    """Calculates the y-component of density-weighted moisture flux on a c-grid"""

    def calc_mid(var):
        s1 = (slice(None), slice(None, -1), slice(None))
        s2 = (slice(None), slice(1, None), slice(None))
        return (var[s1] + var[s2]) / 2

    return calc_mid(rho * q * v)


def calc_div_mf(rho, q, u, v, dx, dy):
    """Calculates the divergence of the moisture flux

    Switches back to original grid, but loses latitudinal extremes.
    Keeps longitudinal extremes due to biperiodic domain.
    """
    mf_u = calc_mf_u(rho, q, u)
    mf_v = calc_mf_v(rho, q, v)
    dqu_dx = (mf_u - np.roll(mf_u, 1, axis=2)) / dx[None, :, None]
    # Note, these indices are not the wrong way round!
    # latitude decreases with increasing index, hence I want the opposite
    # to what you would expect.
    dqv_dy = (mf_v[:, :-1, :] - mf_v[:, 1:, :]) / dy

    return dqu_dx[:, 1:-1] + dqv_dy


class CalcERA5VIMoistureFluxDiv(TaskRule):
    """Calculate the vertically integrated moisture flux divergence

    This is quite an involved calculation:
    * Use ERA5 u, v, T, q, and lnsp as inputs.
    * Calculate rho from lnsp (by calc'ing p, Tv, then rho using ERA5Calc.
    * Calculate u, v moisture flux (rho q u) on c-grid.
    * Calculate divergence of moisture flux on original grid (loses lat extremes)
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
            f'vimfd_{t}': fmtp(cu.FMT_PATH_ERA5P_VIMFD, year=t.year, month=t.month, day=t.day, hour=t.hour)
            for t in e5times
        }
        return outputs

    var_matrix = {
        ('year', 'month'): EXTENDED_YEARS_MONTHS,
    }

    depends_on = [ERA5Calc, calc_mf_u, calc_mf_v, calc_div_mf]

    def rule_run(self):
        Re = 6371e3  # Radius of Earth in m.
        g = 9.81  # accn due to gravity in m/s2.

        e5times = cu.gen_era5_times_for_month(self.year, self.month)
        e5calc = ERA5Calc(self.inputs['model_levels'])

        for i, time in enumerate(e5times):
            # Running out of memory doing full 4D calc: break into 24x31 3D calc.
            print(time)
            e5inputs = {(time, v): self.inputs[f'era5_{time}_{v}'] for v in ['u', 'v', 't', 'q', 'lnsp']}
            # Only load levels where appreciable q, to save memory.
            # q -> 0 by approx. level 70. Go higher (level 60=~100hPa) to be safe.
            with xr.open_mfdataset(e5inputs.values()) as ds:
                e5data = ds.isel(time=0).sel(latitude=slice(60.25, -60.25), level=slice(60, 137)).load()
                u, v = e5data.u.values, e5data.v.values
                q = e5data.q.values
                T = e5data.t.values
                lnsp = e5data.lnsp.values
                longitude = e5data.longitude
                latitude = e5data.latitude

            nlev = 137 - 60 + 2
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

            div_mf = calc_div_mf(rho, q, u, v, dx, dy)

            # Int_zs^zt(div_mf, dz) ==
            # Pressure-coord integral.
            # Int_ps^pt(1 / (rho g) div_mf, dp)
            dp = p[1:] - p[:-1]
            vimfd = (1 / (rho[:, 1:-1, :] * g) * div_mf * dp[:, 1:-1, :]).sum(axis=0)

            attrs = {
                'description': 'vertical integral of horizontal divergence of moisture flux',
                'units': 'kg * m**-2 * s**-1',
            }
            dsout = xr.Dataset(
                coords=dict(
                    time=[time],
                    latitude=latitude[1:-1],
                    longitude=longitude,
                ),
                data_vars={'vertically_integrated_moisture_flux_div': (('latitude', 'longitude'), vimfd, attrs)},
            )
            print(dsout)
            cu.to_netcdf_tmp_then_copy(dsout, self.outputs[f'vimfd_{time}'])


class CalcERA5LayerMeans(TaskRule):
    """Calculate layer means of certain ERA5 variables

    * RHlow: surf-850 hPa
    * RHmid: 700-400 hPa
    * theta_e_mid: 850-750 hPa
    Definitions taken from B. Chen et al., 2017 for RHx2.
    And from X. Chen et al. 2021 for theta_e (from their Fig. 1a, by eye).
    """
    @staticmethod
    def rule_inputs(year, month):
        e5times = cu.gen_era5_times_for_month(year, month)
        fmt = cu.FMT_PATH_ERA5_ML if year not in range(2000, 2007) else cu.FMT_PATH_ERA51_ML
        e5inputs = {
            f'era5_{t}_{var}': fmtp(fmt, year=t.year, month=t.month, day=t.day, hour=t.hour, var=var)
            for t in e5times
            for var in ['t', 'q', 'lnsp']
        }
        e5inputs['model_levels'] = cu.PATH_ERA5_MODEL_LEVELS
        return e5inputs

    @staticmethod
    def rule_outputs(year, month):
        e5times = cu.gen_era5_times_for_month(year, month)
        outputs = {
            f'layer_means_{t}': fmtp(cu.FMT_PATH_ERA5P_LAYER_MEANS, year=t.year, month=t.month, day=t.day, hour=t.hour)
            for t in e5times
        }
        return outputs

    var_matrix = {
        ('year', 'month'): EXTENDED_YEARS_MONTHS,
    }

    def rule_run(self):
        e5times = cu.gen_era5_times_for_month(self.year, self.month)
        e5calc = ERA5Calc(self.inputs['model_levels'])

        for i, time in enumerate(e5times):
            # Running out of memory doing full 4D calc: break into 24x31 3D calc.
            print(time)
            e5inputs = {(time, v): self.inputs[f'era5_{time}_{v}'] for v in ['t', 'q', 'lnsp']}
            # Only load levels where appreciable q, to save memory.
            # q -> 0 by approx. level 70. Go higher (level 60=~100hPa) to be safe.
            with xr.open_mfdataset(e5inputs.values()) as ds:
                e5data = ds.isel(time=0).sel(latitude=slice(60, -60)).load()
                q = e5data.q.values
                T = e5data.t.values
                lnsp = e5data.lnsp.values
                longitude = e5data.longitude
                latitude = e5data.latitude

            p = e5calc.calc_pressure(lnsp)
            # See for model level to pressure:
            # https://confluence.ecmwf.int/display/UDOC/L137+model+level+definitions
            # 114: 856 hPa
            RHlow = e5calc.calc_RH(p[114:], T[114:], q[114:]).mean(axis=0)

            # 90: 407 hPa
            # 105: 703 hPa
            RHmid = e5calc.calc_RH(p[90:106], T[90:106], q[90:106]).mean(axis=0)

            # 107: 742 hPa
            # 114: 857 hPa
            theta_e_mid = e5calc.calc_theta_e(p[107:115], T[107:115], q[107:115]).mean(axis=0)
            RHlow_attrs = {'ERA5 model layers': 'surf-114 (surf-856 hPa)', 'units': ''}
            RHmid_attrs = {'ERA5 model layers': '105-90 (703-407 hPa)', 'units': ''}
            theta_e_mid_attrs = {'ERA5 model layers': '114-107 (857-742 hPa)', 'units': 'K'}

            dsout = xr.Dataset(
                coords=dict(
                    time=[time],
                    latitude=latitude,
                    longitude=longitude,
                ),
                data_vars={
                    'RHlow': (('latitude', 'longitude'), RHlow, RHlow_attrs),
                    'RHmid': (('latitude', 'longitude'), RHmid, RHmid_attrs),
                    'theta_e_mid': (('latitude', 'longitude'), theta_e_mid, theta_e_mid_attrs),
                },
            )
            print(dsout)
            cu.to_netcdf_tmp_then_copy(dsout, self.outputs[f'layer_means_{time}'])


class CalcERA5Delta(TaskRule):
    """Calculate deltas of CAPE, TCWV over 3 hours.

    Stored at the future point - i.e. CAPE(t=0) - CAPE(t=-3) stored at t=0.
    """
    @staticmethod
    def rule_inputs(year, month):
        start = pd.Timestamp(year, month, 1) - pd.Timedelta(hours=3)
        end = pd.Timestamp(year, month, 1) + pd.DateOffset(months=1) - pd.Timedelta(hours=1)
        e5times = pd.date_range(start, end, freq='H')

        e5inputs = {
            f'era5_{t}_{var}': cu.era5_sfc_fmtp(var, t.year, t.month, t.day, t.hour)
            for t in e5times
            for var in ['cape', 'tcwv']
        }
        # Slightly tricky to do derived variables because I need 3 hours prev. (these are guaranteed to be
        # there for ERA5.
        # fmt = cu.FMT_PATH_ERA5P_VIMFD
        # e5inputs.update({
        #     f'era5_{t}_vertically_integrated_moisture_flux_div': fmtp(fmt, year=t.year, month=t.month, day=t.day, hour=t.hour)
        #     for t in e5times
        # })
        return e5inputs

    @staticmethod
    def rule_outputs(year, month):
        e5times = cu.gen_era5_times_for_month(year, month)
        outputs = {
            f'era5_{t}': fmtp(cu.FMT_PATH_ERA5P_DELTA, year=t.year, month=t.month, day=t.day, hour=t.hour)
            for t in e5times
        }
        return outputs

    var_matrix = {
        ('year', 'month'): EXTENDED_YEARS_MONTHS,
    }

    def rule_run(self):
        e5vars = ['cape', 'tcwv']
        e5times = cu.gen_era5_times_for_month(self.year, self.month)

        for i, time in enumerate(e5times):
            # Running out of memory doing full 4D calc: break into 24x31 3D calc.
            print(time)
            e5inputs = {
                (t, var): self.inputs[f'era5_{t}_{var}']
                for t in [time - pd.Timedelta(hours=3), time]
                for var in e5vars
            }
            print(e5inputs)
            with xr.open_mfdataset(e5inputs.values()) as ds:
                e5data = ds.sel(latitude=slice(60, -60)).load()
            print(e5data.isel(time=1).time)
            print(e5data.isel(time=0).time)
            dsout = e5data.isel(time=1) - e5data.isel(time=0)
            dsout = dsout.rename({var: f'delta_3h_{var}'
                                  for var in e5vars})
            CAPE_attrs = {
                'description': '3 hour delta of CAPE (CAPE(t=0) - CAPE(t=-3), stored at t=0',
                'units': 'J * kg**-1'
            }
            TCWV_attrs = {
                'description': '3 hour delta of TCWV (TCWV(t=0) - TCWV(t=-3), stored at t=0',
                'units': 'mm'
            }
            dsout['delta_3h_cape'].attrs.update(CAPE_attrs)
            dsout['delta_3h_tcwv'].attrs.update(TCWV_attrs)
            dsout = dsout.assign_coords({'time': [time]})

            print(dsout)
            cu.to_netcdf_tmp_then_copy(dsout, self.outputs[f'era5_{time}'])


class GenPixelDataOnERA5Grid(TaskRule):
    """Generate ERA5 pixel data by regridding native MCS dataset masks/cloudnumbers

    The MCS dataset pixel-level data has a field cloudnumber, which maps onto the
    corresponding cloudnumber field in the tracks data. Here, for each time (on
    the half-hour, to match MCS dataset), I regrid this data to the ERA5 grid.
    It is done this way round because the pixel-level data is on the IMERG grid, which
    is finer than ERA5 (0.1deg vs 0.25deg).
    This is non-trivial, because they use different longitude endpoints. The regridding/
    coarsening is done by xesmf, which handled this and periodic longitude.

    The scheme for this is:
    * load pixel to ERA5 regridder.
    * for each hour for which there is data, do regridding.
    * load pixel-level cloudnumber/Tb.
    * for each cloudnumber, regrid the mask for that cloudnumber, keeping all values
      greater than 0.5.
    * combine into one integer field.
    * also regrid Tb, precipitation fields.
    * save compressed int16 results (cloudnumber).
    """

    @staticmethod
    def rule_inputs(year, month, day):
        # Just need one to recreate regridder.
        e5inputs = {'cape': fmtp(cu.FMT_PATH_ERA5_SFC, year=2020, month=1, day=1, hour=0, var='cape')}

        # Not all files exist! Do not include those that don't.
        pixel_times, pixel_inputs = pixel_inputs_cache.all_pixel_inputs[(year, month, day)]
        inputs = {**e5inputs, **pixel_inputs}

        inputs['regridder'] = GenRegridder.rule_outputs['regridder']
        return inputs

    @staticmethod
    def rule_outputs(year, month, day):
        pixel_times, pixel_inputs = pixel_inputs_cache.all_pixel_inputs[(year, month, day)]
        pixel_output_paths = [
            fmtp(cu.FMT_PATH_PIXEL_ON_ERA5, year=t.year, month=t.month, day=t.day, hour=t.hour) for t in pixel_times
        ]
        outputs = {f'pixel_on_e5_{t}': p for t, p in zip(pixel_times, pixel_output_paths)}
        return outputs

    # Note, not all days have any data in them. PIXEL_DATE_KEYS has been filtered
    # to exclude these.
    var_matrix = {('year', 'month', 'day'): cu.PIXEL_DATE_KEYS}

    def rule_run(self):
        e5cape = xr.open_dataarray(self.inputs['cape']).sel(latitude=slice(60, -60))
        pixel_times = cu.gen_pixel_times_for_day(self.year, self.month, self.day)

        # Note, this is a subset of times based on which times exist.
        pixel_inputs = {t: self.inputs[f'pixel_{t}'] for t in pixel_times if f'pixel_{t}' in self.inputs}

        data = np.zeros((len(e5cape.latitude), len(e5cape.longitude)))
        regridder = None
        for i, time in enumerate(pixel_inputs.keys()):
            print(time)
            dsout = xr.Dataset(
                coords=dict(
                    time=[time],
                    latitude=e5cape.latitude,
                    longitude=e5cape.longitude,
                ),
                data_vars={
                    'cloudnumber': (('latitude', 'longitude'), data.copy()),
                    'tb': (('latitude', 'longitude'), data.copy()),
                    'precipitation': (('latitude', 'longitude'), data.copy()),
                },
            )
            # Load the pixel data.
            with xr.open_dataset(pixel_inputs[time]) as dspixel:
                pixel_precip = dspixel.precipitation.isel(time=0).load()
                cloudnumber = dspixel.cloudnumber.isel(time=0).load()
                tb = dspixel.tb.isel(time=0).load()

            if regridder is None:
                # Reload the regridder.
                regridder = xe.Regridder(
                    pixel_precip,
                    e5cape,
                    'bilinear',
                    periodic=True,
                    reuse_weights=True,
                    weights=self.inputs['regridder'],
                )

            # Perform the regrid for each cloudnumber.
            mask_regridded = []
            max_cn = int(cloudnumber.values[~np.isnan(cloudnumber.values)].max())
            cns = range(1, max_cn + 1)
            # The idea here is to stack masks together, one for earch cn, then sum along
            # the stacked dim to get a cn field on ERA5 grid.
            for cn in cns:
                mask_regridded.append(regridder((cloudnumber == cn).astype(float)))

            da_mask_regridded = xr.concat(mask_regridded, pd.Index(cns, name='cn'))
            # Note, da_mask_regridded will have shape (len(cns), len(lat), len(lon))
            # and the `* da_mask_regridded.cn` will multiply each 1/0 bool mask by its
            # corresponding # cn, turning 1s into its cloudnumber. Summing along the cn index
            # turns this into a (len(lat), len(lon)) field.
            # Note, a regridded value belongs to a cn if it is over 0.5, so that masks should not
            # overlap.
            cn_on_e5g = ((da_mask_regridded > 0.5).astype(int) * da_mask_regridded.cn).sum(dim='cn')
            mcs_mask = cn_on_e5g.values[0] > 0.5

            tb_on_e5g = regridder(tb)
            precipitation_on_e5g = regridder(pixel_precip)

            dsout.cloudnumber.values = cn_on_e5g
            dsout.cloudnumber.attrs.update({'units': ''})
            dsout.tb.values = tb_on_e5g
            dsout.tb.attrs.update({'units': 'K'})
            dsout.precipitation.values = precipitation_on_e5g
            dsout.tb.attrs.update({'units': 'mm * hr**-1'})
            # N.B. I have checked that the max cloudnumber (1453) in the tracks dataset is < 2**16
            # assert cns.max() < 2**16 - 1
            dsout.attrs.update({
                'description': 'Pixel data from Feng et al. 2021 (https://doi.org/10.1029/2020JD034202)',
                'regridding': 'Regridded onto ERA5 grid',
            })

            encoding = {
                # Save cloudnumber as compressed int16 field for lots of compression!
                'cloudnumber': dict(dtype='int16', zlib=True, complevel=4),
                # Compress these.
                'tb': dict(zlib=True, complevel=4),
                'precipitation': dict(zlib=True, complevel=4),
            }
            cu.to_netcdf_tmp_then_copy(dsout, self.outputs[f'pixel_on_e5_{time}'], encoding=encoding)


class CalcERA5MeanField(TaskRule):
    """Calculate monthly mean field for base and derived ERA5 variables from hourly data"""
    @staticmethod
    def rule_inputs(year, month):
        # start = pd.Timestamp(year, month, 1)
        # end = start + pd.DateOffset(months=1)
        # e5times = pd.date_range(start, end, freq='H')
        e5times = cu.gen_era5_times_for_month(year, month)
        e5inputs = {
            f'era5_{t}_{var}': cu.era5_sfc_fmtp(var, t.year, t.month, t.day, t.hour)
            for t in e5times
            for var in cu.ERA5VARS
        }
        e5proc_shear = CalcERA5Shear.rule_outputs(year=year, month=month)
        e5proc_vimfd = CalcERA5VIMoistureFluxDiv.rule_outputs(year=year, month=month)
        inputs = {
            **e5inputs,
            **e5proc_shear,
            **e5proc_vimfd,
        }
        return inputs

    rule_outputs = {'meanfield': (cu.FMT_PATH_ERA5_MEANFIELD)}

    var_matrix = {
        'year': cu.YEARS,
        'month': cu.MONTHS,
    }

    def load_data(self):
        e5times = cu.gen_era5_times_for_month(self.year, self.month)

        e5paths = [self.inputs[f'era5_{t}_{v}'] for t in e5times for v in cu.ERA5VARS]
        e5proc_shear_paths = [self.inputs[f'shear_{t}'] for t in e5times]
        e5proc_vimfd_paths = [self.inputs[f'vimfd_{t}'] for t in e5times]

        self.logger.debug('Open ERA5')
        e5ds = xr.open_mfdataset(e5paths).sel(latitude=slice(60, -60))
        self.logger.debug('Open proc shear')
        e5shear = xr.open_mfdataset(e5proc_shear_paths)
        self.logger.debug('Open proc VIMFD')
        e5vimfd = xr.open_mfdataset(e5proc_vimfd_paths)

        return xr.merge([e5ds.load(), e5shear.load(), e5vimfd.load()])

    def rule_run(self):
        self.logger.info('Load data')
        ds = self.load_data()
        dsout = ds.mean(dim='time').load()
        dsout = dsout.expand_dims({'time': 1})
        dsout = dsout.assign_coords({'time': [pd.Timestamp(ds.time.mean().item())]})
        dsout.attrs['ntimes'] = len(ds.time)
        dsout.attrs['description'] = 'monthly mean fields from hourly ERA5 data'
        print(dsout)

        comp = dict(zlib=True, complevel=4)
        encoding = {var: comp for var in dsout.data_vars}
        cu.to_netcdf_tmp_then_copy(dsout, self.outputs['meanfield'], encoding=encoding)
