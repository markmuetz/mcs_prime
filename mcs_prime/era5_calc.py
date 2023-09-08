import metpy.calc as mpcalc
from metpy.units import units
import numpy as np
import pandas as pd


class ERA5Calc:
    def __init__(self, model_levels_table_file):
        self.df_ecmwf = pd.read_csv(model_levels_table_file)

    def calc_pressure(self, lnsp):
        sp = np.exp(lnsp)  # pressure in Pa.
        a = self.df_ecmwf['a [Pa]'].values  # a in Pa.
        b = self.df_ecmwf.b.values  # b unitless.
        # Broadcasting to correctly calc 3D pressure field.
        # (level, latitude, longitude)
        p_half = a[:, None, None] + b[:, None, None] * sp[None, :, :]
        p = (p_half[:-1] + p_half[1:]) / 2
        return p

    def calc_Tv(self, T, q):
        return mpcalc.virtual_temperature(T * units.K, q * units('kg/kg')).magnitude

    def calc_rho(self, p, Tv, Rd=287):
        rho = p / (Rd * Tv)
        return rho

    def calc_theta_e(self, p, T, q):
        Td = mpcalc.dewpoint_from_specific_humidity(p * units.Pa, T * units.K, q * units('kg/kg'))
        return mpcalc.equivalent_potential_temperature(p * units.Pa, T * units.K, Td).magnitude

    def calc_RH(self, p, T, q):
        return mpcalc.relative_humidity_from_specific_humidity(p * units.Pa,
                                                               T * units.K,
                                                               q * units('kg/kg')).magnitude

