# coding: utf-8
import datetime as dt

import cartopy.crs as ccrs
import matplotlib.pyplot as plt

from mcs_prime import PATHS
from mcs_prime import McsTracks


tracks = McsTracks.load(PATHS['statsdir'] / 'mcs_tracks_final_extc_20190101.0000_20200101.0000.nc', PATHS['pixeldir'])

tracks.plot_diurnal_cycle(dhours=3)
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
plt.show()
