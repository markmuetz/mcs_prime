# coding: utf-8
import sys
import datetime as dt

import cartopy.crs as ccrs
import matplotlib.pyplot as plt

from mcs_prime import PATHS
from mcs_prime import McsTracks

interactive = len(sys.argv) == 2 and sys.argv[1] == 'interactive'
dhours = 3

tracks = McsTracks.load(PATHS['statsdir'] / 'mcs_tracks_final_extc_20190101.0000_20200101.0000.nc', PATHS['pixeldir'])

for anomaly in [True, False]:
    tracks.plot_diurnal_cycle(dhours=dhours, anomaly=anomaly)
    if interactive:
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()
        plt.show()
    else:
        figpath = PATHS['figdir'] / 'diurnal_cycle' / f'diurnal_cycle_mcs_tracks_final_extc_20190101.0000_20200101.0000.{dhours=}.{anomaly=}.png'
        figpath.parent.mkdir(exist_ok=True, parents=True)
        fig = plt.gcf()
        fig.set_size_inches(20, 11.26)  # full screen
        plt.savefig(figpath, bbox_inches='tight', dpi=100)

