# coding: utf-8
import sys
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

from mcs_prime.mcs_prime_config import status_dict
from mcs_prime import McsTracks, PATHS

interactive = len(sys.argv) == 2 and sys.argv[1] == 'interactive'

tracks = McsTracks.load(PATHS['statsdir'] / 'mcs_tracks_final_extc_20190101.0000_20200101.0000.nc', PATHS['pixeldir'])
statuses = tracks.dstracks.track_status.values.flatten()[~np.isnan(tracks.dstracks.track_status.values.flatten())]
status_counts = Counter(statuses)

fig, ax = plt.subplots()
ax.bar(range(len(status_dict)), [status_counts[k] for k in status_dict.keys()])
ax.set_xticks(range(len(status_dict)), [f'{k}: {v}' for k, v in list(status_dict.items())], rotation=90)
ax.set_ylabel(f'# track points (total={len(statuses)})')
secax = ax.secondary_yaxis('right', functions=(lambda x: 100 * x / len(statuses), lambda x: x / 100 * len(statuses)))
secax.set_ylabel('% of total')
ax.axhline(y=5 / 100 * len(statuses))
fig.subplots_adjust(bottom=0.7, top=0.99)

print(set(status_dict.keys()) - set(statuses.astype(int)))
print(set(statuses.astype(int)) - set(status_dict.keys()))

if interactive:
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.show()
else:
    figpath = PATHS['figdir'] / 'status_counts' / f'status_counts_mcs_tracks_final_extc_20190101.0000_20200101.0000.png'
    figpath.parent.mkdir(exist_ok=True, parents=True)
    fig = plt.gcf()
    fig.set_size_inches(20, 11.26)  # full screen
    plt.savefig(figpath, bbox_inches='tight', dpi=100)
