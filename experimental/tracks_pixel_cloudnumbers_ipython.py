# coding: utf-8
from mcs_prime import McsTracks

tracks = McsTracks.load(PATHS['statsdir'] / 'mcs_tracks_final_extc_20190101.0000_20200101.0000.nc', PATHS['pixeldir'])
from mcs_prime import PATHS

tracks = McsTracks.load(PATHS['statsdir'] / 'mcs_tracks_final_extc_20190101.0000_20200101.0000.nc', PATHS['pixeldir'])
tracks.dstracks
tracks.pixel_data
tracks.pixel_data.get_frame(dt.datetime(2019, 6, 21, 0, 30))
import datetime as dt

tracks.pixel_data.get_frame(dt.datetime(2019, 6, 21, 0, 30))
f = tracks.pixel_data.get_frame(dt.datetime(2019, 6, 21, 0, 30))
f.dspixel.cloudnumber
f.dspixel.cloudnumber.values
f.dspixel.cloudnumber.values.unique()
np.unique(f.dspixel.cloudnumber.values)
import numpy as np

np.unique(f.dspixel.cloudnumber.values)
np.unique(f.dspixel.cloudnumber.values).astype(int)
np.unique(f.dspixel.cloudnumber.values).astype(int)[:5]
np.unique(f.dspixel.cloudnumber.values).astype(int)[5:]
np.unique(f.dspixel.cloudnumber.values).astype(int)[-5:]
np.unique(f.dspixel.cloudnumber.values).astype(int)[:-1]
import pandas as pd

time = pd.Timestamp(dt.datetime(2019, 6, 21, 0, 30)).to_numpy()
time
tracks.dstracks.base_time == time
(tracks.dstracks.base_time == time).sum()
tracks.dstracks.cloudnumber[(tracks.dstracks.base_time == time)]
tracks.dstracks.cloudnumber.values[(tracks.dstracks.base_time == time)]
f
track_cns = tracks.pixel_data.get_frame(dt.datetime(2019, 6, 21, 0, 30))
track_cns
track_cns = tracks.dstracks.cloudnumber[(tracks.dstracks.base_time == time)]
track_cns = tracks.dstracks.cloudnumber.values[(tracks.dstracks.base_time == time)]
f.dspixel.cloudnumber.values.unique()
frame_unique_cns = np.unique(f.dspixel.cloudnumber.values).astype(int)[:-1]
frame_cns = f.dspixel.cloudnumber.values[0].astype(int)
frame_cns
frame_cns[np.isnan(f.dspixel.cloudnumber.values[0])] = 0
frame_cns
frame_cns.isin
np.isin(frame_cns, track_cns)
plt.imshow(np.isin(frame_cns, track_cns), origin='lower')
import matplotlib.pyplot as plt

plt.imshow(np.isin(frame_cns, track_cns), origin='lower')
plt.figure()
plt.imshow(~np.isin(frame_cns, track_cns), origin='lower')
plt.show()
plt.imshow(np.isin(frame_cns, track_cns), origin='lower')
plt.figure()
plt.imshow(frame_cns > 0, origin='lower')
plt.show()
get_ipython().run_line_magic('history', '~1/')
get_ipython().run_line_magic(
    'save', '/home/markmuetz/projects/mcs_prime/experimental/corr_and_land_mask_ipython.py ~1/'
)
get_ipython().run_line_magic('history', '')
