import datetime as dt

import matplotlib.pyplot as plt
import xarray as xr

from mcs_prime import PATHS, McsTracks, PixelData
from mcs_prime.util import round_times_to_nearest_second


if __name__ == '__main__':
    plt.ion()
    stats_year_path = PATHS['statsdir'] / 'mcs_tracks_final_extc_20190101.0000_20200101.0000.nc'
    try:
        dstracks
    except NameError:
        dstracks = xr.open_dataset(stats_year_path)
        round_times_to_nearest_second(dstracks)

    pixel_data = PixelData(PATHS['pixeldir'])
    tracks = McsTracks(dstracks, pixel_data)

    time = dt.datetime(2019, 6, 21, 6, 30)
    selected_tracks = tracks.tracks_at_time(time)

    # track = tracks.get_track(15378)
    track = tracks.get_track(15477)
    track.animate()
