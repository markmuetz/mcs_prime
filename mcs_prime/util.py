import numpy as np


def round_times_to_nearest_second(dstracks):
    """Round times in dstracks.base_time to the nearest second.

    Sometimes the dstracks dataset has minor inaccuracies in the time, e.g.
    '2000-06-01T00:30:00.000013440' (13440 ns). Remove these.

    :param dstracks: xarray.Dataset to convert.
    :return: None
    """

    # N.B. I tried to do this using pure np funcions, but could not work
    # out how to convert np.int64 into a np.datetime64. Seems like it should be
    # easy.

    def remove_time_incaccuracy(t):
        return np.datetime64(int(round(t / 1e9) * 1e9), 'ns')

    # vec_remove_time_incaccuracy = np.vectorize(remove_time_incaccuracy)
    def vec_remove_time_incaccuracy(times):
        return [remove_time_incaccuracy(t) for t in times]
    tmask = ~np.isnan(dstracks.base_time.values)
    dstracks.base_time.values[tmask] = vec_remove_time_incaccuracy(dstracks.base_time.values[tmask].astype(int))
    tmask = ~np.isnan(dstracks.start_basetime.values)
    dstracks.start_basetime.values[tmask] = vec_remove_time_incaccuracy(
        dstracks.start_basetime.values[tmask].astype(int)
    )
    tmask = ~np.isnan(dstracks.end_basetime.values)
    dstracks.end_basetime.values[tmask] = vec_remove_time_incaccuracy(dstracks.end_basetime.values[tmask].astype(int))
