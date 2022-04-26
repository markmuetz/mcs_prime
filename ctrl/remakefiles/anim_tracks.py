import datetime as dt

import xarray as xr
from remake import Remake, TaskRule
from remake.util import sysrun

from mcs_prime import PATHS, McsTracks, McsTrack, PixelData
from mcs_prime.util import round_times_to_nearest_second

anim_tracks = Remake()


def track_ids_at_time(time):
    stats_year_path = PATHS['statsdir'] / 'mcs_tracks_final_extc_20190101.0000_20200101.0000.nc'

    dstracks = xr.open_dataset(stats_year_path)
    round_times_to_nearest_second(dstracks)
    tracks = McsTracks(dstracks)
    selected_tracks = tracks.tracks_at_time(time)

    return selected_tracks.dstracks.tracks.values


class AnimTracks20190621_0630(TaskRule):
    rule_inputs = {}
    rule_outputs = {'track_{track_id}': PATHS['figdir'] / 'anim_tracks' / '20190621' / 'track_{track_id}.gif'}

    var_matrix = {'track_id': track_ids_at_time(dt.datetime(2019, 6, 21, 6, 30))}

    depends_on = [McsTracks, McsTrack, PixelData]

    def rule_run(self):
        stats_year_path = PATHS['statsdir'] / 'mcs_tracks_final_extc_20190101.0000_20200101.0000.nc'

        dstracks = xr.open_dataset(stats_year_path)
        round_times_to_nearest_second(dstracks)

        pixel_data = PixelData(PATHS['pixeldir'])
        tracks = McsTracks(dstracks, pixel_data)

        track_id = int(list(self.outputs.keys())[0].split('_')[1])
        track = tracks.get_track(track_id)

        output = self.outputs[f'track_{track_id}']
        figpaths = track.animate(savefigs=True, figdir=output.parent)
        path = figpaths[0]
        # path.stem is 'track_<track_id>_<track_timestep>'.
        stem_glob = '_'.join(path.stem.split('_')[:2])
        path_glob = path.parent / f'{stem_glob}_*.png'

        sysrun(f'convert -delay 20 -loop 5 {path_glob} {output}')
