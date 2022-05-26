from pathlib import Path
import socket
import warnings

ALL_PATHS = {
    'mistakenot': {
        'datadir': Path('/home/markmuetz/Datasets/MCS_PRIME'),
        'statsdir': Path('/home/markmuetz/Datasets/MCS_PRIME/MCS_database/MCS_Global/stats'),
        'pixeldir': Path('/home/markmuetz/Datasets/MCS_PRIME/MCS_database/MCS_Global/mcstracking'),
        'outdir': Path('/home/markmuetz/MCS_PRIME_output/output'),
        'figdir': Path('/home/markmuetz/MCS_PRIME_output/figs'),
    },
    'jasmin': {
        'datadir': Path('/gws/nopw/j04/cosmic/mmuetz/data/MCS_PRIME'),
        'statsdir': Path('/gws/nopw/j04/cosmic/mmuetz/data/MCS_PRIME/MCS_Global/stats'),
        'pixeldir': Path('/gws/nopw/j04/cosmic/mmuetz/data/MCS_PRIME/MCS_Global/mcstracking'),
        'outdir': Path('/gws/nopw/j04/cosmic/mmuetz/data/MCS_PRIME/output'),
        'figdir': Path('/gws/nopw/j04/cosmic/mmuetz/data/MCS_PRIME/figs'),
        'era5dir': Path('/badc/ecmwf-era5'),
    },
}


def _short_hostname():
    hostname = socket.gethostname()
    if '.' in hostname and hostname.split('.')[1] == 'jasmin':
        return 'jasmin'
    return hostname


hostname = _short_hostname()
if hostname[:4] == 'host' or hostname == 'jupyter-mmuetz':
    hostname = 'jasmin'

if hostname not in ALL_PATHS:
    raise Exception(f'Unknown hostname: {hostname}')

PATHS = ALL_PATHS[hostname]
for k, path in PATHS.items():
    if not path.exists():
        warnings.warn(f'Warning: path missing {k}: {path}')

# Generated using:
# dict((int(v[0]), v[1].strip()) for v in [l.strip().split(':') for l in tracks.dstracks.track_status.attrs['comments'].split(';')]))
status_dict = {
    0: 'Track stops',
    1: 'Simple track continuation',
    2: 'This is the bigger cloud in simple merger',
    3: 'This is the bigger cloud from a simple split that stops at this time',
    4: 'This is the bigger cloud from a split and this cloud continues to the next time',
    5: 'This is the bigger cloud from a split that subsequently is the big cloud in a merger',
    13: 'This cloud splits at the next time step',
    15: 'This cloud is the bigger cloud in a merge that then splits at the next time step',
    16: 'This is the bigger cloud in a split that then splits at the next time step',
    18: 'Merge-split at same time (big merge, splitter, and big split)',
    21: 'This is the smaller cloud in a simple merger',
    24: 'This is the bigger cloud of a split that is then the small cloud in a merger',
    31: 'This is the smaller cloud in a simple split that stops',
    32: 'This is a small split that continues onto the next time step',
    33: 'This is a small split that then is the bigger cloud in a merger',
    34: 'This is the small cloud in a merger that then splits at the next time step',
    37: 'Merge-split at same time (small merge, splitter, big split)',
    44: 'This is the smaller cloud in a split that is smaller cloud in a merger at the next time step',
    46: 'Merge-split at same time (big merge, splitter, small split)',
    52: 'This is the smaller cloud in a split that is smaller cloud in a merger at the next time step',
    65: 'Merge-split at same time (smaller merge, splitter, small split)'
}
