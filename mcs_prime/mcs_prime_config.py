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
        'datadir': Path('/tbd'),
        'statsdir': Path('/tbd'),
        'pixeldir': Path('/tbd'),
        'outdir': Path('/tbd'),
        'figdir': Path('/tbd'),
    },
}


def _short_hostname():
    hostname = socket.gethostname()
    if '.' in hostname and hostname.split('.')[1] == 'jasmin':
        return 'jasmin'
    return hostname


hostname = _short_hostname()
if hostname[:4] == 'host':
    hostname = 'jasmin'

if hostname not in ALL_PATHS:
    raise Exception(f'Unknown hostname: {hostname}')

PATHS = ALL_PATHS[hostname]
for k, path in PATHS.items():
    if not path.exists():
        warnings.warn(f'Warning: path missing {k}: {path}')
