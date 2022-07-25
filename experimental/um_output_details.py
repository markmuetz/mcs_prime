import sys
from pathlib import Path

import iris


def get_cell_methods(cms):
    if cms:
        return ','.join([f'{m.method}:{",".join(m.coord_names)} - {",".join(m.intervals)}'
                         for m in cms])
    else:
        return None


def cube_details(c):
    try:
        nt = len(c.coord('time').points)
    except:
        nt = None
    try:
        nlat = len(c.coord('latitude').points)
    except:
        nlat = None
    try:
        nlon = len(c.coord('longitude').points)
    except:
        nlon = None
    try:
        nlev = len(c.coord('model_level_number').points)
    except:
        nlev = None
    cms = get_cell_methods(c.cell_methods)
    return [
        c.name(),
        '{}:{}'.format(c.attributes['STASH'].section, c.attributes['STASH'].item),
        nt, nlat, nlon, nlev, cms
    ]


def format_all_cubes(all_cubes):
    output = [['file', 'name', 'stash_code', 'nt', 'nlat', 'nlon', 'nlev', 'cell_methods']]
    for k, cubes in all_cubes.items():
        for cube in cubes:
            output.append([k] + cube_details(cube))
    return output


if __name__ == '__main__':
    paths = [Path(p) for p in sys.argv[1:]]
    all_cubes = {}
    for p in sorted(paths):
        print(p.stem)
        c = iris.load(str(p))
        all_cubes[p.stem] = c
    output = format_all_cubes(all_cubes)
    Path('stash_output.csv').write_text('\n'.join([','.join(str(v) for v in row) for row in output]))

