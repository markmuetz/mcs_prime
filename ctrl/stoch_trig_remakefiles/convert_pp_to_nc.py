from pathlib import Path
import shutil

import iris

from remake import Remake, TaskRule

import mcs_prime.mcs_prime_config_util as cu

DATADIR = cu.PATHS['datadir']

slurm_config = {'account': 'short4hr', 'queue': 'short-serial-4hr', 'mem': 64000}
rmk = Remake(config=dict(slurm=slurm_config, content_checks=False))

# SUITES = ['u-cv415']
# SUITES = ['u-dg040', 'u-dg041', 'u-dg042']
# SUITES = ['u-dg041', 'u-dg042']
#
# PP_DIRS = [
#     DATADIR / 'UM_sims' / suite
#     for suite in SUITES
# ]
PP_DIRS = []
# /gws/nopw/j04/mcs_prime/mmuetz/data/UM_sims/u-dg135/share/cycle/20200701T0000Z/engl/um/em0
PP_DIRS.extend([DATADIR / f'UM_sims/u-dg135/share/cycle/20200701T0000Z/engl/um/em{i}' for i in range(10)])
PP_DIRS.extend([DATADIR / f'UM_sims/u-di728/share/cycle/20200701T0000Z/engl/um/em{i}' for i in range(10)])
PP_DIRS.extend([DATADIR / f'UM_sims/u-di727/share/cycle/20200701T0000Z/engl/um/em{i}' for i in range(10)])
# PP_DIRS.extend([DATADIR / f'UM_sims/zhixiao_mirror/ens_um/pa_um_off'])
# PP_DIRS.extend([DATADIR / f'UM_sims/zhixiao_mirror/ens_um_corrected/pa_um_on'])


def pp_to_converted(pp_path, converter):
    return pp_path.parent / (pp_path.stem + f'.{converter}.nc')


def ls_pp_paths(pp_dir, converter):
    pp_paths = sorted(pp_dir.glob(f'*.pp'))
    ff_paths = sorted([ff for ff in pp_dir.glob('*') if ff.suffix == ''])
    # print(ff_paths)
    pp_paths.extend(ff_paths)
    out_paths = [pp_to_converted(p, converter) for p in pp_paths]
    return pp_paths, out_paths, [o.exists() for o in out_paths]


def get_paths_for_converter(pp_dirs, converter):
    paths = []
    for pp_dir in pp_dirs:
        pp_paths, out_paths, out_paths_exist = ls_pp_paths(pp_dir, converter)
        paths.extend([
            (pp_path, out_path)
            for pp_path, out_path in zip(pp_paths, out_paths)
        ])
    return paths


IRIS_PATHS = get_paths_for_converter(PP_DIRS, 'iris')
CF_PATHS = get_paths_for_converter(PP_DIRS, 'cf')


def iris_to_netcdf_tmp_then_copy(cubes, outpath):
    tmpdir = Path('/work/scratch-nopw2/mmuetz')
    assert outpath.is_absolute()
    tmppath = tmpdir / Path(*outpath.parts[1:])
    tmppath.parent.mkdir(exist_ok=True, parents=True)
    iris.save(cubes, tmppath)
    shutil.move(tmppath, outpath)


class IrisConvertPP2NC(TaskRule):
    """
    """
    rule_inputs = {'pp_path': '{pp_path}'}
    rule_outputs = {'out_path': '{out_path}'}

    var_matrix = {
        ('pp_path', 'out_path'): IRIS_PATHS
    }

    def rule_run(self):
        print(f'{self.pp_path} -> {self.out_path}')

        pp_path = self.inputs['pp_path']
        out_path = self.outputs['out_path']

        cubes = iris.load(pp_path)
        iris_to_netcdf_tmp_then_copy(cubes, out_path)
