import iris

from remake import Remake, TaskRule

import mcs_prime.mcs_prime_config_util as cu

DATADIR = cu.PATHS['datadir']

slurm_config = {'account': 'short4hr', 'queue': 'short-serial-4hr', 'mem': 64000}
rmk = Remake(config=dict(slurm=slurm_config, content_checks=False))

SUITES = ['u-dg040', 'u-dg041', 'u-dg042']

def pp_to_converted(pp_path, converter):
    return pp_path.parent / (pp_path.stem + f'.{converter}.nc')


def ls_suite_pp_paths(suite, converter):
    suitedir = DATADIR / 'UM_sims' / suite
    short_suite = suite[2:]
    pp_paths = sorted(suitedir.glob(f'{short_suite}a.p*.pp'))
    out_paths = [pp_to_converted(p, converter) for p in pp_paths]
    return pp_paths, out_paths, [o.exists() for o in out_paths]

def get_paths_for_converter(suites, converter):
    paths = []
    for suite in suites:
        pp_paths, out_paths, out_paths_exist = ls_suite_pp_paths(suite, converter)
        paths.extend([
            (pp_path, out_path)
            for pp_path, out_path in zip(pp_paths, out_paths)
        ])
    return paths

IRIS_PATHS = get_paths_for_converter(SUITES, 'iris')
CF_PATHS = get_paths_for_converter(SUITES, 'cf')


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
        iris.save(cubes, out_path)
