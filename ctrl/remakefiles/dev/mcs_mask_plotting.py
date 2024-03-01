import matplotlib.pyplot as plt
import mcs_prime.mcs_prime_config_util as cu
from mcs_prime import PATHS, mcs_mask_plotter
import pandas as pd
from remake import Remake, TaskRule
from remake.util import format_path as fmtp


slurm_config = {'account': 'short4hr', 'queue': 'short-serial-4hr', 'mem': 64000}
mcs_mask_plotting = Remake(config=dict(slurm=slurm_config, content_checks=False))

settings_list = [
    {
        'name': 'tcwv1',
        'time': pd.Timestamp(2020, 1, 1, 6, 30),
        'plot_kwargs': dict(var='tcwv', extent=(-25, -5, -2.5, 7.5), grid_x=[-20, -15, -10], grid_y=[-2, 0, 2, 4, 6], cbar_kwargs=dict(orientation='horizontal')),
    },
    {
        'name': 'cape1',
        'time': pd.Timestamp(2020, 1, 1, 6, 30),
        'plot_kwargs': dict(var='cape', extent=(-25, -5, -2.5, 7.5), grid_x=[-20, -15, -10], grid_y=[-2, 0, 2, 4, 6], cbar_kwargs=dict(orientation='horizontal')),
    },
]
settings_dict = dict([(s['name'], s) for s in settings_list])


class PlotMcsMasks(TaskRule):
    @staticmethod
    def rule_inputs(settings_name):
        settings = settings_dict[settings_name]
        time = settings['time']
        pdata = mcs_mask_plotter.McsMaskPlotterData(pd.DatetimeIndex([time]), [settings['plot_kwargs']['var']])
        inputs = {
            **pdata.tracks_inputs,
            **pdata.e5inputs,
            **pdata.pixel_on_e5_inputs,
        }

        return inputs

    @staticmethod
    def rule_outputs(settings_name):
        settings = settings_dict[settings_name]
        filename = f'{settings["time"]}_{settings_name}.pdf'
        return {f'fig_{settings_name}': PATHS['figdir'] / 'mcs_mask_plotting' / filename}

    depends_on = [
        mcs_mask_plotter.McsMaskPlotterData,
        mcs_mask_plotter.McsMaskPlotter,
    ]

    var_matrix = {
        'settings_name': list(settings_dict.keys()),
    }

    def rule_run(self):

        settings = settings_dict[self.settings_name]
        time = settings['time']
        print(settings)

        pdata = mcs_mask_plotter.McsMaskPlotterData(pd.DatetimeIndex([time]), [settings['plot_kwargs']['var']])
        pdata.load()

        plotter = mcs_mask_plotter.McsMaskPlotter(pdata)
        plotter.plot(None, time, **settings['plot_kwargs'])

        plt.savefig(self.outputs[f'fig_{self.settings_name}'])
