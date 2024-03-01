import datetime as dt
import math
import pickle
from pathlib import Path
from timeit import default_timer

import numpy as np
import pandas as pd
import xarray as xr
from cartopy.util import add_cyclic_point
from mcs_prime import PATHS, McsTracks, PixelData
from mcs_prime.util import round_times_to_nearest_second
from remake import Remake, TaskRule
from remake.util import format_path as fmtp

slurm_config = {'account': 'short4hr', 'queue': 'short-serial-4hr', 'mem': 16000}
# slurm_config = {'queue': 'short-serial', 'mem': 16000}
track_era5_prop = Remake(config=dict(slurm=slurm_config))


NTRACKS_PER_BATCH = 1000

# What's going on here?
# Well, I need to know how many tracks are in each year, so I can work out how many batches
# I need to use each year. This has to be done before any tasks are created, so do on module load
# and cache results. If I need to add more years I can just delete and recalc.
batch_info_path = PATHS['outdir'] / 'track_era5_env_precursor' / f'year_batch_info_{NTRACKS_PER_BATCH}.pkl'
if not batch_info_path.exists():
    batch_info = []
    for year in range(2019, 2021):
        stats_year_path = list(PATHS['statsdir'].glob(f'mcs_tracks_final_extc_{year}*.0000.nc'))[0]
        tracks = McsTracks.open(stats_year_path, PATHS["pixeldir"], round_times=False)
        nbatch = math.ceil(len(tracks.dstracks.tracks) / NTRACKS_PER_BATCH)
        print(year, len(tracks.dstracks.tracks), nbatch)
        batch_info.extend([(year, b) for b in range(nbatch)])
    with batch_info_path.open('wb') as fp:
        pickle.dump(batch_info, fp)
else:
    with batch_info_path.open('rb') as fp:
        batch_info = pickle.load(fp)

# batch_info = [(y, b) for y, b in batch_info if y == 2019]
years = sorted(set([y for y, b in batch_info]))
nbatch_per_year = {year: len([(y, b) for y, b in batch_info if y == year]) for year in years}


def xr_add_cyclic_point(da, lon_name='longitude'):
    """Pad data array in longitude dimension.

    * Taken from https://stackoverflow.com/a/60913814/54557
    * Modified to handle data on model levels as well.
    * Use add_cyclic_point to pad input data.
    * Relies on min lon value in da being 0.

    :param da: xr.DataArray with dimensions including longitude
    :param lon_name: name of longitude dimension in da
    :returns: padded copy of da
    """
    lon_idx = da.dims.index(lon_name)
    wrap_data, wrap_lon = add_cyclic_point(da.values, coord=da.longitude, axis=lon_idx)

    # Copy old coords and modify longitude.
    new_coords = {dim: da.coords[dim] for dim in da.dims}
    new_coords[lon_name] = wrap_lon

    # Generate output DataArray with new data but same structure as input.
    out_da = xr.DataArray(data=wrap_data, coords=new_coords, dims=da.dims, attrs=da.attrs)
    return out_da


class TrackERA5EnvPrecursor(TaskRule):
    rule_inputs = {}
    rule_outputs = {
        'daily_track_era5_data': (
            PATHS['outdir']
            / 'track_era5_env_precursor'
            / '{year}'
            / 'daily_track_era5_data_{year}_batch-{batch:02d}.nc'
        )
    }

    var_matrix = {('year', 'batch'): batch_info}

    def rule_run(self):
        # Load the tracks data.
        stats_year_path = list(PATHS['statsdir'].glob(f'mcs_tracks_final_extc_{self.year}*.0000.nc'))[0]
        tracks = McsTracks.open(stats_year_path, PATHS["pixeldir"])
        start_track = self.batch * NTRACKS_PER_BATCH
        end_track = (self.batch + 1) * NTRACKS_PER_BATCH
        tracks = McsTracks(tracks.dstracks.isel(tracks=slice(start_track, end_track)), PATHS['pixeldir'])

        start_time = pd.Timestamp(tracks.dstracks.start_basetime.values.min()).to_pydatetime()
        end_time = pd.Timestamp(tracks.dstracks.end_basetime.values.max()).to_pydatetime()
        print(start_time, end_time)

        start = default_timer()
        # Force load of some data.
        tracks.dstracks.base_time.load()
        tracks.dstracks.meanlon.load()
        tracks.dstracks.meanlat.load()

        variables = ['cape', 'tcwv']

        # Create an output dataset that is similar to the tracks dataset for the vars I am getting.
        var_data = {}
        for var in variables:
            data = tracks.dstracks.meanlat.copy().values
            data[~np.isnan(data)] = np.nan
            var_data[var] = (('tracks', 'times'), data)
        dsout = xr.Dataset(
            coords=dict(tracks=tracks.dstracks.tracks, times=tracks.dstracks.times),
            data_vars=var_data,
        )

        num_filtered_points = 0

        track_time = start_time
        while track_time <= end_time:
            print(track_time)
            # track data every hour on the half hour.
            # ERA5 data every hour on the hour.
            e5time1 = track_time - dt.timedelta(minutes=30)
            e5time2 = e5time1 + dt.timedelta(hours=1)

            # Cannot interp track data - get ERA5 before and after and interp
            # using e.g. ...mean(dim=time).
            paths = [
                (
                    PATHS['era5dir']
                    / f'data/oper/an_sfc/{t.year}/{t.month:02d}/{t.day:02d}'
                    / (f'ecmwf-era5_oper_an_sfc_{t.year}{t.month:02d}{t.day:02d}' f'{t.hour:02d}00.{var}.nc')
                )
                for var in variables
                for t in [e5time1, e5time2]
            ]
            # Lat limited to that of tracks, and midpoint time value.
            e5 = xr.open_mfdataset(paths).sel(latitude=slice(60, -60)).mean(dim='time').load()

            track_point_mask = tracks.dstracks.base_time == pd.Timestamp(track_time)
            track_point_lon = tracks.dstracks.meanlon.values[track_point_mask]
            track_point_lat = tracks.dstracks.meanlat.values[track_point_mask]

            # Convert -180-180 -> 0-360.
            track_point_lon = track_point_lon % 360

            # This is the same way you select values along a transect.
            # But here I just want at unconnected points.
            lon = xr.DataArray(track_point_lon, dims='track_point')
            lat = xr.DataArray(track_point_lat, dims='track_point')

            if len(lon):
                var_data = {}
                for var in variables:
                    # if lon > 359.75, cannot interp and value is nan.
                    # data = e5[var].interp(longitude=lon, latitude=lat).values
                    # Fix by padding data.
                    e5_padded = xr_add_cyclic_point(e5[var])
                    data = e5_padded.interp(longitude=lon, latitude=lat).values
                    mask = np.isnan(data)
                    assert mask.sum() == 0
                    # Note, mask works on this because it has the same shape.
                    dsout[var].values[track_point_mask] = data

            e5.close()
            track_time += dt.timedelta(hours=1)

        # TODO: Set to show how created.
        # dsout.attrs = dict(
        # )
        end = default_timer()
        comp = dict(zlib=True, complevel=4)
        encoding = {var: comp for var in dsout.data_vars}
        dsout.to_netcdf(self.outputs['daily_track_era5_data'], encoding=encoding)
        print(end - start)


class TrackERA5EnvPrecursorShear(TaskRule):
    rule_inputs = {}
    rule_outputs = {
        'daily_track_era5_data': (
            PATHS['outdir']
            / 'track_era5_env_precursor'
            / '{year}'
            / 'daily_track_era5_data_shear_{year}_batch-{batch:02d}.nc'
        )
    }

    var_matrix = {('year', 'batch'): batch_info}

    def rule_run(self):
        # Load the tracks data.
        stats_year_path = list(PATHS['statsdir'].glob(f'mcs_tracks_final_extc_{self.year}*.0000.nc'))[0]
        tracks = McsTracks.open(stats_year_path, PATHS["pixeldir"])
        start_track = self.batch * NTRACKS_PER_BATCH
        end_track = (self.batch + 1) * NTRACKS_PER_BATCH
        tracks = McsTracks(tracks.dstracks.isel(tracks=slice(start_track, end_track)), PATHS['pixeldir'])

        start_time = pd.Timestamp(tracks.dstracks.start_basetime.values.min()).to_pydatetime()
        end_time = pd.Timestamp(tracks.dstracks.end_basetime.values.max()).to_pydatetime()
        print(start_time, end_time)

        start = default_timer()
        # Force load of some data.
        tracks.dstracks.base_time.load()
        tracks.dstracks.meanlon.load()
        tracks.dstracks.meanlat.load()

        variables = ['u', 'v']
        # Pick out 4 model levels to work out low/mid/deep shear.
        # Levels defined here:
        # https://confluence.ecmwf.int/display/UDOC/L137+model+level+definitions
        # pressure/height shown for surface=1013.25hPa.
        levels = [
            133,  # 1000hPa/107m
            111,  # 804hPa/1911m
            96,  # 508hPa/5469m
            74,  # 197hPa/11890m
        ]

        # Create an output dataset that is similar to the tracks dataset for the vars I am getting.
        var_data = {}
        for var in variables:
            data = np.full([len(tracks.dstracks.tracks), len(tracks.dstracks.times), len(levels)], np.nan)
            var_data[var] = (('tracks', 'times', 'level'), data)
        dsout = xr.Dataset(
            coords=dict(tracks=tracks.dstracks.tracks, times=tracks.dstracks.times, level=levels),
            data_vars=var_data,
        )

        num_filtered_points = 0

        track_time = start_time
        while track_time <= end_time:
            print(track_time)
            # track data every hour on the half hour.
            # ERA5 data every hour on the hour.
            e5time1 = track_time - dt.timedelta(minutes=30)
            e5time2 = e5time1 + dt.timedelta(hours=1)

            # Cannot interp track data - get ERA5 before and after and interp
            # using e.g. ...mean(dim=time).
            paths = [
                (
                    PATHS['era5dir']
                    / f'data/oper/an_ml/{t.year}/{t.month:02d}/{t.day:02d}'
                    / (f'ecmwf-era5_oper_an_ml_{t.year}{t.month:02d}{t.day:02d}' f'{t.hour:02d}00.{var}.nc')
                )
                for var in variables
                for t in [e5time1, e5time2]
            ]
            # Lat limited to that of tracks, and midpoint time value.
            e5 = xr.open_mfdataset(paths).sel(latitude=slice(60, -60), level=levels).mean(dim='time').load()

            track_point_mask = tracks.dstracks.base_time == pd.Timestamp(track_time)
            track_point_lon = tracks.dstracks.meanlon.values[track_point_mask]
            track_point_lat = tracks.dstracks.meanlat.values[track_point_mask]

            # Convert -180-180 -> 0-360.
            track_point_lon = track_point_lon % 360

            # This is the same way you select values along a transect.
            # But here I just want at unconnected points.
            lon = xr.DataArray(track_point_lon, dims='track_point')
            lat = xr.DataArray(track_point_lat, dims='track_point')

            if len(lon):
                var_data = {}
                for var in variables:
                    # if lon > 359.75, cannot interp and value is nan.
                    # data = e5[var].interp(longitude=lon, latitude=lat).values
                    # Fix by padding data.
                    e5_padded = xr_add_cyclic_point(e5[var])
                    data = e5_padded.interp(longitude=lon, latitude=lat).values
                    mask = np.isnan(data)
                    assert mask.sum() == 0
                    # Note, mask works on dsout[var] because it has the same shape as dstracks.
                    # data comes from ERA5, and has coords (level, npoints),
                    # dsout has coords which are the opposite way round, because mask selects npoint's
                    # worth of data and the final coord is levels.
                    # hence use of data.T.
                    dsout[var].values[track_point_mask] = data.T

            e5.close()
            track_time += dt.timedelta(hours=1)

        # TODO: Set to show how created.
        # dsout.attrs = dict(
        # )
        end = default_timer()
        comp = dict(zlib=True, complevel=4)
        encoding = {var: comp for var in dsout.data_vars}
        dsout.to_netcdf(self.outputs['daily_track_era5_data'], encoding=encoding)
        print(end - start)


class CombineTrackERA5EnvPrecursor(TaskRule):
    @staticmethod
    def rule_inputs(year, datatype):
        if datatype == '2D':
            inputs = {
                f'batch-{b:02d}': fmtp(TrackERA5EnvPrecursor.rule_outputs['daily_track_era5_data'], year=year, batch=b)
                for b in range(nbatch_per_year[year])
            }
        elif datatype == 'shear':
            inputs = {
                f'shear-batch-{b:02d}': fmtp(
                    TrackERA5EnvPrecursorShear.rule_outputs['daily_track_era5_data'], year=year, batch=b
                )
                for b in range(nbatch_per_year[year])
            }
        return inputs

    def rule_outputs(year, datatype):
        if datatype == '2D':
            outputs = {
                'track_era5_data': (
                    PATHS['outdir'] / 'track_era5_env_precursor' / f'{year}' / f'daily_track_era5_data_{year}.nc'
                )
            }
        elif datatype == 'shear':
            outputs = {
                'track_era5_data': (
                    PATHS['outdir'] / 'track_era5_env_precursor' / f'{year}' / f'daily_track_era5_data_shear_{year}.nc'
                )
            }
        return outputs

    var_matrix = {'year': years, 'datatype': ['2D', 'shear']}

    def rule_run(self):
        input_paths = self.inputs.values()
        ds = xr.open_mfdataset(
            input_paths,
            concat_dim='tracks',
            combine='nested',
            mask_and_scale=False,
        )
        comp = dict(zlib=True, complevel=4)
        encoding = {var: comp for var in ds.data_vars}
        ds.to_netcdf(self.outputs['track_era5_data'], encoding=encoding)
