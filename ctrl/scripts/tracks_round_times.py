from mcs_prime import McsTracks, PATHS
from mcs_prime.util import round_times_to_nearest_second


for year in range(2000, 2020):
    print(year)
    inpath = next(PATHS['statsdir'].glob(f'mcs_tracks_final_extc_{year}*.nc'))
    outpath = PATHS['statsdir'] / (inpath.stem + '_rounded_times' + inpath.suffix)
    print(inpath, outpath)
    tracks = McsTracks.load(inpath, PATHS['pixeldir'])
    tracks.dstracks.to_netcdf(outpath)
