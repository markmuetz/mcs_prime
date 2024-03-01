# MCS PRIME

## Mesoscale Convective Systems: PRobabilistic forecasting and upscale IMpacts in the grey zonE (MCS:PRIME)

Remakefiles used for all analysis by M. Muetzelfeldt.

[Remakefiles](https://github.com/markmuetz/remake) form the heart of the analysis code.

Designed to be used with the `mcs_prime_env` environment.
Will only work on JASMIN, using SLURM.

## Data:

### ERA5

Makes use of the ERA5 model level data stored under: `/badc/ecmwf-era5/data/oper`
CIN is not included there, can be downloaded using: `era5_download.py`

### MCS tracking dataset

Uses Feng et al. 2021 tracking dataset, v2 (global version).
Located on JASMIN under MCS:PRIME GWS: `/gws/nopw/j04/mcs_prime/mmuetz/data/MCS_Global`

## Processing:

`era5_process.py`

Process the ERA5 data.

* Calc shear
* Calc Vertically Integrated Moisture Flux Divergence (turned into MFC in paper)
* Calc layer means of RH, theta e
* Calc monthly means

## Analysis:

### MCS local envs:

`mcs_local_envs.py`

Perform local env analysis. This either takes the form of saving the mean at different radii from the MCS centroid, or storing the 2D field at different radii (composite near to MCS init).

### ERA5 histograms

`era5_histograms.py`

ERA5 histograms AND conditional histograms based on 5 MCS regions: MCS core, MCS shield, non-MCS core, non-MCS shield and environment. 

## Plotting:

`mcs_env_cond_figs.py`

Produce all figures for use in paper.

