#!/usr/bin/env python
import cdsapi
c = cdsapi.Client()
 
# data download specifications:
cls     = "ea"         # do not change
expver  = "1"          # do not change
levtype = "ml"         # do not change
stream  = "oper"       # do not change
date    = "2018-01-01" # date: Specify a single date as "2018-01-01" or a period as "2018-08-01/to/2018-01-31". For periods > 1 month see https://confluence.ecmwf.int/x/l7GqB
tp      = "an"         # type: Use "an" (analysis) unless you have a particular reason to use "fc" (forecast).
time    = "00:00:00"   # time: ERA5 data is hourly. Specify a single time as "00:00:00", or a range as "00:00:00/01:00:00/02:00:00" or "00:00:00/to/23:00:00/by/1".
 
c.retrieve('reanalysis-era5-complete', {
    'class'   : cls,
    'date'    : date,
    'expver'  : expver,
    'levelist': '1/2/3/4/5/6/7/8/9/10/11/12/13/14/15/16/17/18/19/20/21/22/23/24/25/26/27/28/29/30/31/32/33/34/35/36/37/38/39/40/41/42/43/44/45/46/47/48/49/50/51/52/53/54/55/56/57/58/59/60/61/62/63/64/65/66/67/68/69/70/71/72/73/74/75/76/77/78/79/80/81/82/83/84/85/86/87/88/89/90/91/92/93/94/95/96/97/98/99/100/101/102/103/104/105/106/107/108/109/110/111/112/113/114/115/116/117/118/119/120/121/122/123/124/125/126/127/128/129/130/131/132/133/134/135/136/137',         # For each of the 137 model levels
    'levtype' : 'ml',
    'param'   : '130/133', # Temperature (t) and specific humidity (q)
    'stream'  : stream,
    'time'    : time,
    'type'    : tp,
    'grid'    : [1.0, 1.0], # Latitude/longitude grid: east-west (longitude) and north-south resolution (latitude). Default: 0.25 x 0.25
    #'area'    : area, #example: [60, -10, 50, 2], # North, West, South, East. Default: global
}, 'tq_ml.grib')
 
 
c.retrieve('reanalysis-era5-complete', {
    'class'   : cls,
    'date'    : date,
    'expver'  : expver,
    'levelist': '1',       # Geopotential (z) and Logarithm of surface pressure (lnsp) are 2D fields, archived as model level 1
    'levtype' : levtype,
    'param'   : '129/152', # Geopotential (z) and Logarithm of surface pressure (lnsp)
    'stream'  : stream,
    'time'    : time,
    'type'    : tp,
    'grid'    : [1.0, 1.0], # Latitude/longitude grid: east-west (longitude) and north-south resolution (latitude). Default: 0.25 x 0.25
    #'area'    : area, #example: [60, -10, 50, 2], # North, West, South, East. Default: global
}, 'zlnsp_ml.grib')