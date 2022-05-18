# coding: utf-8
get_ipython().run_line_magic('run', 'load_2019.py')
tracks
get_ipython().run_line_magic('run', 'load_2019.py')
tracks
tracks.dstracks.area
tracks.dstracks.area.shape
tracks.dstracks.area[:, 1:] - tracks.dstracks.area[:, :-1]
tracks.dstracks.area
tracks.dstracks.area.values
tracks.dstracks.area[:, 1:] - tracks.dstracks.area[:, :-1]
tracks.dstracks.area[:, 1:].values - tracks.dstracks.area[:, :-1].values
(tracks.dstracks.area[:, 1:].values - tracks.dstracks.area[:, :-1].values).shape
area_growth = tracks.dstracks.area[:, 1:].values - tracks.dstracks.area[:, :-1].values
area_growth
nanmask = ~np.isnan(area_growth)
import numpy as np
nanmask = ~np.isnan(area_growth)
nanmask
area_growth[nanmask]
area_growth[nanmask].shape
area_growth[nanmask] > 0
(area_growth[nanmask] > 0).sum()
(area_growth[nanmask] <= 0).sum()
import matplotlib.pyplot as plt
plt.hist(area_growth[nanmask])
plt.show()
plt.hist(area_growth[nanmask], bins=100)
plt.show()
plt.hist(area_growth[nanmask], bins=np.linspace(-5e5, 5e5, 101))
plt.show()
plt.hist(area_growth[nanmask], bins=np.linspace(-2e5, 2e5, 101))
plt.show()
from scipy.stats import norm
norm.fit(area_growth[nanmask])
plt.hist(area_growth[nanmask], bins=np.linspace(-2e5, 2e5, 101), density=True)
plt.clf()
get_ipython().run_line_magic('pinfo', 'plt.hist')
bins = np.linspace(-2e5, 2e5, 101)
plt.hist(area_growth[nanmask], bins=bins, density=True)
mean, std = norm.fit(area_growth[nanmask])
y_norm = norm.pdf(bins, mean, std)
plt.plot(bins, y_norm)
plt.show()
mean
std
plt.hist(area_growth[nanmask], bins=bins, density=True)
y_norm = norm.pdf(bins, mean, 10000)
plt.plot(bins, y_norm)
plt.show()
plt.hist(area_growth[nanmask], bins=bins, density=True)
plt.plot(bins, norm.pdf(bins, mean, 20000)))
plt.plot(bins, norm.pdf(bins, mean, 20000))
plt.hist(area_growth[nanmask], bins=bins, density=True)
plt.show()
from scipy.stats import laplace
get_ipython().run_line_magic('pinfo', 'laplace.fit')
laplace.fit(area_growth[nanmask])
laplace_args = laplace.fit(area_growth[nanmask])
plt.hist(area_growth[nanmask], bins=bins, density=True)
plt.plot(bins, laplace.pdf(bins, *laplace_args))
plt.show()
get_ipython().run_line_magic('pinfo', 'norm.fit')
plt.hist(area_growth[nanmask], bins=bins, density=True)
plt.show()
tracks.dstracks
tracks.dstracks.data_vars
tracks.dstracks.total_rain
total_rain_growth = tracks.dstracks.total_rain[:, 1:].values - tracks.dstracks.total_rain[:, :-1].values
plt.hist(total_rain_growth[nanmask])
plt.show()
plt.hist(total_rain_growth[nanmask], bins=np.linspace(-400, 400, 101))
plt.show()
plt.hist(total_rain_growth[nanmask], bins=np.linspace(-4000, 4000, 101))
plt.show()
tracks.dstracks.data_vars
