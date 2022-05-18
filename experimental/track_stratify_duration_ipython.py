# coding: utf-8
get_ipython().run_line_magic('run', 'load_2019.py')
tracks
tracks.dstracks.length
tracks.dstracks.track_length
tracks.dstracks.track_duration
tracks.dstracks.track_duration.values
import matplotlib.pyplot as plt
plt.hist(tracks.dstracks.track_duration.values, bins=100)
plt.show()
plt.hist(tracks.dstracks.track_duration.values, bins=100)
plt.show()
plt.hist(tracks.dstracks.track_duration.values, bins=np.linspace(0, 100, 101))
import numpy as np
plt.hist(tracks.dstracks.track_duration.values, bins=np.linspace(0, 100, 101))
plt.show()
from scipy.stats import poisson
poisson_args = poison.fit(tracks.dstracks.track_duration.values)
poisson_args = poisson.fit(tracks.dstracks.track_duration.values)
plt.hist(tracks.dstracks.track_duration.values, bins=np.linspace(0, 100, 101))
plt.show()
plt.hist(tracks.dstracks.track_duration.values / 24, bins=np.linspace(0, 100, 101))
plt.show()
plt.hist(tracks.dstracks.track_duration.values / 24, bins=np.linspace(0, 4, 101))
plt.show()
plt.hist(tracks.dstracks.track_duration.values / 24, bins=np.linspace(0, 100 / 24, 101))
plt.show()
tracks.dstracks.track_duration.values
plt.hist(tracks.dstracks.track_duration.values / 24., bins=np.linspace(0, 100 / 24, 101))
tracks.dstracks.track_duration.values
plt.show()
plt.hist(tracks.dstracks.track_duration.values / 24., bins=np.linspace(0, 100 / 24, 200))
plt.show()
plt.hist(tracks.dstracks.track_duration.values / 24., bins=np.linspace(0, 100 / 24, 80))
plt.show()
tracks.dstracks.track_duration.values / 24
plt.hist(tracks.dstracks.track_duration.values, bins=np.linspace(0, 100, 101))
plt.show()
plt.hist(tracks.dstracks.track_duration.values / 24, bins=np.linspace(0, 100 / 24, 101))
plt.show()
plt.hist(tracks.dstracks.track_duration.values / 24, bins=np.linspace(1 / 48, 100 / 24 + 1 / 48, 101))
plt.show()
(tracks.dstracks.track_duration.values <= 12).sum()
((tracks.dstracks.track_duration.values > 12) & (tracks.dstracks.track_duration.values <= 24)).sum()
(tracks.dstracks.track_duration.values > 24).sum()
np.percentile(tracks.dstracks.track_duration.values, [33.3, 66.6])
(tracks.dstracks.track_duration.values <= 13).sum()
((tracks.dstracks.track_duration.values > 13) & (tracks.dstracks.track_duration.values <= 24)).sum()
((tracks.dstracks.track_duration.values > 13) & (tracks.dstracks.track_duration.values <= 21)).sum()
(tracks.dstracks.track_duration.values > 21).sum()
plt.hist(tracks.dstracks.track_duration.values, bins=np.linspace(0, 12, 13))
plt.hist(tracks.dstracks.track_duration.values, bins=np.linspace(12, 24, 13))
plt.hist(tracks.dstracks.track_duration.values, bins=np.linspace(24, 100, 77))
plt.show()
plt.hist(tracks.dstracks.track_duration.values, bins=np.linspace(0, 12, 13) + 0.5)
plt.hist(tracks.dstracks.track_duration.values, bins=np.linspace(12, 24, 13) + 0.5)
plt.hist(tracks.dstracks.track_duration.values, bins=np.linspace(24, 100, 77) + 0.5)
plt.show()
