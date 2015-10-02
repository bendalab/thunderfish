__author__ = 'raab'
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
import scipy.stats as sp

pulse_diffs = np.load('pulse_diffs.npy')
# pulse_diffs = pulse_diffs.tolist()

wave_diffs = np.load('wave_diffs.npy')
# wave_diffs = wave_diffs.tolist()

pulse_prop = np.load('pulse_proportions.npy')
# pulse_prop = pulse_prop.tolist()

wave_prop = np.load('wave_proportions.npy')

wd = [1 for i in np.arange(len(wave_diffs))]
pd = [1 for j in np.arange(len(pulse_diffs))]
wp = [2 for k in np.arange(len(wave_prop))]
pp = [2 for l in np.arange(len(pulse_prop))]

fig = plt.subplots(facecolor = 'white')
plt.subplot(1, 2, 1)
plt.title('difference (p90-p10)')
plt.plot(wd, wave_diffs, '.')
plt.plot(pd, pulse_diffs, '.', color='r')

plt.subplot(1, 2, 2)
plt.title('proportion ((p75-p25)/(p99-p1))')
plt.plot(wp, wave_prop, '.')
plt.plot(pp, pulse_prop, '.', color='r')

plt.show()