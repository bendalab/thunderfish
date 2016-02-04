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

pulse_trace_prop = np.load('pulse_trace_proportions.npy')

wave_trace_prop = np.load('wave_trace_proportions.npy')

wd = [1 for i in np.arange(len(wave_diffs))]
pd = [1 for j in np.arange(len(pulse_diffs))]
wp = [1 for k in np.arange(len(wave_prop))]
pp = [1 for l in np.arange(len(pulse_prop))]
wt = [1 for m in np.arange(len(wave_trace_prop))]
pt = [1 for n in np.arange(len(pulse_trace_prop))]

fig, ax = plt.subplots(facecolor = 'white', nrows=1, ncols=3)
ax[0].set_title('difference (p90-p10)')
ax[0].plot(wd, wave_diffs, '.')
ax[0].plot(pd, pulse_diffs, '.', color='r')

ax[1].set_title('proportion ((p75-p25)/(p99-p1))')
ax[1].plot(wp, wave_prop, '.')
ax[1].plot(pp, pulse_prop, '.', color='r')

ax[2].set_title('trace proportions')
ax[2].plot(wt, wave_trace_prop, '.')
ax[2].plot(pt, pulse_trace_prop, '.', color='r')

plt.show()

fig, ax = plt.subplots(facecolor = 'white')
plt.xlabel('difference (p90-p10)')
plt.ylabel('proportion ((p75-p25)/(p99-p1))')
ax.plot(wave_diffs, wave_trace_prop, '.')
ax.plot(pulse_diffs, pulse_trace_prop, '.', color='r')
plt.show()