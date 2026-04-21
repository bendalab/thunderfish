"""
Analysis of EOD frequencies.


## Muscial intervals

- `musical_intervals`: names and frequency ratios of musical intervals
- `musical_intervals_short`: short names for musical intervals


## Frequency analysis

- `freq_diffs()`: matrix with frequency differences.
- `freq_ratios()`: matrix with frequency ratios.
- `freq_intervals()`: matrices with musical intervals and deviations thereof.


## Visualization

- `plot_freq_diffs()`: plot matrix with frequency differences.
- `plot_freq_ratios()`: plot matrix with frequency ratios.
- `plot_musical_intervals()`: plot matrix with musical intervals.

"""

import numpy as np

from matplotlib import ticker


musical_intervals = {
    'unison': (1/1, 1, 1, 0),             # 1
    'minor second': (16/15, 16, 15, 1),   # 1.0667
    'major second': (9/8, 9, 8, 2),       # 1.125
    'minor third': (6/5, 6, 5, 3),        # 1.2
    'major third': (5/4, 5, 4, 4),        # 1.25
    'forth': (4/3, 4, 3, 5),              # 1.3333
    'tritone': (45/32, 45, 32, 6),        # 1.4063, half way between forth and fifth: 17/6/2=1.4167, sqrt(2)=1.4142
    'fifth': (3/2, 3, 2, 7),              # 1.5
    'minor sixth': (8/5, 8, 5, 8),        # 1.6
    'major sixth': (5/3, 5, 3, 9),        # 1.6667
    'subminor seventh': (7/4, 7, 4, 9.5), # 1.75
    'minor seventh': (9/5, 9, 5, 10),     # 1.8
    'major seventh': (15/8, 15, 8, 11),   # 1.875
    'octave': (2/1, 2, 1, 12)             # 2
}
"""Name, frequency ratio, nominator, denominator, and index of musical intervals
"""

musical_intervals_short =  {
    'unison': 'P1',
    'minor second': 'm2',
    'major second': 'M2',
    'minor third': 'm3',
    'major third': 'M3',
    'forth': 'P4',
    'tritone': 'd5',
    'fifth': 'P5',
    'minor sixth': 'm6',
    'major sixth': 'M6',
    'subminor seventh': 'd7',
    'minor seventh': 'm7',
    'major seventh': 'M7',
    'octave': 'P8'
}
"""Short names for musical intervals
"""


def freq_diffs(freqs):
    """Matrix with frequency differences.
    
    Parameters
    ----------
    freqs: 1-D array of float
        List of frequencies.

    Returns
    -------
    deltafs: 2-D array of float
        Matrix with frequency differences for each pair of frequencies.
    """
    deltafs = freqs.reshape(-1, 1) - freqs.reshape(1, -1)
    return deltafs


def freq_ratios(freqs):
    """Matrix with frequency ratios.
    
    Parameters
    ----------
    freqs: 1-D array of float
        List of frequencies.

    Returns
    -------
    ratios: 2-D array of float
        Matrix with frequency ratios for each pair of frequencies.
    """
    ratios = freqs.reshape(-1, 1)/freqs.reshape(1, -1)
    return ratios


def freq_intervals(freqs):
    """Matrices with musical intervals and deviations thereof.
    
    Parameters
    ----------
    freqs: 1-D array of float
        List of frequencies.

    Returns
    -------
    intervals: 2-D array of int
        Matrix with index of nearest musical interval.
    diffs: 2-D array of float
        Matrix with deviation of freqeuncy ratio to nearest
        musical interval ratio.
    diff_fracs: 2-D array of float
        Matrix with deviation of freqeuncy ratio to nearest
        musical interval ratio relative to musical interval ratio.
    """
    all_intervals = np.array([musical_intervals[k][0]
                              for k in musical_intervals])
    ratios = freqs.reshape(-1, 1)/freqs.reshape(1, -1)
    intervals = np.zeros(ratios.shape, dtype=int)
    diffs = np.zeros(ratios.shape)
    diff_fracs = np.zeros(ratios.shape)
    for r in range(ratios.shape[0]):
        for c in range(ratios.shape[1]):
            if r != c and 0.5 <= ratios[r, c] < 2.05:
                if ratios[r, c] >= 0.98:
                    ratio = ratios[r, c]
                else:
                    ratio = 1/ratios[r, c]
                intervals[r, c] = np.argmin(np.abs(all_intervals - ratio))
                diffs[r, c] = ratio - all_intervals[intervals[r, c]]
                diff_fracs[r, c] = diffs[r, c]/all_intervals[intervals[r, c]]
            else:
                intervals[r, c] = -1
                diffs[r, c] = np.nan
                diff_fracs[r, c] = np.nan
    return intervals, diffs, diff_fracs


def plot_freq_diffs(ax, freqs, cmap='seismic', fontsize='large'):
    """Plot matrix with frequency differences.

    Parameters
    ----------
    ax: matplotlib axes
        Axes used for plotting.
    freqs: 1-D array of float
        List of frequencies.
    cmap: matplotlib color map
        Color map for coloring frequency differences.
    fontsize: str or float
        Font size for labels indicating frequency differences.
    """
    ax.set_title('Differences $\\Delta f$')
    deltafs = freq_diffs(freqs)
    vmax = np.max(np.abs(deltafs))
    cma = ax.pcolormesh(deltafs[::-1, :], cmap=cmap,
                        vmin=-vmax, vmax=vmax)
    for r in range(deltafs.shape[0]):
        for c in range(deltafs.shape[1]):
            ax.text(c + 0.5, r + 0.5, f'{deltafs[-1 - r, c]:.1f}',
                    ha='center', va='center',
                    fontsize=fontsize, clip_on=True,
                    bbox=dict(boxstyle='round,pad=0.1', ec='none',
                              fc='white', alpha=0.8))
    ax.set_aspect('equal')
    ax.xaxis.set_major_locator(ticker.FixedLocator(np.arange(len(freqs)) + 0.5))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter([f'{f:.1f}' for f in freqs]))
    ax.yaxis.set_major_locator(ticker.FixedLocator(np.arange(len(freqs)) + 0.5))
    ax.yaxis.set_major_formatter(ticker.FixedFormatter([f'{f:.1f}' for f in reversed(freqs)]))
    ax.set_xlabel('EOD$f_i$ [Hz]')
    ax.set_ylabel('EOD$f_j$ [Hz]')
    ax.get_figure().colorbar(cma, ax=ax, label='$\\Delta f$ [Hz]')


def plot_freq_ratios(ax, freqs, cmap='seismic', fontsize='large'):
    """Plot matrix with frequency ratios.

    Parameters
    ----------
    ax: matplotlib axes
        Axes used for plotting.
    freqs: 1-D array of float
        List of frequencies.
    cmap: matplotlib color map
        Color map for coloring frequency ratios.
    fontsize: str or float
        Font size for labels indicating frequency ratios.
    """
    ax.set_title('Ratios $f_1/f_0$')
    ratios = freq_ratios(freqs)
    cma = ax.pcolormesh(ratios[::-1, :], cmap=cmap, norm='log',
                        vmin=1/5, vmax=5)
    for r in range(ratios.shape[0]):
        for c in range(ratios.shape[1]):
            ax.text(c + 0.5, r + 0.5, f'{ratios[-1 - r, c]:.3f}',
                    ha='center', va='center',
                    fontsize=fontsize, clip_on=True,
                    bbox=dict(boxstyle='round,pad=0.1', ec='none',
                              fc='white', alpha=0.8))
    ax.set_aspect('equal')
    ax.xaxis.set_major_locator(ticker.FixedLocator(np.arange(len(freqs)) + 0.5))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter([f'{f:.1f}' for f in freqs]))
    ax.yaxis.set_major_locator(ticker.FixedLocator(np.arange(len(freqs)) + 0.5))
    ax.yaxis.set_major_formatter(ticker.FixedFormatter([f'{f:.1f}' for f in reversed(freqs)]))
    ax.set_xlabel('EOD$f_i$ [Hz]')
    ax.set_ylabel('EOD$f_j$ [Hz]')
    ax.get_figure().colorbar(cma, ax=ax, label='Ratio', ticks=[1/5, 1/2, 1, 2, 5], format='%g')


def plot_musical_intervals(ax, freqs, cmap='YlOrRd_r', fontsize='large'):
    """Plot matrix with musical intervals.

    Parameters
    ----------
    ax: matplotlib axes
        Axes used for plotting.
    freqs: 1-D array of float
        List of frequencies.
    cmap: matplotlib color map
        Color map for coloring deviations from musical intervals.
    fontsize: str or float
        Font size for labels indicating musical intervals and
        deviations thereof.
    """
    ax.set_title('Musical intervals')
    all_intervals = np.array([musical_intervals[k][0] for k in musical_intervals])
    if len(freqs) < 6:
        all_names = list(musical_intervals.keys())
    else:
        all_names = [musical_intervals_short[k] for k in musical_intervals]
    intervals, diffs, diff_fracs = freq_intervals(freqs)
    cma = ax.pcolormesh(100*np.abs(diff_fracs[::-1, :]), cmap=cmap,
                        vmin=0, vmax=1)
    for r in range(intervals.shape[0]):
        for c in range(intervals.shape[1]):
            if -1 - r != c and intervals[-1 - r, c] >= 0:
                idx = intervals[-1 - r, c]
                if len(freqs) < 6:
                    label = f'{all_intervals[idx]:.4f}\n{all_names[idx]}\n$\\Delta$={diffs[-1 - r, c]:.4f}\n{100*diff_fracs[-1 - r, c]:.1f}%'
                else:
                    label = f'{all_names[idx]}\n{100*diff_fracs[-1 - r, c]:.1f}%'
                ax.text(c + 0.5, r + 0.5, label,
                        ha='center', va='center',
                        fontsize=fontsize, clip_on=True,
                        bbox=dict(boxstyle='round,pad=0.1', ec='none',
                                  fc='white', alpha=0.8))
    ax.set_aspect('equal')
    ax.xaxis.set_major_locator(ticker.FixedLocator(np.arange(len(freqs)) + 0.5))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter([f'{f:.1f}' for f in freqs]))
    ax.yaxis.set_major_locator(ticker.FixedLocator(np.arange(len(freqs)) + 0.5))
    ax.yaxis.set_major_formatter(ticker.FixedFormatter([f'{f:.1f}' for f in reversed(freqs)]))
    ax.set_xlabel('EOD$f_i$ [Hz]')
    ax.set_ylabel('EOD$f_j$ [Hz]')
    ax.get_figure().colorbar(cma, ax=ax, extend='max',
                             format=ticker.PercentFormatter(decimals=1),
                             label='Deviation from musical interval')
