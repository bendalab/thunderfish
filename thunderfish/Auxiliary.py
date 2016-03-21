__author__ = 'juan'
# Imports
import os
import numpy as np
import wave
import matplotlib.pyplot as plt
import sys
import seaborn as sns
from scipy.signal import butter, filtfilt
from IPython import embed

def peakdet(v, delta, x=None):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html
    Returns two arrays
    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    % [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    % maxima and minima ("peaks") in the vector V.
    % MAXTAB and MINTAB consists of two columns. Column 1
    % contains indices in V, and column 2 the found values.
    %
    % With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    % in MAXTAB and MINTAB are replaced with the corresponding
    % X-values.
    %
    % A point is considered a maximum peak if it has the maximal
    % value, and was preceded (to the left) by a value lower by
    % DELTA.
    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.
    """
    maxtab = []
    maxidx = []

    mintab = []
    minidx = []

    if x is None:
        x = np.arange(len(v), dtype=int)

    v = np.asarray(v)

    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')

    if not np.isscalar(delta):
        sys.exit('Input argument delta must be a scalar')

    if delta <= 0:
        sys.exit('Input argument delta must be positive')

    mn, mx = np.Inf, -np.Inf
    mnpos, mxpos = np.NaN, np.NaN

    lookformax = True

    for i in np.arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]

        if this < mn:
            mn = this
            mnpos = x[i]


        if lookformax:
            if this < mx-delta:
                maxtab.append(mx)
                maxidx.append(mxpos)
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append(mn)
                minidx.append(mnpos)
                mx = this
                mxpos = x[i]
                lookformax = True

    return np.array(maxtab), np.array(maxidx), np.array(mintab), np.array(minidx)

def df_histogram(freqs_array):
    """ This Function takes an array of wave-fish fundamental frequencies and calculates all possible
    difference-frequencies between all combinations of EODFs.

    :rtype : array with all possible dfs
    :param freqs_array: array of fish fundamental frequencies.
    """
    all_diffs = np.hstack([freqs_array - e for e in freqs_array])
    ret = all_diffs[all_diffs != 0.]
    return ret


def plot_dfs_histogram(dfs_array, binwidth='FD'):
    """ Plots a histogram of the difference frequencies

    :param binwidth: select the size of the binwidth. use 'FD' for Freedman-Diaconis rule
    :param dfs_array: array-like. list of difference frequencies.
    """
    q75, q25 = np.percentile(abs(dfs_array), [75, 25])

    inch_factor = 2.54
    sns.set_context("poster")
    sns.axes_style('white')
    sns.set_style("ticks")
    fig, ax = plt.subplots(figsize=(15./inch_factor, 10./inch_factor))

    if binwidth == 'FD':
        ax.hist(dfs_array, bins=int(2*(q75-q25) * len(dfs_array)**(-1./3.)),
                facecolor='cornflowerblue', alpha=0.8)  # Freedman-Diaconis rule for binwidth
    else:
        ax.hist(dfs_array, bins=binwidth, color='cornflowerblue', alpha=0.8)

    # Plot Cosmetics

    ax.set_ylabel('Counts', fontsize=16)
    ax.set_xlabel('Possible Beat-Frequencies [Hz]', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_title('Distribution of Beat-Frequencies', fontsize=16)
    sns.despine(fig=fig, ax=ax, offset=10)
    fig.tight_layout()
    fig.savefig('figures/histo_of_dfs.pdf')
    plt.close()


def draw_bwin_analysis_plot(filename, t_trace, eod_trace, no_of_peaks, cvs, mean_ampl):
    fs = 20
    inch_factor = 2.54
    sns.set_context("poster")
    sns.axes_style('white')
    sns.set_style("ticks")

    fig = plt.figure(figsize=(40. / inch_factor, 35. / inch_factor), num='Fish No. '+filename[-10:-8])

    ax1 = fig.add_subplot(5, 1, 1)
    ax2 = fig.add_subplot(5, 1, 2)
    ax3 = fig.add_subplot(5, 1, 3)
    ax4 = fig.add_subplot(5, 1, 4)
    ax5 = fig.add_subplot(5, 1, 5)

    # ax2 plots the Number of detected EOD-cycles
    window_nr = np.arange(len(no_of_peaks)) * 0.2
    ax2.plot(window_nr, no_of_peaks, 'o', ms=10, color='grey', mew=2., mec='black', alpha=0.6)
    ax2.set_ylabel('No. of detected\nEOD-cycles', fontsize=fs)

    # ax3 plots the Amplitude Coefficient of Variation
    ax3.plot(window_nr, cvs, 'o', ms=10, color='grey', mew=2., mec='black', alpha=0.6)
    ax3.set_ylabel('Soundtrace Amplitude\nVariation Coefficient', fontsize=fs)

    # ax4 plots the Mean Amplitude of each Window
    ax4.plot(window_nr, mean_ampl, 'o', ms=10, color='grey', mew=2., mec='black', alpha=0.6)
    ax4.set_ylabel('Mean Window\nAmplitude [a.u]', fontsize=fs)

    ax = np.array([ax1, ax2, ax3, ax4, ax5])
    return ax


def draw_bwin_in_plot(ax, sampl_rate, t_trace, eod_trace, start_bwin, len_bwin, pk_idx, tr_idx):

    fs = 20
    w_end = start_bwin + len_bwin
    w_bool = (t_trace >= start_bwin) & (t_trace <= w_end)
    time_bwin = t_trace[w_bool]
    eod_bwin = eod_trace[w_bool]

    # ax1 plots the raw soundtrace with peak and trough detection
    ax1 = ax[0]
    ax1.plot(t_trace, eod_trace, color='royalblue', lw=3)
    ax1.plot(t_trace[pk_idx], eod_trace[pk_idx], 'o', mfc='crimson', mec='k', mew=2., ms=12)
    ax1.plot(t_trace[tr_idx], eod_trace[tr_idx], 'o', mfc='lime', mec='k', mew=2., ms=12)

    ax1.set_xlim([start_bwin/2., start_bwin/2. + 0.1])
    ax1.set_xlabel('Time [sec]', fontsize=fs)
    ax1.set_ylabel('Amplitude [a.u]', fontsize=fs)


    # Plot raw trace
    ax5 = ax[-1]
    ax5.plot(t_trace[t_trace < start_bwin], eod_trace[[t_trace < start_bwin]], color='royalblue', lw=3, rasterized=True)
    ax5.plot(t_trace[t_trace > start_bwin+len_bwin], eod_trace[[t_trace > start_bwin+len_bwin]],
             color='royalblue', lw=3, rasterized=True)
    # Plot best window
    ax5.plot(time_bwin, eod_bwin, color='purple', alpha=0.8, lw=3, rasterized=True)
    plt.text(start_bwin+len_bwin/2., np.max(eod_bwin) + 0.2*np.max(eod_bwin),
             'Best Window', ha='center', va='center')

    ax5.set_xlabel('Time [sec]', fontsize=fs)
    ax5.set_ylabel('Amplitude [a.u]', fontsize=fs)

    for enu, axis in enumerate(ax):
        ax_ylims = axis.get_ylim()
        fix_plot_ticks(axis, ax_ylims)
        axis.tick_params(which='both', labelsize=fs-2)
        if 0 < enu <= 3:
            axis.set_xlabel('Time [sec]', fontsize=fs)
        if enu > 0:
            axis.set_xlim((0, 25))

        sns.despine(ax=axis, offset=10)
    plt.tight_layout()
    plt.show()


def fix_plot_ticks(ax, axlims, tick_no=5):
    ticks = np.linspace(axlims[0], axlims[1], tick_no)
    ax.set_yticks(ticks)
    pass


def butter_lowpass(highcut, fs, order=5):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order, high, btype='low')
    return b, a


def butter_lowpass_filter(data, highcut, fs, order=5):
    b, a = butter_lowpass(highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def load_trace(modfile, analysis_length=30):
    """ Loads modified .wav file from avconv with the library wave. This function returns an array with the time trace,
    an array with the amplitude values with the same length as the time-trace-array and finally it returns the sample-
    rate as a float. In order to accelerate the code, the default analysis-length are
    the first 30 seconds of the sound file.
    """
    recording = wave.open(modfile)
    sample_rate = recording.getframerate()
    frames_to_read = analysis_length * sample_rate  # read not more than the first 30 seconds
    data = np.fromstring(recording.readframes(frames_to_read), np.int16).astype(float)  # read data
    data -= np.mean(data)
    time_length = float(data.size) / sample_rate
    t_trace = np.linspace(0.0, time_length, num=data.size)
    return t_trace, data, sample_rate


def conv_to_single_ch_audio(audiofile):
    """ This function uses the software avconv to convert the current file to a single channel audio wav-file
    (or mp3-file), with which we can work afterwards using the package wave. Returns the name of the modified
    file as a string

    :rtype : str
    :param audiofile: sound-file that was recorded
    """

    base, ext = os.path.splitext(audiofile)
    base = base.split('/')[-1]
    new_mod_filename = 'recording_' + base + '_mod.wav'
    os.system('avconv -i {0:s} -ac 1 -y -acodec pcm_s16le {1:s}'.format(audiofile, new_mod_filename))
    return new_mod_filename


def create_outp_folder(filepath, out_path='.'):

    field_folder = '/'.join(filepath.split('.')[-2].split('/')[-3:-1])

    paths = {1: out_path, 2: field_folder}

    for k in paths.keys():
        if paths[k][-1] != '/':
            paths[k] += '/'
    new_folder = ''.join(paths.values())

    return new_folder


if __name__ == '__main__':
    # Test the df_histogram function
    a = np.load('%s' % sys.argv[1])
    dfs = df_histogram(a)
    plot_dfs_histogram(dfs)