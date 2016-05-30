import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
from IPython import embed
import scipy.stats as sps
from harmonicgroups import *


def load_data():
    wave_psd_prop = np.load('wave_PSD_algor.npy')
    pulse_psd_prop = np.load('pulse_PSD_algor.npy')
    multi_wave_psd_prop = np.load('multi_wave_PSD_algor.npy')

    wave_p2v_prop = np.load('wave_p2v_algor.npy')
    pulse_p2v_prop = np.load('pulse_p2v_algor.npy')
    multi_wave_p2v_prop = np.load('multi_wave_p2v_algor.npy')

    wave_hist_algor = np.load('wave_hist_algor.npy')
    pulse_hist_algor = np.load('pulse_hist_algor.npy')
    multi_wave_hist_algor = np.load('multi_wave_hist_algor.npy')

    mixed_psd_prop = np.load('mixed_PSD_algor.npy')
    mixed_p2v_prop = np.load('mixed_p2v_algor.npy')
    mixed_hist_algor = np.load('mixed_hist_algor.npy')

    return wave_psd_prop, pulse_psd_prop, multi_wave_psd_prop, wave_p2v_prop, pulse_p2v_prop, multi_wave_p2v_prop,\
           wave_hist_algor, pulse_hist_algor, multi_wave_hist_algor, mixed_psd_prop, mixed_p2v_prop, mixed_hist_algor


def plot_data(wave_psd_prop, pulse_psd_prop, multi_wave_psd_prop, wave_p2v_prop, pulse_p2v_prop,
              multi_wave_p2v_prop, wave_hist_algor, pulse_hist_algor, multi_wave_hist_algor, mixed_psd_prop,
              mixed_p2v_prop, mixed_hist_algor):
    inch_factor = 2.54
    sns.set_context("poster")
    sns.axes_style('white')
    sns.set_style("ticks")
    fs = 20

    fig1, ax = plt.subplots(figsize=(40./ inch_factor, 25./inch_factor))
    ax.plot(wave_psd_prop, wave_p2v_prop, 'o', color = 'royalblue', label='Wave-type', alpha= 0.7, markersize= 20, mec='k', mew=3)
    ax.plot(pulse_psd_prop, pulse_p2v_prop, 'o', color= 'firebrick', label= 'Pulse-type', alpha= 0.7, markersize= 20, mec='k', mew=3)
    ax.plot(mixed_psd_prop, mixed_p2v_prop, 'o', color = 'darkorange', label='Mixed', alpha= 0.7, markersize= 17, mec='k', mew=3)
    ax.tick_params(axis='both', which='major', labelsize=fs - 2)

    plt.legend(fontsize=fs-2)
    # ax.set_xlabel('P / ' + r'$\rho$', fontsize=fs)
    ax.set_xlabel(r'$\langle \rho/P \rangle$', fontsize=fs)
    ax.set_ylabel(r'$\langle \tau/T \rangle$', fontsize=fs)
    ax.set_title('Relation between\n' +r'$\langle \rho/P \rangle$' + ' and ' +r'$\langle \tau/T \rangle$' , fontsize=fs + 2)
    ax.set_ylim([0, 0.5])

    sns.despine(fig=fig1, ax=ax, offset=10)
    fig1.tight_layout()
    # plt.draw()

def plot_psd_algo(wave_prop, pulse_prop, multi_wave_prop, mixed_prop):


    inch_factor = 2.54
    sns.set_context("poster")
    sns.axes_style('white')
    sns.set_style("ticks")
    fs = 20

    fig2= plt.figure(figsize=(40./ inch_factor, 20./ inch_factor))
    ax1 = fig2.add_subplot(2, 3, (1, 4))
    dafr = pd.DataFrame([wave_prop, pulse_prop]) #turn
    dafr = dafr.transpose()
    dafr.columns = ['Wave', 'Pulse']
    sns.violinplot(data=dafr,  ax=ax1, inner=None, linewidth=2, color='k', palette=("royalblue", "firebrick"), scale="width")
    ax1.set_ylabel(r'$\langle \rho/P \rangle$', fontsize=fs)
    ax1.set_xlabel('EOD-type', fontsize = fs)
    ax1.set_title(r'$\langle \rho/P \rangle$'+' for different\nEOD-types', fontsize = fs + 2)
    ax1.tick_params(axis='both', which='major', labelsize=fs - 2)
    ax1.set_ylim([0, 0.6])

    pulse_psd_data = np.load('pulse_psd_data.npy')
    ax2 = fig2.add_subplot(2, 3, (2, 3))
    ax2.axis([0, 1500, min(pulse_psd_data[1][:len(pulse_psd_data[0][pulse_psd_data[0]<1500])]), max(pulse_psd_data[1])+10])
    ax2.plot(pulse_psd_data[0], pulse_psd_data[1], lw=2, color='firebrick', alpha=0.9)
    ax2.set_xlabel('Frequency [Hz]', fontsize= fs)
    ax2.set_ylabel('Power [dB]', fontsize= fs)
    ax2.set_title('Pulsefish PSD\n', fontsize= fs +2)
    ax2.tick_params(axis='both', which='major', labelsize=fs - 2)

    ###################################
    fres = pulse_psd_data[0][-1] / len(pulse_psd_data[0])
    freq_steps = int(125 / fres)

    p1 = []
    p99 = []
    p25 = []
    p75 = []
    for i in np.arange((1500 / fres) // freq_steps):  # does all of this till the frequency of 3k Hz
        bdata = pulse_psd_data[1][i * freq_steps:i * freq_steps + freq_steps-1]
        p1.append(np.percentile(bdata, 1))
        p99.append(np.percentile(bdata, 99))
        p25.append(np.percentile(bdata, 25))
        p75.append(np.percentile(bdata, 75))
        # embed()
    x_freqs = [freq_steps / 2 *fres + i * freq_steps * fres for i in np.arange(len(p1))]

    ax2.plot(x_freqs, p1, '-', color= 'green')
    ax2.plot(x_freqs, p25, '-', color= 'darkorange')
    ax2.plot(x_freqs, p75, '-', color= 'darkorange')
    ax2.plot(x_freqs, p99, '-', color= 'green')

    ax2.plot([x_freqs[3], x_freqs[3]], [p1[3], p99[3]], '-', color= 'k', lw=5, label= 'P')
    ax2.plot([x_freqs[4], x_freqs[4]], [p25[4], p75[4]], '-', color= 'royalblue', lw=5, label= r'$ \rho $')
    plt.legend(bbox_to_anchor=(1, 1.2), ncol=2)


    wave_psd_data = np.load('wave_psd_data.npy')
    ax3 = fig2.add_subplot(2, 3, (5, 6))
    ax3.axis([0, 1500, min(wave_psd_data[1][:len(wave_psd_data[0][wave_psd_data[0]<1500])]), max(wave_psd_data[1])+10])
    ax3.plot(wave_psd_data[0], wave_psd_data[1], lw=2, color='royalblue', alpha=0.9)
    ax3.set_xlabel('Frequency [Hz]', fontsize= fs)
    ax3.set_ylabel('Power [dB]', fontsize= fs)
    ax3.set_title('Wavefish PSD\n', fontsize= fs +2)
    ax3.tick_params(axis='both', which='major', labelsize=fs - 2)

    sns.despine(fig=fig2, ax=[ax1, ax2, ax3], offset=10)
    fig2.tight_layout()
    # plt.draw()


def accept_peak_heights(times, data, peak_inx, index, trough_inx, min_inx, threshold, check_conditions):
    return [times[peak_inx], data[peak_inx]]


def plot_p2v_algo(wave_p2v_prop, pulse_p2v_prop, multi_wave_p2v_prop):
    samplerate = 44100     ### really hard code

    inch_factor = 2.54
    sns.set_context("poster")
    sns.axes_style('white')
    sns.set_style("ticks")
    fs = 20

    fig3= plt.figure(figsize=(40./ inch_factor, 20./ inch_factor))

    ax1 = fig3.add_subplot(2, 3, (1, 4))
    dafr2 = pd.DataFrame([wave_p2v_prop, pulse_p2v_prop])
    dafr2 =dafr2.transpose()
    dafr2.columns = ['Wave', 'Pulse']
    sns.violinplot(data=dafr2, ax=ax1, inner=None, linewidth=2, color='k', palette=("royalblue", "firebrick"), scale="width")
    ax1.set_ylabel(r'$\langle \tau/T \rangle$', fontsize=fs)
    ax1.set_xlabel('EOD-type', fontsize= fs)
    ax1.set_title(r'$\langle \tau/T \rangle$'+' for different\nEOD-types', fontsize= fs + 2)
    ax1.tick_params(axis='both', which='major', labelsize=fs - 2)
    ax1.set_ylim([0, 0.5])

    pulse_trace_data = np.load('pulse_trace_data.npy')
    time = np.arange(3000) * 1.0 / samplerate * 1000
    data = pulse_trace_data[500:3500] / 2.0 ** 15
    peaks = detect_peaks(time, data, 0.25*(np.max(data)-np.min(data)), accept_peak_heights)
    troughs = detect_peaks(time, -data, 0.25*(np.max(data)-np.min(data)), accept_peak_heights)
    ax2 = fig3.add_subplot(2, 3, (2, 3))
    ax2.plot(peaks[:,0], peaks[:,1], 'go', markersize=15, alpha= 0.7)
    ax2.plot(troughs[:,0], -troughs[:,1], 'o',color='darkorange', markersize=15, alpha= 0.7)

    ax2.plot(time, data, lw=2, color='firebrick', alpha=0.9)
    ax2.set_xlim([0, 60])
    ax2.set_ylim([-1, 1])
    ax2.set_xlabel('Time [ms]', fontsize= fs)
    ax2.set_ylabel('Amplitude [a.u.]', fontsize= fs)
    ax2.set_title('Pulsefish EODs\n', fontsize= fs + 2)
    ax2.tick_params(axis='both', which='major', labelsize=fs - 2)

    wave_trace_data = np.load('wave_trace_data.npy')
    time = np.arange(500) * 1.0 / samplerate * 1000
    data = wave_trace_data[:500] / 2.0 ** 15
    peaks = detect_peaks(time, data, 0.75*(np.max(data)-np.min(data)), accept_peak_heights)
    troughs = detect_peaks(time, -data, 0.75*(np.max(data)-np.min(data)), accept_peak_heights)
    ax3 = fig3.add_subplot(2, 3, (5, 6))
    ax3.plot(peaks[:,0], peaks[:,1], 'go', markersize=15, alpha=0.7)
    ax3.plot(troughs[:,0], -troughs[:,1], 'o', color='darkorange',  markersize=15, alpha=0.7)

    # ax3.annotate("", xy= (peaks[1][0], peaks[1][1]), xytext=(peaks[0][0], peaks[0][1]), arrowprops={"facecolor": 'red', "shrink": 0.05})
    # ax3.annotate('T', xy=(1.55, 0.042), xycoords='data', xytext=(5, 0), textcoords='offset points')

    # ax3.annotate("", xy= (troughs[1][0], -troughs[1][1]), xytext=(peaks[0][0], -troughs[1][1]), arrowprops={"facecolor": 'k', "shrink": 0.05})
    # ax3.annotate(r'$\tau$', xy=(1.2, -0.04), xycoords='data', xytext=(5, 0), textcoords='offset points')

    ax3.plot([peaks[0][0], peaks[1][0]], [peaks[0][1], peaks[0][1]], '-', color='k', label='T', lw=5)
    ax3.plot([peaks[0][0], troughs[1][0]], [-troughs[1][1], -troughs[1][1]], '-', color='royalblue', label=r'$\tau$', lw=5)
    plt.legend(bbox_to_anchor=(1, 1.2), ncol=2)

    ax3.plot(np.arange(500) * 1.0 / samplerate * 1000, wave_trace_data[:500] / 2.0 ** 15, lw=2, color='royalblue', alpha=0.9)
    ax3.set_xlim([0, 11])
    ax3.set_ylim([-0.06, 0.06])
    ax3.set_xlabel('Time [ms]', fontsize= fs)
    ax3.set_ylabel('Amplitude [a.u.]', fontsize= fs)
    ax3.set_title('Wavefish EODs\n', fontsize= fs +2)
    ax3.tick_params(axis='both', which='major', labelsize=fs - 2)

    sns.despine(fig=fig3, ax=[ax1, ax2, ax3], offset=10)
    fig3.tight_layout()
    # plt.draw()


def plot_hist_algo(wave_hist_algor, pulse_hist_algor, multi_wave_hist_algor):
    inch_factor = 2.54
    sns.set_context("poster")
    sns.axes_style('white')
    # sns.set_style("ticks")

    fig4= plt.figure(figsize=(35./ inch_factor, 20./ inch_factor))
    ax1 = fig4.add_subplot(2, 3, (1, 4))
    dafr = pd.DataFrame([wave_hist_algor, multi_wave_hist_algor, pulse_hist_algor]) #turn
    dafr = dafr.transpose()
    dafr.columns = ['wave', 'multi-wave', 'pulse']
    sns.violinplot(data=dafr,  ax=ax1, col=("blue", "green", "red"))
    ax1.set_ylabel('psd_proportion')
    ax1.set_xlabel('EOD-type')
    ax1.set_title('Fishsorting based on PSD')

    wave_psd_data = np.load('wave_psd_data.npy')
    wave_hist_data = wave_psd_data[1][:len(wave_psd_data[0][wave_psd_data[0]<1500])]
    ax3 = fig4.add_subplot(2, 3, (2, 5))
    n, bin, patch = ax3.hist(wave_hist_data, 50, color='blue', alpha=0.7, normed=True)
    # ax3.set_ylim([0, max(n)+10])
    ax3.set_ylabel('counts in histogram bin')
    ax3.set_xlabel('amplitude of PSD')
    ax3.set_title('Histogram of pulsefish PSD')

    pulse_psd_data = np.load('pulse_psd_data.npy')
    pulse_hist_data = pulse_psd_data[1][:len(pulse_psd_data[0][pulse_psd_data[0]<1500])]
    ax2 = fig4.add_subplot(2, 3, (3, 6))
    ax2.hist(pulse_hist_data, 50, color='red', alpha=0.7, normed=True)
    # ax2.set_ylim([0, max(n)+10])
    ax2.set_ylabel('counts in histogram bin')
    ax2.set_xlabel('amplitude of PSD')
    ax2.set_title('Histogram of pulsefish PSD')

    fig4.tight_layout()

def main():
    wave_psd_prop, pulse_psd_prop, multi_wave_psd_prop, wave_p2v_prop, pulse_p2v_prop, multi_wave_p2v_prop, \
    wave_hist_algor, pulse_hist_algor, multi_wave_hist_algor, mixed_psd_prop, mixed_p2v_prop, \
    mixed_hist_algor= load_data()

    plot_data(wave_psd_prop, pulse_psd_prop, multi_wave_psd_prop, wave_p2v_prop, pulse_p2v_prop, multi_wave_p2v_prop,
              wave_hist_algor, pulse_hist_algor, multi_wave_hist_algor, mixed_psd_prop, mixed_p2v_prop, mixed_hist_algor)

    plot_psd_algo(wave_psd_prop, pulse_psd_prop, multi_wave_psd_prop, mixed_psd_prop)

    plot_p2v_algo(wave_p2v_prop, pulse_p2v_prop, multi_wave_p2v_prop)

    # plot_hist_algo(wave_hist_algor, pulse_hist_algor, multi_wave_hist_algor)

    plt.show()

if __name__ == '__main__':
    main()
