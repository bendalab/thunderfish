import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
from IPython import embed
import scipy.stats as sps


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

    sns.set_context("poster")
    sns.axes_style('white')
    # sns.set_style("ticks")

    fig1, ax = plt.subplots()
    ax.plot(wave_psd_prop, wave_p2v_prop, 'o', color = 'blue', label='wave-type', alpha= 0.7, markersize= 5)
    ax.plot(pulse_psd_prop, pulse_p2v_prop, 'o', color= 'red', label= 'pulse-type', alpha= 0.7, markersize= 5)
    # ax.plot(mixed_psd_prop, mixed_p2v_prop, 'o', color = 'orange', label='mixed', alpha= 1, markersize= 6)

    plt.legend()
    ax.set_xlabel('psd_algo', fontsize=14)
    ax.set_ylabel('p2v_algo', fontsize=14)
    ax.set_title('psd_algo vs. p2v_algo', fontsize=14)

    fig1.tight_layout()
    plt.draw()

    # fig2, ax2 = plt.subplots()
    # ax2.plot(wave_psd_prop, wave_hist_algor, 'o', color = 'blue', label='wave-type', alpha= 0.7, markersize=5)
    # ax2.plot(multi_wave_psd_prop, multi_wave_hist_algor, 'o', color = 'green', label='multi wave-type', alpha= 0.7, markersize=5)
    # ax2.plot(pulse_psd_prop, pulse_hist_algor, 'o', color = 'red', label='pulse-type', alpha= 0.7, markersize=5)
    #
    # plt.legend()
    # ax2.set_xlabel('psd_algo', fontsize=14)
    # ax2.set_ylabel('hist_algo', fontsize=14)
    # ax2.set_title('psd_algo vs. hist_algo')
    # fig2.tight_layout()
    # plt.draw()
    #
    # fig3, ax3 = plt.subplots()
    # ax3.plot(wave_p2v_prop, wave_hist_algor, 'o', color = 'blue', label='wave-type', alpha= 0.7, markersize=5)
    # # ax3.plot(multi_wave_p2v_prop, multi_wave_hist_algor, 'o', color = 'green', label='multi wave-type', alpha= 0.7, markersize=5)
    # ax3.plot(pulse_p2v_prop, pulse_hist_algor, 'o', color = 'red', label='pulse-type', alpha= 0.7, markersize=5)
    #
    # plt.legend()
    # ax3.set_xlabel('p2v_algo', fontsize=14)
    # ax3.set_ylabel('hist_algo', fontsize=14)
    # ax3.set_title('p2v_algo vs. hist_algo')
    # fig3.tight_layout()
    plt.draw()

def plot_psd_algo(wave_prop, pulse_prop, multi_wave_prop, mixed_prop):
    inch_factor = 2.54
    sns.set_context("poster")
    sns.axes_style('white')
    # sns.set_style("ticks")

    fig2= plt.figure(figsize=(35./ inch_factor, 20./ inch_factor))
    ax1 = fig2.add_subplot(2, 3, (1, 4))
    dafr = pd.DataFrame([wave_prop, pulse_prop]) #turn
    dafr = dafr.transpose()
    dafr.columns = ['wave', 'pulse']
    sns.violinplot(data=dafr,  ax=ax1, col=("blue", "red"))
    ax1.set_ylabel('psd_proportion')
    ax1.set_xlabel('EOD-type')
    ax1.set_title('Fishsorting based on PSD')

    pulse_psd_data = np.load('pulse_psd_data.npy')
    ax2 = fig2.add_subplot(2, 3, (2, 3))
    ax2.axis([0, 1500, min(pulse_psd_data[1][:len(pulse_psd_data[0][pulse_psd_data[0]<1500])]), max(pulse_psd_data[1])+10])
    ax2.plot(pulse_psd_data[0], pulse_psd_data[1], lw=2, color='red', alpha=0.7)
    ax2.set_xlabel('time [ms]')
    ax2.set_ylabel('Power [dB]')
    ax2.set_title('Pulsefish PSD')

    wave_psd_data = np.load('wave_psd_data.npy')
    ax3 = fig2.add_subplot(2, 3, (5, 6))
    ax3.axis([0, 1500, min(wave_psd_data[1][:len(wave_psd_data[0][wave_psd_data[0]<1500])]), max(wave_psd_data[1])+10])
    ax3.plot(wave_psd_data[0], wave_psd_data[1], lw=2, color='blue', alpha=0.7)
    ax3.set_xlabel('time [ms]')
    ax3.set_ylabel('Power [dB]')
    ax3.set_title('Wavefish PSD')

    fig2.tight_layout()
    plt.draw()


def plot_p2v_algo(wave_p2v_prop, pulse_p2v_prop, multi_wave_p2v_prop):
    samplerate = 44100     ### really hard code

    inch_factor = 2.54
    sns.set_context("poster")
    sns.axes_style('white')
    # sns.set_style("ticks")

    fig3= plt.figure(figsize=(35./ inch_factor, 20./ inch_factor))

    ax1 = fig3.add_subplot(2, 3, (1, 4))
    dafr2 = pd.DataFrame([wave_p2v_prop, pulse_p2v_prop])
    dafr2 =dafr2.transpose()
    dafr2.columns = ['wave', 'pulse']
    sns.violinplot(data=dafr2, ax=ax1, col=("blue", "red"))
    ax1.set_ylabel('p2v_proportion')
    ax1.set_xlabel('EOD-type')
    ax1.set_title('Fishsorting based on soundtrace')

    pulse_trace_data = np.load('pulse_trace_data.npy')
    ax2 = fig3.add_subplot(2, 3, (2, 3))
    ax2.plot(np.arange(3000) * 1.0 / samplerate * 1000, pulse_trace_data[500:3500] / 2.0 ** 15, lw=2, color='red', alpha=0.7)
    ax2.set_xlim([0, 60])
    ax2.set_xlabel('time [ms]')
    ax2.set_ylabel('Amplitude [a.u.]')
    ax2.set_title('Pulsefish EODs')

    wave_trace_data = np.load('wave_trace_data.npy')
    ax3 = fig3.add_subplot(2, 3, (5, 6))
    ax3.plot(np.arange(500) * 1.0 / samplerate * 1000, wave_trace_data[:500] / 2.0 ** 15, lw=2, color='blue', alpha=0.7)
    ax3.set_xlim([0, 11])
    ax3.set_xlabel('time [ms]')
    ax3.set_ylabel('Amplitude [a.u.]')
    ax3.set_title('Wavefish EODs')

    fig3.tight_layout()
    plt.draw()


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