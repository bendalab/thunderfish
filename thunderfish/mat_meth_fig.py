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

    wave_p2v_prop = np.load('wave_p2v_prop.npy')
    pulse_p2v_prop = np.load('pulse_p2v_prop.npy')

    wave_skewness = np.load('wave_skewness.npy')
    pulse_skewness = np.load('pulse_skewness.npy')

    # embed()
    # quit()

    return wave_psd_prop, pulse_psd_prop, wave_p2v_prop, pulse_p2v_prop, wave_skewness, pulse_skewness


def plot_data(wave_prop, pulse_prop, wave_p2v_prop, pulse_p2v_prop):
    # make boxplots !!!
    # lieber punktewolke !!!
    sns.set_context("poster")
    sns.axes_style('white')
    # sns.set_style("ticks")

    fig, ax = plt.subplots()
    ax.plot(wave_prop, wave_p2v_prop, 'o', color = 'red', label='wave-type', alpha= 0.7, markersize= 5)
    ax.plot(pulse_prop, pulse_p2v_prop, 'o', color= 'blue', label= 'pulse-type', alpha= 0.7, markersize= 5)
    plt.legend()
    # ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel('psd_algo', fontsize=14)
    ax.set_ylabel('soundtrace_algo', fontsize=14)
    plt.ylim([-0.1, 1.1])
    # fig.tight_layout()
    plt.draw()

    #####################################
    fig, ax = plt.subplots()
    ax.plot([1+random.uniform(-0.1, 0.1) for i in np.arange(len(wave_prop))], wave_prop, '.', color = 'red', label='wave-type', alpha= 0.7)
    ax.plot([1+random.uniform(-0.1, 0.1) for i in np.arange(len(pulse_prop))], pulse_prop, '.', color = 'blue', label='pulse-type', alpha = 0.7)

    plt.legend()
    ax.set_xticks([1, 2])

    ax.set_ylabel('psd_prop', color='k')
    for tl in ax.get_yticklabels():
        tl.set_color('k')

    ax2 = ax.twinx()
    ax2.plot([2+random.uniform(-0.1, 0.1) for i in np.arange(len(wave_p2v_prop))], wave_p2v_prop, '.', color = 'red', alpha= 0.7)
    ax2.plot([2+random.uniform(-0.1, 0.1) for i in np.arange(len(pulse_p2v_prop))], pulse_p2v_prop, '.', color = 'blue', alpha= 0.7)
    ax2.set_ylabel('soundtrace_prop', color='k')
    for tl in ax2.get_yticklabels():
        tl.set_color('k')

    ax.set_xticklabels(['psd_algo', 'soundtrace_algo'])
    plt.xlim([0.5, 2.5])
    plt.ylim([-0.1, 1.1])
    plt.tight_layout()
    plt.draw()


def plot_psd_algo(wave_prop, pulse_prop, wave_p2v_prop, pulse_p2v_prop, wave_skewness, pulse_skewness):
    inch_factor = 2.54
    sns.set_context("poster")
    sns.axes_style('white')
    # sns.set_style("ticks")

    fig2= plt.figure(figsize=(45./ inch_factor, 20./ inch_factor))
    ax1 = fig2.add_subplot(2, 4, (1, 5))
    dafr = pd.DataFrame([wave_prop, pulse_prop]) #turn
    dafr = dafr.transpose()
    dafr.columns = ['wave', 'pulse']
    sns.violinplot(data=dafr,  ax=ax1)
    ax1.set_ylabel('psd_proportion')
    ax1.set_xlabel('EOD-type')
    ax1.set_title('Fishsorting based on PSD')

    ax4 = fig2.add_subplot(2, 4, (2, 6))
    dafr2 = pd.DataFrame([wave_skewness, pulse_skewness])
    dafr2 = dafr2.transpose()
    dafr2.columns = ['wave', 'pulse']
    sns.violinplot(data=dafr2, ax= ax4)
    ax4.set_ylabel('skewness')
    ax4.set_xlabel('PSD-type')
    ax4.set_title('Fishsorting based on PSD')

    pulse_psd_data = np.load('pulse_psd_data.npy')
    ax2 = fig2.add_subplot(2, 4, (3, 4))
    ax2.axis([0, 3000, min(pulse_psd_data[1][:len(pulse_psd_data[0][pulse_psd_data[0]<3000])]), max(pulse_psd_data[1])+10])
    ax2.plot(pulse_psd_data[0], pulse_psd_data[1], lw=2, color='green', alpha=0.7)
    # ax2.hist(pulse_psd_data[1][pulse_psd_data[0]<1000], 100)
    # print sps.kurtosis(pulse_psd_data[1][pulse_psd_data[0]<1000])
    ax2.set_xlabel('time [ms]')
    ax2.set_ylabel('Amplitude [a.u.]')
    ax2.set_title('Pulsefish PSD')

    wave_psd_data = np.load('wave_psd_data.npy')
    ax3 = fig2.add_subplot(2, 4, (7, 8))
    ax3.axis([0, 3000, min(wave_psd_data[1][:len(wave_psd_data[0][wave_psd_data[0]<3000])]), max(wave_psd_data[1])+10])
    ax3.plot(wave_psd_data[0], wave_psd_data[1], lw=2, color='blue', alpha=0.7)
    # ax3.hist(wave_psd_data[1][wave_psd_data[0]<1000], 100)
    # print sps.kurtosis(wave_psd_data[1][wave_psd_data[0]<1000])
    ax3.set_xlabel('time [ms]')
    ax3.set_ylabel('Amplitude [a.u.]')
    ax3.set_title('Wavefish PSD')

    # embed()
    # quit()

    fig2.tight_layout()
    plt.draw()


def plot_p2v_algo(wave_prop, pulse_prop, wave_p2v_prop, pulse_p2v_prop):
    samplerate = 44100     ### really hard code

    inch_factor = 2.54
    sns.set_context("poster")
    sns.axes_style('white')
    # sns.set_style("ticks")

    fig3= plt.figure(figsize=(45./ inch_factor, 20./ inch_factor))

    ax1 = fig3.add_subplot(2, 3, (1, 4))
    dafr2 = pd.DataFrame([wave_p2v_prop, pulse_p2v_prop])
    dafr2 =dafr2.transpose()
    dafr2.columns = ['wave', 'pulse']
    sns.violinplot(data=dafr2, ax=ax1)
    ax1.set_ylabel('p2v_proportion')
    ax1.set_xlabel('EOD-type')
    ax1.set_title('Fishsorting based on soundtrace')

    pulse_trace_data = np.load('pulse_trace_data.npy')
    ax2 = fig3.add_subplot(2, 3, (2, 3))
    ax2.plot(np.arange(3000) * 1.0 / samplerate * 1000, pulse_trace_data[:3000] / 2.0 ** 15, lw=2, color='green', alpha=0.7)
    ax2.set_xlabel('time [ms]')
    ax2.set_ylabel('Amplitude [a.u.]')
    ax2.set_title('Pulsefish EODs')

    wave_trace_data = np.load('wave_trace_data.npy')
    ax3 = fig3.add_subplot(2, 3, (5, 6))
    ax3.plot(np.arange(500) * 1.0 / samplerate * 1000, wave_trace_data[:500] / 2.0 ** 15, lw=2, color='blue', alpha=0.7)
    ax3.set_xlabel('time [ms]')
    ax3.set_ylabel('Amplitude [a.u.]')
    ax3.set_title('Wavefish EODs')

    fig3.tight_layout()
    plt.draw()


def main():
    wave_psd_prop, pulse_psd_prop, wave_p2v_prop, pulse_p2v_prop, wave_skewness, pulse_skewness= load_data()

    # plot_data(wave_psd_prop, pulse_psd_prop, wave_p2v_prop, pulse_p2v_prop)

    plot_psd_algo(wave_psd_prop, pulse_psd_prop, wave_p2v_prop, pulse_p2v_prop, wave_skewness, pulse_skewness)

    plot_p2v_algo(wave_psd_prop, pulse_psd_prop, wave_p2v_prop, pulse_p2v_prop)

    plt.show()

if __name__ == '__main__':
    main()