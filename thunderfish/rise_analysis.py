import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scp
from IPython import embed


def load_data():
    m_rise_proportions = np.load('m_rise_proportions.npy')
    f_rise_proportions = np.load('f_rise_proportions.npy')

    m_df = np.load('m_df.npy')
    f_df = np.load('f_df.npy')

    mb_df = np.load('mb_df.npy')
    fb_df = np.load('fb_df.npy')

    return m_rise_proportions, f_rise_proportions, m_df, f_df, mb_df, fb_df


def plot_df(m_df, f_df, mb_df, fb_df, plot_single=False, combi_plot=True):

    inch_factor = 2.54
    colors = ['#BA2D22', '#F47F17', '#53379B', '#3673A4', '#AAB71B', '#DC143C', '#1E90FF', 'magenta']
    box_colors = [colors[6], colors[6], colors[1], colors[1]]

    male_p = np.zeros(len(m_df))
    male_p_up = np.zeros(len(m_df))
    male_p_do = np.zeros(len(m_df))
    female_p = np.zeros(len(m_df))
    female_p_up = np.zeros(len(m_df))
    female_p_do = np.zeros(len(m_df))

    if combi_plot:
        c_m_df = np.array([])
        c_mb_df = np.array([])
        c_f_df = np.array([])
        c_fb_df = np.array([])

        c_m_df_up = np.array([])
        c_mb_df_up = np.array([])
        c_f_df_up = np.array([])
        c_fb_df_up = np.array([])

        c_m_df_do = np.array([])
        c_mb_df_do = np.array([])
        c_f_df_do = np.array([])
        c_fb_df_do = np.array([])

    for i in range(len(m_df)):
        all_m_df = m_df[i][:, 0]
        all_mb_df = np.concatenate(([mb_df[i][j][:, 0] for j in range(len(mb_df[i]))]))
        all_f_df = f_df[i][:, 0]
        all_fb_df = np.concatenate(([fb_df[i][j][:, 0] for j in range(len(fb_df[i]))]))

        all_m_df_up = m_df[i][:, 1]
        all_mb_df_up = np.concatenate(([mb_df[i][j][:, 1] for j in range(len(mb_df[i]))]))
        all_f_df_up = f_df[i][:, 1]
        all_fb_df_up = np.concatenate(([fb_df[i][j][:, 1] for j in range(len(fb_df[i]))]))

        all_m_df_do = m_df[i][:, 2]
        all_mb_df_do = np.concatenate(([mb_df[i][j][:, 2] for j in range(len(mb_df[i]))]))
        all_f_df_do = f_df[i][:, 2]
        all_fb_df_do = np.concatenate(([fb_df[i][j][:, 2] for j in range(len(fb_df[i]))]))

        _, male_p[i] = scp.mannwhitneyu(all_m_df, all_mb_df)
        _, male_p_up[i] = scp.mannwhitneyu(all_m_df_up, all_mb_df_up)
        _, male_p_do[i] = scp.mannwhitneyu(all_m_df_do, all_mb_df_do)

        _, female_p[i] = scp.mannwhitneyu(all_f_df, all_fb_df)
        _, female_p_up[i] = scp.mannwhitneyu(all_f_df_up, all_fb_df_up)
        _, female_p_do[i] = scp.mannwhitneyu(all_f_df_do, all_fb_df_do)

        if plot_single:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, facecolor='white', figsize=(20. / inch_factor, 12. / inch_factor))
            bp = ax1.boxplot([all_m_df[~np.isnan(all_m_df)], all_mb_df[~np.isnan(all_mb_df)], all_f_df[~np.isnan(all_f_df)], all_fb_df[~np.isnan(all_fb_df)]], sym='', patch_artist=True)
            for enu, box in enumerate(bp['boxes']):
                box.set(facecolor=box_colors[enu])
            bp = ax2.boxplot([all_m_df_up[~np.isnan(all_m_df_up)], all_mb_df_up[~np.isnan(all_mb_df_up)],
                              all_f_df_up[~np.isnan(all_f_df_up)], all_fb_df_up[~np.isnan(all_fb_df_up)]], sym='', patch_artist=True)
            for enu, box in enumerate(bp['boxes']):
                box.set(facecolor=box_colors[enu])
            bp = ax3.boxplot([all_m_df_do[~np.isnan(all_m_df_do)], all_mb_df_do[~np.isnan(all_mb_df_do)],
                              all_f_df_do[~np.isnan(all_f_df_do)], all_fb_df_do[~np.isnan(all_fb_df_do)]], sym='', patch_artist=True)
            for enu, box in enumerate(bp['boxes']):
                box.set(facecolor=box_colors[enu])

            for enu, box in enumerate(bp['boxes']):
                box.set(facecolor=box_colors[enu])

            for ax in [ax1, ax2, ax3]:
                ax.set_xticklabels(['m', 'm (b)', 'f', 'f (b)'])
                ax.set_ylim([0, 210])

            ax1.set_title('$\Delta f$')
            ax2.set_title('pos $\Delta f$')
            ax3.set_title('neg $\Delta f$')

            ax1.set_ylabel('min $\Delta$ frequency [Hz]')
            plt.tight_layout()
            plt.show()

            plt.show()

        if combi_plot:
            c_m_df = np.concatenate((c_m_df, all_m_df))
            c_mb_df = np.concatenate((c_mb_df, all_mb_df))
            c_f_df = np.concatenate((c_f_df, all_f_df))
            c_fb_df = np.concatenate((c_fb_df, all_fb_df))

            c_m_df_up = np.concatenate((c_m_df_up, all_m_df_up))
            c_mb_df_up = np.concatenate((c_mb_df_up, all_mb_df_up))
            c_f_df_up = np.concatenate((c_f_df_up, all_f_df_up))
            c_fb_df_up = np.concatenate((c_fb_df_up, all_fb_df_up))

            c_m_df_do = np.concatenate((c_m_df_do, all_m_df_do))
            c_mb_df_do = np.concatenate((c_mb_df_do, all_mb_df_do))
            c_f_df_do = np.concatenate((c_f_df_do, all_f_df_do))
            c_fb_df_do = np.concatenate((c_fb_df_do, all_fb_df_do))

    if combi_plot:
        _, m_p = scp.mannwhitneyu(c_m_df[~np.isnan(c_m_df)], c_mb_df[~np.isnan(c_mb_df)])
        _, f_p = scp.mannwhitneyu(c_f_df[~np.isnan(c_f_df)], c_fb_df[~np.isnan(c_fb_df)])
        _, m_up_p = scp.mannwhitneyu(c_m_df_up[~np.isnan(c_m_df_up)], c_mb_df_up[~np.isnan(c_mb_df_up)])
        _, f_up_p = scp.mannwhitneyu(c_f_df_up[~np.isnan(c_f_df_up)], c_fb_df_up[~np.isnan(c_fb_df_up)])
        _, m_do_p = scp.mannwhitneyu(c_m_df_do[~np.isnan(c_m_df_do)], c_mb_df_do[~np.isnan(c_mb_df_do)])
        _, f_do_p = scp.mannwhitneyu(c_f_df_do[~np.isnan(c_f_df_do)], c_fb_df_do[~np.isnan(c_fb_df_do)])

        print('\n Minimum dfs:')
        print('male - all dfs:   p = %.3f' % (m_p * 6.))
        print('female - all dfs: p = %.3f' % (f_p * 6.))
        print('male - pos dfs:   p = %.3f' % (m_up_p * 6.))
        print('female - pos dfs: p = %.3f' % (f_up_p * 6.))
        print('male - neg dfs:   p = %.3f' % (m_do_p * 6.))
        print('female - neg dfs: p = %.3f' % (f_do_p * 6.))

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, facecolor='white', figsize=(20. / inch_factor, 12. / inch_factor))
        bp = ax1.boxplot([c_m_df[~np.isnan(c_m_df)], c_mb_df[~np.isnan(c_mb_df)], c_f_df[~np.isnan(c_f_df)],
                          c_fb_df[~np.isnan(c_fb_df)]], sym='', patch_artist=True)

        for enu, box in enumerate(bp['boxes']):
            box.set(facecolor=box_colors[enu])
        bp = ax2.boxplot([c_m_df_up[~np.isnan(c_m_df_up)], c_mb_df_up[~np.isnan(c_mb_df_up)],
                          c_f_df_up[~np.isnan(c_f_df_up)], c_fb_df_up[~np.isnan(c_fb_df_up)]], sym='', patch_artist=True)
        for enu, box in enumerate(bp['boxes']):
            box.set(facecolor=box_colors[enu])
        bp = ax3.boxplot([c_m_df_do[~np.isnan(c_m_df_do)], c_mb_df_do[~np.isnan(c_mb_df_do)],
                          c_f_df_do[~np.isnan(c_f_df_do)], c_fb_df_do[~np.isnan(c_fb_df_do)]], sym='', patch_artist=True)
        for enu, box in enumerate(bp['boxes']):
            box.set(facecolor=box_colors[enu])

        for ax in [ax1, ax2, ax3]:
            ax.set_xticklabels(['m', 'm (b)', 'f', 'f (b)'])
            ax.set_ylim([0, 210])

        ax1.set_title('$\Delta f$')
        ax2.set_title('pos $\Delta f$')
        ax3.set_title('neg $\Delta f$')

        ax1.set_ylabel('min $\Delta$ frequency [Hz]')
        plt.tight_layout()
        plt.show()

def plot_rise_proportions(m_rise_proportions, f_rise_proportions):
    inch_factor = 2.54
    colors = ['#BA2D22', '#F47F17', '#53379B', '#3673A4', '#AAB71B', '#DC143C', '#1E90FF', 'magenta']

    for i in range(len(m_rise_proportions)):
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, facecolor='white', figsize=(20. / inch_factor, 12. / inch_factor))
        x_val = np.arange(0.05, 1.05, 0.1)
        ax1.plot(x_val, m_rise_proportions[i][1], color=colors[6], linewidth=2)
        ax1.fill_between(x_val, m_rise_proportions[i][0], m_rise_proportions[i][2], color=colors[6], alpha=0.4)

        ax2.plot(x_val, f_rise_proportions[i][1], color=colors[1], linewidth=2)
        ax2.fill_between(x_val, f_rise_proportions[i][0], f_rise_proportions[i][2], color=colors[1], alpha=0.4)

        plt.show()


def rise_analysis():
    m_rise_proportions, f_rise_proportions, m_df, f_df, mb_df, fb_df = load_data()

    plot_rise_proportions(m_rise_proportions, f_rise_proportions)

    # plot_df(m_df, f_df, mb_df, fb_df)

    embed()
    quit()

if __name__ == '__main__':
    rise_analysis()