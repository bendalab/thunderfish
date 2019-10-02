import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
from time import sleep
from tqdm import tqdm
from matplotlib.animation import ArtistAnimation
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import glob



def sf(t, s):
    return 1 / (1 + np.exp(s * -t))


def field_anim():
    anim_handles = []
    imgs = []
    for i in np.arange(11)+1:
        imgs.append(mpimg.imread('/home/raab/figures/field_anim/field%.0f.png' % int(i)))

    # fig, ax = plt.subplots(2, 1, figsize=(20/2.54, 20/2.54), facecolor='white')
    fig = plt.figure(figsize=(12/2.54, 12/2.54), facecolor='white')


    ax = fig.add_axes([.0, .325, 1, .75])
    ax.axis('off')
    im = ax.imshow(imgs[0])

    ax1= fig.add_axes([.1, .1, .8, .3])
    t = np.arange(0, 2, 1/20000)
    f1 = 1* np.sin(2 * np.pi * t)
    sin, = ax1.plot(t, f1, color='k', clip_on=False)
    ax1.set_xlim(0, 2)
    ax1.set_xticks([0, 0.5, 1, 1.5, 2])
    ax1.set_xticklabels([0, '1$\pi$', '2$\pi$', '$3\pi$', '$4\pi$'])

    ax1.set_ylim(-1, 1)
    ax1.set_yticks([-1, 0, 1])
    ax1.set_ylabel('Amplitude [mV]', fontsize=10)
    ax1.set_xlabel('Phase', fontsize=10)


    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')


    dot_counter = 0

    dots_x = np.arange(0, 2, 0.05) # 20 frames per cycle --> 50ms delay = 1Hz; 5ms delay = 10Hz
    img_idx = [6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 10, 9, 8, 7]

    # dots_x = np.arange(0, 2, .25) # 4 frames per cycle --> 250ms delay = 1Hz; 25ms delay = 10 Hz, 5ms delay = 50 Hz
    # img_idx = [6, 1, 6, 11, 6, 1, 6, 11]

    # dots_x = np.arange(.25, 2, .5) # 2 frames per cycle --> 500ms delay = 1Hz; 50ms delay = 10 Hz, 5ms delay = 100 Hz
    # img_idx = [1, 11, 1, 11]

    sleep(.1)

    # try:
    #     while True:
    if dot_counter >= len(dots_x) - 1:
        dot_counter = 0

    for i in img_idx:
        im = ax.imshow(imgs[i - 1])
        d, = ax1.plot(dots_x[dot_counter], 1* np.sin(2 * np.pi * dots_x[dot_counter]), 'o', color='firebrick')
        dot_counter += 1
        anim_handles.append([im, sin, d])

    # for img in imgs[:-1]:
    #     # im.remove()
    #     im = ax.imshow(img)
    #     # im.set_data(img)
    #
    #     # d.remove()
    #     d, = ax1.plot(dots_x[dot_counter], 1* np.cos(2 * np.pi * dots_x[dot_counter]), 'o', color='firebrick')
    #     # d.set_data(dots_x[dot_counter], 1* np.cos(2 * np.pi * dots_x[dot_counter]))
    #     dot_counter += 1
    #
    #     anim_handles.append([im, sin, d])
    #     # fig.canvas.draw()
    #     sleep(1/40)
    #
    # for img in imgs[::-1][:-1]:
    #     # im.remove()
    #     im = ax.imshow(img)
    #     # im.set_data(img)
    #
    #     # d.remove()
    #     d, = ax1.plot(dots_x[dot_counter], 1* np.cos(2 * np.pi * dots_x[dot_counter]), 'o', color='firebrick')
    #     # d.set_data(dots_x[dot_counter], 1* np.cos(2 * np.pi * dots_x[dot_counter]))
    #     dot_counter += 1
    #
    #     anim_handles.append([im, sin, d])
    #     # fig.canvas.draw()
    #     sleep(1/40)
    #
    # for img in imgs[:-1]:
    #     # im.remove()
    #     im = ax.imshow(img)
    #     # im.set_data(img)
    #
    #     # d.remove()
    #     d, = ax1.plot(dots_x[dot_counter], 1* np.cos(2 * np.pi * dots_x[dot_counter]), 'o', color='firebrick')
    #     # d.set_data(dots_x[dot_counter], 1* np.cos(2 * np.pi * dots_x[dot_counter]))
    #     dot_counter += 1
    #
    #     anim_handles.append([im, sin, d])
    #     # fig.canvas.draw()
    #     sleep(1/40)
    #
    # for img in imgs[::-1][:-1]:
    #     # im.remove()
    #     im = ax.imshow(img)
    #     # im.set_data(img)
    #
    #     # d.remove()
    #     d, = ax1.plot(dots_x[dot_counter], 1* np.cos(2 * np.pi * dots_x[dot_counter]), 'o', color='firebrick')
    #     # d.set_data(dots_x[dot_counter], 1* np.cos(2 * np.pi * dots_x[dot_counter]))
    #     dot_counter += 1
    #
    #     anim_handles.append([im, sin, d])
    #     # fig.canvas.draw()
    #     sleep(1/40)

    ani = ArtistAnimation(fig, anim_handles, interval=100, blit=True, repeat_delay=0) # 5 ms delay --> 10Hz
    # ani = ArtistAnimation(fig, anim_handles, interval=5, blit=True, repeat_delay=0)
    # ani = ArtistAnimation(fig, anim_handles, interval=5, blit=True, repeat_delay=0)
    ani.save('/home/raab/figures/field_anim/field_anim_0_5Hz.gif', dpi=160, writer='imagemagick')

    plt.close()

def main():

    save_folder = '/home/raab/figures/field_anim/imgs/'

    t = np.arange(0, 2, 1/20000)
    f1 = 1* np.sin(2 * np.pi * 604 * t)
    f2 = 1* np.sin(2 * np.pi * 600 * t)


    # ToDo: a2 -> dependent variable --> distace fixed ... sinus in the end: close + far
    # a2 = np.zeros(len(t))
    # a2[(t >= 3) & (t < 10)] = 0.5 * sf(t[(t >= 3) & (t < 10)] - 3 - 3.5, 1.5)
    # a2[(t >= 10) & (t < 20)] = np.ones(len(t[(t >= 10) & (t < 20)])) * 0.5
    # a2[(t >= 20) & (t < 30)] = np.cos(2 * np.pi * 0.2 * t[(t >= 20) & (t < 30)]) * 0.25 + 0.25
    # a2[(t >= 30) & (t < 45)] = np.cos(2 * np.pi * 0.4 * t[(t >= 30) & (t < 45)]) * 0.25 + 0.25

    # create comb signal, beat and envelope traces
    # comb_sig = f1+f2*a2

    # distance vs. amplitude (Knudseln 1974)
    # d = [5, 7.5, 10, 20, 30, 40, 50]
    # p = [10000, 3500, 2000, 500, 200, 120, 80]
    #
    # d_inter = np.arange(5, 50.01, .01)
    # p_inter = np.interp(d_inter, d, p)
    # p_inter = p_inter / np.max(p_inter)
    #
    # dist = []
    # for Cp in tqdm(a2):
    #     dist.append(d_inter[p_inter > Cp][-1])

    # get field images
    field_pos = mpimg.imread('/home/raab/figures/field_anim/field1.png')
    field_pos_inv = mpimg.imread('/home/raab/figures/field_anim/field1_inv.png')

    field_neg = mpimg.imread('/home/raab/figures/field_anim/field11.png')
    field_neg_inv = mpimg.imread('/home/raab/figures/field_anim/field11_inv.png')

    field_neu = mpimg.imread('/home/raab/figures/field_anim/field6.png')
    field_neu_inv = mpimg.imread('/home/raab/figures/field_anim/field6_inv.png')

    # create figure
    fig = plt.figure(figsize=(20/2.54, 12/2.54), facecolor='white')

    gs = gridspec.GridSpec(1, 1)
    gs.update(left=0.1, right=0.95, bottom=.1, top=.95)

    ax = plt.subplot(gs[0, 0])

    ax.set_ylabel('Amplitude [mV]', fontsize=10)
    ax.set_xlabel('Time [s]', fontsize=10)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    combsig_handle, = ax.plot(t, f1, color='dimgrey', alpha = 0)

    f1_init_handle, = ax.plot(t, f1 * 0.4 + 0.5, color='forestgreen')
    f2_init_handle, = ax.plot(t, f2 * 0.4 - 0.5, color='cornflowerblue')

    f1_handle, = ax.plot(t, f1, color='forestgreen', alpha = 0)
    f2_handle, = ax.plot(t, f2, color='cornflowerblue', alpha = 0)

    ax.set_xticks(np.arange(0, 45.01, 0.01))
    ax.set_xlim([0.99, 1.01])

    ax.set_yticks([-0.9, -0.5, -0.1, .1, .5, .9])
    ax.set_yticklabels([-1, 0, 1, -1, 0, 1])
    ax.set_ylim([-1, 1])

    embed()
    quit()

    draw = False
    save = True

    if draw:
        plt.show(block=False)

    ####################################################################################################################
    print('stage 1')


    ax.set_xlim([0.99, 1.01])
    # ax_dist.set_xlim([1, 1.01])

    if save:
        plt.savefig('/home/raab/figures/field_anim/imgs/f%.0f.png' % len(glob.glob('/home/raab/figures/field_anim/imgs/*.png')), dpi=300)
    else:
        sleep(2)

    for i in np.arange(1, -0.02, -0.02):
        f1_init_handle.set_alpha(i)
        f2_init_handle.set_alpha(i)

        if draw:
            fig.canvas.draw()
            sleep(1/120)
        if save:
            plt.savefig('/home/raab/figures/field_anim/imgs/f%.0f.png' % len(glob.glob('/home/raab/figures/field_anim/imgs/*.png')), dpi=300)


    print('stage 2')
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_yticklabels([-2, -1, 0, 1, 2])
    ax.set_ylim([-2, 2])
    # fig.canvas.draw()

    # remove init traces (2)
    # sleep(2)
    for i in np.arange(0, 1.02, 0.02):
        combsig_handle.set_alpha(i)
        if draw:
            fig.canvas.draw()
            sleep(1 / 120)
        if save:
            plt.savefig('/home/raab/figures/field_anim/imgs/f%.0f.png' % len(glob.glob('/home/raab/figures/field_anim/imgs/*.png')), dpi=300)

    sleep(2)


    ####################################################################################################################
    # include comb trace
    print('stage 3')
    # combsig_handle, = ax.plot((t[(t >= 0.99) & (t <= 1.01)] - 0.99) / 0.02, comb_sig[(t >= 0.99) & (t <= 1.01)], color='dimgrey', alpha=1)
    # for i in range(20):
    #     anim_handles.append([combsig_handle])
    # # fig.canvas.draw()
    # # sleep(2)
    # # combsig_handle.remove()

    ax.set_xticks(np.arange(0, 45.1, 0.1))
    # ax_dist.set_xticks(np.arange(0, 45.1, 0.1))

    for x0, x1 in zip(np.linspace(0.99, 0, 50), np.linspace(1.01, 2, 50)):
        ax.set_xlim(x0, x1)
        if draw:
            fig.canvas.draw()
            sleep(1 / 120)
        if save:
            plt.savefig('/home/raab/figures/field_anim/imgs/f%.0f.png' % len(glob.glob('/home/raab/figures/field_anim/imgs/*.png')), dpi=300)

    ax.set_xticks(np.arange(0, 46, 1))
    ax.set_xlim([0, 2])
    # ax_dist.set_xticks(np.arange(0, 46, 1))
    if draw:
        fig.canvas.draw()
        sleep(2)


    ####################################################################################################################
    print('stage 4')

    combsig_handle.remove()
    for i in np.linspace(0, 0.5, 50):
        combsig_handle, = ax.plot(t, f1+f2*i, color='dimgrey', alpha = 1)
        if draw:
            fig.canvas.draw()
            sleep(1 / 120)
        if save:
            plt.savefig('/home/raab/figures/field_anim/imgs/f%.0f.png' % len(glob.glob('/home/raab/figures/field_anim/imgs/*.png')), dpi=300)

        if i != 0.5:
            combsig_handle.remove()

    beat_t = np.arange(0, 2.01, 0.01)
    beat = []
    for t0 in tqdm(beat_t):
        beat.append(np.max((f1+f2*i)[(t > t0 - 0.005) & (t < t0 + 0.005)]))
    beat_handle, = ax.plot(beat_t, np.array(beat) + 0.1, color='firebrick', lw=2, alpha=0)

    env_t = np.arange(0, 2.25, .25)
    env = []
    for t0 in tqdm(env_t):
        env.append(np.max((f1+f2*i)[(t > t0 - 0.25) & (t < t0 + 0.25)]))
    env_handle, = ax.plot(env_t, np.array(env) + 0.2, color='blue', lw = 2, alpha = 0)

    for i in np.linspace(0, 1, 50):
        beat_handle.set_alpha(i)
        env_handle.set_alpha(i)
        if draw:
            fig.canvas.draw()
            sleep(1 / 120)
        if save:
            plt.savefig('/home/raab/figures/field_anim/imgs/f%.0f.png' % len(glob.glob('/home/raab/figures/field_anim/imgs/*.png')), dpi=300)

    if draw:
        plt.show()

    quit()
    ##################
    ##################
    ##################

    fig = plt.figure(figsize=(20/2.54, 12/2.54), facecolor='white')

    gs = gridspec.GridSpec(5, 2)
    gs.update(left=0.1, right=0.95, bottom=.1, top=.95)

    axf = plt.subplot(gs[3:, 0])
    axf.axis('off')
    axff = plt.subplot(gs[3:, 1])
    axff.axis('off')


    fish_pic_left = axf.imshow(field_neu_inv)
    fish_pic_right = axff.imshow(field_neu)

    ax = plt.subplot(gs[:3, :])
    ax.set_xlim([0, 2])
    ax.set_xticks([0, 1, 2])

    ax.set_ylim([-2, 2])
    ax.set_yticks(np.arange(-2, 2.1, 1))

    ax.set_ylabel('Amplitude [mV]', fontsize=10)
    ax.set_xlabel('Time [s]', fontsize=10)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')


    combsig_handle, = ax.plot(t, f1 + f2 * 0.5, color='dimgrey', alpha=1)
    beat_handle, = ax.plot(beat_t, np.array(beat) + 0.1, color='firebrick', lw=2, alpha=1)
    env_handle, = ax.plot(env_t, np.array(env) + 0.2, color='blue', lw = 2, alpha = 1)

    f1_handle, = ax.plot(t, f1, color='forestgreen', alpha = 0)
    f2_handle, = ax.plot(t, f2 * 0.5, color='cornflowerblue', alpha = 0)

    if draw:
        plt.show(block=False)
        sleep(2)

    fish_pic_left.set_data(field_pos_inv)
    fish_pic_right.set_data(field_pos)
    for x0, x1 in zip(np.linspace(0, 0.99, 50), np.linspace(2, 1.01, 50)):
        ax.set_xlim([x0, x1])
        if draw:
            fig.canvas.draw()
            sleep(1/ 120)
        if save:
            plt.savefig('/home/raab/figures/field_anim/imgs/f%.0f.png' % len(glob.glob('/home/raab/figures/field_anim/imgs/*.png')), dpi=300)

    for i in np.linspace(0, 1, 50):
        f1_handle.set_alpha(i)
        f2_handle.set_alpha(i)
        if draw:
            fig.canvas.draw()
            sleep(1/120)
        if save:
            plt.savefig('/home/raab/figures/field_anim/imgs/f%.0f.png' % len(glob.glob('/home/raab/figures/field_anim/imgs/*.png')), dpi=300)

    for i in np.linspace(1, 0, 50):
        f1_handle.set_alpha(i)
        f2_handle.set_alpha(i)
        if draw:
            fig.canvas.draw()
            sleep(1/120)
        if save:
            plt.savefig('/home/raab/figures/field_anim/imgs/f%.0f.png' % len(glob.glob('/home/raab/figures/field_anim/imgs/*.png')), dpi=300)

    for x0, x1 in zip(np.linspace(0.99, 0.75, 50), np.linspace(1.01, 1.25, 50)):
        ax.set_xlim([x0, x1])
        if draw:
            fig.canvas.draw()
            sleep(1/ 120)
        if save:
            plt.savefig('/home/raab/figures/field_anim/imgs/f%.0f.png' % len(glob.glob('/home/raab/figures/field_anim/imgs/*.png')), dpi=300)

    fish_pic_left.set_data(field_pos_inv)
    fish_pic_right.set_data(field_neg)

    for x0, x1 in zip(np.linspace(0.75, 1.115, 50), np.linspace(1.25, 1.135, 50)):
        ax.set_xlim([x0, x1])
        if draw:
            fig.canvas.draw()
            sleep(1/ 120)
        if save:
            plt.savefig('/home/raab/figures/field_anim/imgs/f%.0f.png' % len(glob.glob('/home/raab/figures/field_anim/imgs/*.png')), dpi=300)

    for i in np.linspace(0, 1, 50):
        f1_handle.set_alpha(i)
        f2_handle.set_alpha(i)
        if draw:
            fig.canvas.draw()
            sleep(1/120)
        if save:
            plt.savefig('/home/raab/figures/field_anim/imgs/f%.0f.png' % len(glob.glob('/home/raab/figures/field_anim/imgs/*.png')), dpi=300)

    for i in np.linspace(1, 0, 50):
        f1_handle.set_alpha(i)
        f2_handle.set_alpha(i)
        if draw:
            fig.canvas.draw()
            sleep(1/120)
        if save:
            plt.savefig('/home/raab/figures/field_anim/imgs/f%.0f.png' % len(glob.glob('/home/raab/figures/field_anim/imgs/*.png')), dpi=300)

    for x0, x1 in zip(np.linspace(1.115, 0, 50), np.linspace(1.135, 2, 50)):
        ax.set_xlim([x0, x1])
        if draw:
            fig.canvas.draw()
            sleep(1/ 120)
        if save:
            plt.savefig('/home/raab/figures/field_anim/imgs/f%.0f.png' % len(glob.glob('/home/raab/figures/field_anim/imgs/*.png')), dpi=300)

    plt.show()


    ###################
    ##################
    ###################

    print('stage 4')
    # running and beat/envelope intro
    alpha = 0
    for x0, x1 in zip(np.linspace(0, 10, 200), np.linspace(2, 12, 200)):


        ii = np.where(t < x1)[0][-1]
        dist_dot, = ax_d_amp.plot(a2[ii], dist[ii], 'o', color='blue')

        # ax.set_xlim(x0, x1)
        combsig_handle, = ax.plot((t[(t >= x0) & (t <= x1)]- x0) / (x1 - x0), comb_sig[(t >= x0) & (t <= x1)], color='dimgrey', alpha=1)

        # ToDo: upgrade this too !!!
        # ax_dist.set_xlim([x1 - np.diff([x0, x1]) / 2, x1])
        # fig.canvas.draw()

        if x0 > 3:
            if alpha < 1:
                alpha += .01
            beat_handle, = ax.plot((beat_t[(beat_t >= x0 - beat_t[1] - beat_t[0]) & (beat_t <= x1 + beat_t[1] - beat_t[0])]- x0) / (x1 - x0),
                                   np.array(beat)[(beat_t >= x0 - beat_t[1] - beat_t[0]) & (beat_t <= x1 + beat_t[1] - beat_t[0])] + 0.1, color='firebrick', lw=2, alpha=alpha)
            env_handle, = ax.plot((env_t[(env_t >= x0 - env_t[1] - env_t[0]) & (env_t <= x1 + env_t[1] - env_t[0])] - x0) / (x1 - x0),
                                  np.array(env)[(env_t >= x0 - env_t[1] - env_t[0]) & (env_t <= x1 + env_t[1] - env_t[0])] + 0.3, color='blue', lw=2, alpha=alpha)

            anim_handles.append([beat_handle, env_handle, dist_dot, combsig_handle])
        else:
            anim_handles.append([dist_dot, combsig_handle])
        # if x0 > 3:
        #     beat_handle.remove()
        #     env_handle.remove()
        # combsig_handle.remove()
        # dist_dot.remove()
        # sleep(1/120)

    # sleep(2)


    # for i in range(20):
    #     anim_handles.append([combsig_handle])



    ####################################################################################################################
    print('stage 5')

    # zoom in -- in phase
    fish_pic_left = ax_f1.imshow(field_pos_inv)
    # fish_pic_left.set_data(field_pos_inv)
    fish_pic_right = ax_f2.imshow(field_pos)
    anim_handles.append([fish_pic_right, fish_pic_left])
    # fish_pic_right.set_data(field_pos)
    for x0, x1 in zip(np.linspace(10, 10.99, 100), np.linspace(12, 11.01, 100)):
        # ax.set_xlim(x0, x1)
        combsig_handle, = ax.plot(t[(t >= x0) & (t <= x1)], comb_sig[(t >= x0) & (t <= x1)], color='dimgrey', alpha=1)
        beat_handle, = ax.plot(beat_t[(beat_t >= x0 - beat_t[1] - beat_t[0]) & (beat_t <= x1 + beat_t[1] - beat_t[0])],
                               np.array(beat)[(beat_t >= x0 - beat_t[1] - beat_t[0]) & (beat_t <= x1 + beat_t[1] - beat_t[0])] + 0.1, color='firebrick', lw=2)
        env_handle, = ax.plot(env_t[(env_t >= x0 - env_t[1] - env_t[0]) & (env_t <= x1 + env_t[1] - env_t[0])],
                              np.array(env)[(env_t >= x0 - env_t[1] - env_t[0]) & (env_t <= x1 + env_t[1] - env_t[0])] + 0.3, color='blue', lw=2)
        ii = np.where(t < x1)[0][-1]
        dist_dot, = ax_d_amp.plot(a2[ii], dist[ii], 'o', color='blue')

        ax_dist.set_xlim([x1 - np.diff([x0, x1]) / 2, x1])

        anim_handles.append([combsig_handle, beat_handle, env_handle, dist_dot])
        # fig.canvas.draw()
        # combsig_handle.remove()
        # beat_handle.remove()
        # env_handle.remove()
        # dist_dot.remove()
        # sleep(1 / 120)

    # sleep(2)

    # for i in range(20):
    #     anim_handles.append([combsig_handle])


    ###################################################################################################################

    # show init traces (2)
    combsig_handle, = ax.plot(t[(t >= 10.98) & (t <= 11.02)], comb_sig[(t >= 10.98) & (t <= 11.02)], color='dimgrey', alpha=1)
    beat_handle, = ax.plot(beat_t[(beat_t >= x0 - beat_t[1] - beat_t[0]) & (beat_t <= x1 + beat_t[1] - beat_t[0])],
                           np.array(beat)[(beat_t >= x0 - beat_t[1] - beat_t[0]) & (beat_t <= x1 + beat_t[1] - beat_t[0])] + 0.1,color='firebrick', lw=2)
    env_handle, = ax.plot(env_t[(env_t >= x0 - env_t[1] - env_t[0]) & (env_t <= x1 + env_t[1] - env_t[0])],
                          np.array(env)[(env_t >= x0 - env_t[1] - env_t[0]) & (env_t <= x1 + env_t[1] - env_t[0])] + 0.3, color='blue', lw=2)
    ii = np.where(t < x1)[0][-1]
    dist_dot, = ax_d_amp.plot(a2[ii], dist[ii], 'o', color='blue')


    for i in np.arange(0, 1.02, 0.02):
        f1_handle, = ax.plot(t[(t >= 10.98) & (t <= 11.02)], f1[(t >= 10.98) & (t <= 11.02)], color='forestgreen', alpha=i)
        f2_handle, = ax.plot(t[(t >= 10.98) & (t <= 11.02)], f2[(t >= 10.98) & (t <= 11.02)] * a2[(t >= 10.98) & (t <= 11.02)], color='cornflowerblue', alpha=i)
        # fig.canvas.draw()

        anim_handles.append(f1_handle, f2_handle)

        # f1_handle.remove()
        # f2_handle.remove()
        # sleep(1/120)

    # hide init traces (2)
    sleep(2)
    combsig_handle, = ax.plot(t[(t >= 10.98) & (t <= 11.02)], comb_sig[(t >= 10.98) & (t <= 11.02)], color='dimgrey', alpha=1)
    beat_handle, = ax.plot(beat_t[(beat_t >= x0 - beat_t[1] - beat_t[0]) & (beat_t <= x1 + beat_t[1] - beat_t[0])],
                           np.array(beat)[(beat_t >= x0 - beat_t[1] - beat_t[0]) & (beat_t <= x1 + beat_t[1] - beat_t[0])] + 0.1,color='firebrick', lw=2)
    env_handle, = ax.plot(env_t[(env_t >= x0 - env_t[1] - env_t[0]) & (env_t <= x1 + env_t[1] - env_t[0])],
                          np.array(env)[(env_t >= x0 - env_t[1] - env_t[0]) & (env_t <= x1 + env_t[1] - env_t[0])] + 0.3, color='blue', lw=2)
    for i in np.arange(1, -0.02, -0.02):
        f1_handle, = ax.plot(t[(t >= 10.98) & (t <= 11.02)], f1[(t >= 10.98) & (t <= 11.02)], color='forestgreen', alpha=i)
        f2_handle, = ax.plot(t[(t >= 10.98) & (t <= 11.02)], f2[(t >= 10.98) & (t <= 11.02)] * a2[(t >= 10.98) & (t <= 11.02)], color='cornflowerblue', alpha=i)
        fig.canvas.draw()

        f1_handle.remove()
        f2_handle.remove()
        sleep(1/120)

    dist_dot.remove()

    sleep(2)

    ####################################################################################################################
    # zoom out
    for x0, x1 in zip(np.linspace(10.99, 10.75, 25), np.linspace(11.01, 11.25, 25)):
        ax.set_xlim(x0, x1)
        combsig_handle, = ax.plot(t[(t >= x0) & (t <= x1)], comb_sig[(t >= x0) & (t <= x1)], color='dimgrey', alpha=1)
        beat_handle, = ax.plot(beat_t[(beat_t >= x0 - beat_t[1] - beat_t[0]) & (beat_t <= x1 + beat_t[1] - beat_t[0])],
                               np.array(beat)[(beat_t >= x0 - beat_t[1] - beat_t[0]) & (beat_t <= x1 + beat_t[1] - beat_t[0])] + 0.1, color='firebrick', lw=2)
        env_handle, = ax.plot(env_t[(env_t >= x0 - env_t[1] - env_t[0]) & (env_t <= x1 + env_t[1] - env_t[0])],
                              np.array(env)[(env_t >= x0 - env_t[1] - env_t[0]) & (env_t <= x1 + env_t[1] - env_t[0])] + 0.3, color='blue', lw=2)
        ii = np.where(t < x1)[0][-1]
        dist_dot, = ax_d_amp.plot(a2[ii], dist[ii], 'o', color='blue')

        ax_dist.set_xlim([x1 - np.diff([x0, x1]) / 2, x1])
        fig.canvas.draw()

        combsig_handle.remove()
        beat_handle.remove()
        env_handle.remove()
        dist_dot.remove()
        sleep(1 / 120)

    ####################################################################################################################
    # zoom in -- out phase
    sleep(2)
    fish_pic_left.set_data(field_pos_inv)
    fish_pic_right.set_data(field_neg)
    for x0, x1 in zip(np.linspace(10.75, 11.115, 25), np.linspace(11.25, 11.135, 25)):
        ax.set_xlim(x0, x1)
        combsig_handle, = ax.plot(t[(t >= x0) & (t <= x1)], comb_sig[(t >= x0) & (t <= x1)], color='dimgrey', alpha=1)
        beat_handle, = ax.plot(beat_t[(beat_t >= x0 - beat_t[1] - beat_t[0]) & (beat_t <= x1 + beat_t[1] - beat_t[0])],
                               np.array(beat)[(beat_t >= x0 - beat_t[1] - beat_t[0]) & (beat_t <= x1 + beat_t[1] - beat_t[0])] + 0.1, color='firebrick', lw=2)
        env_handle, = ax.plot(env_t[(env_t >= x0 - env_t[1] - env_t[0]) & (env_t <= x1 + env_t[1] - env_t[0])],
                              np.array(env)[(env_t >= x0 - env_t[1] - env_t[0]) & (env_t <= x1 + env_t[1] - env_t[0])] + 0.3, color='blue', lw=2)
        ii = np.where(t < x1)[0][-1]
        dist_dot, = ax_d_amp.plot(a2[ii], dist[ii], 'o', color='blue')

        ax_dist.set_xlim([x1 - np.diff([x0, x1]) / 2, x1])

        fig.canvas.draw()
        combsig_handle.remove()
        beat_handle.remove()
        env_handle.remove()
        dist_dot.remove()

        sleep(1 / 120)

    ####################################################################################################################
    # show init traces (2)
    combsig_handle, = ax.plot(t[(t >= x0) & (t <= x1)], comb_sig[(t >= x0) & (t <= x1)], color='dimgrey', alpha=1)
    beat_handle, = ax.plot(beat_t[(beat_t >= x0 - beat_t[1] - beat_t[0]) & (beat_t <= x1 + beat_t[1] - beat_t[0])],
                           np.array(beat)[(beat_t >= x0 - beat_t[1] - beat_t[0]) & (beat_t <= x1 + beat_t[1] - beat_t[0])] + 0.1, color='firebrick', lw=2)
    env_handle, = ax.plot(env_t[(env_t >= x0 - env_t[1] - env_t[0]) & (env_t <= x1 + env_t[1] - env_t[0])],
                          np.array(env)[(env_t >= x0 - env_t[1] - env_t[0]) & (env_t <= x1 + env_t[1] - env_t[0])] + 0.3, color='blue', lw=2)
    ii = np.where(t < x1)[0][-1]
    dist_dot, = ax_d_amp.plot(a2[ii], dist[ii], 'o', color='blue')
    sleep(2)
    for i in np.arange(0, 1.02, 0.02):
        f1_handle, = ax.plot(t[(t >= 11.105) & (t <= 11.145)], f1[(t >= 11.105) & (t <= 11.145)], color='forestgreen', alpha=i)
        f2_handle, = ax.plot(t[(t >= 11.105) & (t <= 11.145)], f2[(t >= 11.105) & (t <= 11.145)] * a2[(t >= 11.105) & (t <= 11.145)], color='cornflowerblue', alpha=i)
        fig.canvas.draw()

        f1_handle.remove()
        f2_handle.remove()
        sleep(1/120)

    # hide init traces (2)
    sleep(2)
    for i in np.arange(1, -0.02, -0.02):
        f1_handle, = ax.plot(t[(t >= 11.105) & (t <= 11.145)], f1[(t >= 11.105) & (t <= 11.145)], color='forestgreen', alpha=i)
        f2_handle, = ax.plot(t[(t >= 11.105) & (t <= 11.145)], f2[(t >= 11.105) & (t <= 11.145)] * a2[(t >= 11.105) & (t <= 11.145)], color='cornflowerblue', alpha=i)
        fig.canvas.draw()

        f1_handle.remove()
        f2_handle.remove()
        sleep(1/120)
    combsig_handle.remove()
    beat_handle.remove()
    env_handle.remove()
    dist_dot.remove()

    ####################################################################################################################
    # zoom out
    sleep(2)
    for x0, x1 in zip(np.linspace(11.115, 10, 25), np.linspace(11.135, 12, 25)):
        ax.set_xlim(x0, x1)
        combsig_handle, = ax.plot(t[(t >= x0) & (t <= x1)], comb_sig[(t >= x0) & (t <= x1)], color='dimgrey', alpha=1)
        beat_handle, = ax.plot(beat_t[(beat_t >= x0 - beat_t[1] - beat_t[0]) & (beat_t <= x1 + beat_t[1] - beat_t[0])],
                               np.array(beat)[(beat_t >= x0 - beat_t[1] - beat_t[0]) & (beat_t <= x1 + beat_t[1] - beat_t[0])] + 0.1, color='firebrick', lw=2)
        env_handle, = ax.plot(env_t[(env_t >= x0 - env_t[1] - env_t[0]) & (env_t <= x1 + env_t[1] - env_t[0])],
                              np.array(env)[(env_t >= x0 - env_t[1] - env_t[0]) & (env_t <= x1 + env_t[1] - env_t[0])] + 0.3, color='blue', lw=2)
        ii = np.where(t < x1)[0][-1]
        dist_dot, = ax_d_amp.plot(a2[ii], dist[ii], 'o', color='blue')

        ax_dist.set_xlim([x1 - np.diff([x0, x1]) / 2, x1])
        fig.canvas.draw()
        combsig_handle.remove()
        beat_handle.remove()
        env_handle.remove()
        dist_dot.remove()
        sleep(1 / 120)

    fish_pic_left.set_data(field_neu_inv)
    fish_pic_right.set_data(field_neu)
    fig.canvas.draw()

    ####################################################################################################################
    # zoom out + run
    sleep(2)
    for x0, x1 in zip(np.linspace(10, 12, 25), np.linspace(12, 20, 25)):
        ax.set_xlim(x0, x1)
        combsig_handle, = ax.plot(t[(t >= x0) & (t <= x1)], comb_sig[(t >= x0) & (t <= x1)], color='dimgrey', alpha=1)
        beat_handle, = ax.plot(beat_t[(beat_t >= x0 - beat_t[1] - beat_t[0]) & (beat_t <= x1 + beat_t[1] - beat_t[0])],
                               np.array(beat)[(beat_t >= x0 - beat_t[1] - beat_t[0]) & (beat_t <= x1 + beat_t[1] - beat_t[0])] + 0.1, color='firebrick', lw=2)
        env_handle, = ax.plot(env_t[(env_t >= x0 - env_t[1] - env_t[0]) & (env_t <= x1 + env_t[1] - env_t[0])],
                              np.array(env)[(env_t >= x0 - env_t[1] - env_t[0]) & (env_t <= x1 + env_t[1] - env_t[0])] + 0.3, color='blue', lw=2)
        ii = np.where(t < x1)[0][-1]
        dist_dot, = ax_d_amp.plot(a2[ii], dist[ii], 'o', color='blue')

        ax_dist.set_xlim([x1 - np.diff([x0, x1]) / 2, x1])
        fig.canvas.draw()
        combsig_handle.remove()
        beat_handle.remove()
        env_handle.remove()
        dist_dot.remove()
        sleep(1 / 120)

    ####################################################################################################################
    # run
    sleep(2)
    for x0, x1 in zip(np.linspace(12, 37, 100), np.linspace(20, 45, 100)):
        ax.set_xlim(x0, x1)
        combsig_handle, = ax.plot(t[(t >= x0) & (t <= x1)], comb_sig[(t >= x0) & (t <= x1)], color='dimgrey', alpha=1)
        beat_handle, = ax.plot(beat_t[(beat_t >= x0 - beat_t[1] - beat_t[0]) & (beat_t <= x1 + beat_t[1] - beat_t[0])],
                               np.array(beat)[(beat_t >= x0 - beat_t[1] - beat_t[0]) & (beat_t <= x1 + beat_t[1] - beat_t[0])] + 0.1, color='firebrick', lw=2)
        env_handle, = ax.plot(env_t[(env_t >= x0 - env_t[1] - env_t[0]) & (env_t <= x1 + env_t[1] - env_t[0])],
                              np.array(env)[(env_t >= x0 - env_t[1] - env_t[0]) & (env_t <= x1 + env_t[1] - env_t[0])] + 0.3, color='blue', lw=2)
        ii = np.where(t < x1)[0][-1]
        dist_dot, = ax_d_amp.plot(a2[ii], dist[ii], 'o', color='blue')

        ax_dist.set_xlim([x1 - np.diff([x0, x1]) / 2, x1])
        fig.canvas.draw()
        combsig_handle.remove()
        beat_handle.remove()
        env_handle.remove()
        dist_dot.remove()
        sleep(1 / 120)

    plt.show()
    embed()
    quit()



if __name__ == '__main__':
    main()