import pytest
import os
import numpy as np
import matplotlib.pyplot as plt
from thunderlab.configfile import ConfigFile
import thunderfish.bestwindow as bw


def test_best_window():
    # generate data:
    rate = 100000.0
    clip = 1.3
    time = np.arange(0.0, 1.0, 1.0 / rate)
    snippets = []
    f = 600.0
    amf = 20.0
    for ampl in [0.2, 0.5, 0.8]:
        for am_ampl in [0.0, 0.3, 0.9]:
            data = ampl * np.sin(2.0 * np.pi * f * time) * (1.0 + am_ampl * np.sin(2.0 * np.pi * amf * time))
            data[data > clip] = clip
            data[data < -clip] = -clip
            snippets.extend(data)
    data = np.asarray(snippets)

    # compute best window:
    print("call bestwindow() function...")
    idx0, idx1, clipped = bw.best_window_indices(data, rate, expand=False,
                                                 win_size=1.0, win_shift=0.1,
                                                 min_clip=-clip, max_clip=clip,
                                                 w_cv_ampl=10.0, tolerance=0.5)

    assert idx0 == 6 * len(time), 'bestwindow() did not correctly detect start of best window'
    assert idx1 == 7 * len(time), 'bestwindow() did not correctly detect end of best window'
    assert clipped == pytest.approx(0.0), 'bestwindow() did not correctly detect clipped fraction'

    t0, t1, clipped = bw.best_window_times(data, rate, expand=False,
                                           win_size=1.0, win_shift=0.1,
                                           min_clip=-clip, max_clip=clip,
                                           w_cv_ampl=10.0, tolerance=0.5)

    bdata, clipped = bw.best_window(data, rate, expand=False,
                                    win_size=1.0, win_shift=0.1,
                                    min_clip=-clip, max_clip=clip,
                                    w_cv_ampl=10.0, tolerance=0.5)


    cfg = ConfigFile()
    bw.add_clip_config(cfg)
    bw.add_best_window_config(cfg)
    cfg.add('unwrapData', False, '', 'unwrap clipped data') 
    for win_pos in ['beginning', 'center', 'end', 'best', '0.1s', 'xxx']:
        bw.analysis_window(data, rate, clip, win_pos, cfg,
                           show_bestwindow=False)
    bw.analysis_window(data, rate, clip, 'best', cfg, show_bestwindow=True)
    

    # clipping:
    clip_win_size = 0.5
    min_clip, max_clip = bw.clip_amplitudes(data, int(clip_win_size * rate),
                                            min_ampl=-1.3, max_ampl=1.3,
                                            min_fac=2.0, nbins=40)

    assert min_clip <= -0.8 * clip and min_clip >= -clip, 'clip_amplitudes() failed to detect minimum clip amplitude'
    assert max_clip >= 0.8 * clip and max_clip <= clip, 'clip_amplitudes() failed to detect maximum clip amplitude'

    # plotting 1:
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    bw.plot_data_window(ax, data, rate, 'a.u.', idx0, idx1, clipped)
    fig.savefig('bestwindow.png')
    assert os.path.exists('bestwindow.png'), 'plotting failed'
    os.remove('bestwindow.png')

    # plotting 2:
    fig, ax = plt.subplots(5, sharex=True)
    bw.best_window_indices(data, rate, expand=False,
                           win_size=1.0, win_shift=0.1,
                           min_clip=-clip, max_clip=clip,
                           w_cv_ampl=10.0, tolerance=0.5,
                           plot_data_func=bw.plot_best_window, ax=ax)
    fig.savefig('bestdata.png')
    assert os.path.exists('bestdata.png'), 'plotting failed'
    os.remove('bestdata.png')
    

def test_bestwindow_main():
    bw.main()
