from nose.tools import assert_equal, assert_not_equal
from copy import deepcopy
import thunderfish.configfile as cf
import thunderfish.bestwindow as bw
from collections import OrderedDict
import os


def test_config_file():
    cfg = cf.ConfigFile()
    bw.add_clip_config(cfg)
    bw.add_best_window_config(cfg)

    cfgfile = 'test.cfg'
    cfgdifffile = 'testdiff.cfg'

    # manipulate some values:
    cfg2 = deepcopy(cfg)
    cfg2.set('bestWindowSize', 100.0)
    cfg2.set('weightCVAmplitude', 20.0)
    cfg2.set('clipBins', 300)
    cfg3 = deepcopy(cfg2)

    # write configurations to files:
    cfg.dump(cfgfile, 'header', maxline=50)
    cfg2.dump(cfgdifffile, diff_only=True)

    # test modified configuration:
    assert_not_equal(cfg, cfg2, 'cfg and cfg2 should differ')

    # read it in:
    cfg2.load(cfgfile)
    assert_equal(cfg, cfg2, 'cfg and cfg2 should be the same')

    # read manipulated values:
    cfg2.load(cfgdifffile)
    assert_equal(cfg2, cfg3, 'cfg2 and cfg3 should be the same')

    # read it in:
    cfg3.load_files(cfgfile, 'data.dat')
    assert_equal(cfg, cfg3, 'cfg and cfg3 should be the same')

    # clean up:
    os.remove(cfgfile)
    os.remove(cfgdifffile)
