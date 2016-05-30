from nose.tools import assert_equal
import thunderfish.configfile as cf
import thunderfish.bestwindow as bw
from collections import OrderedDict
import os


def test_config_file():
    cfg = cf.ConfigFile()
    bw.add_clip_config(cfg)
    bw.add_best_window_config(cfg)

    cfgfile = 'test.cfg'

    # write configuration to a file:
    cfg.dump(cfgfile, 'header', 50)

    # manipulate some values:
    cfg2 = cf.ConfigFile(cfg)
    cfg2.set('bestWindowSize', 100.0)
    cfg2.set('weightCVAmplitude', 20.0)
    cfg2.set('clipBins', 300)
    cfg3 = cf.ConfigFile(cfg2)
        
    # read it in:
    cfg2.load(cfgfile)
    assert_equal(cfg, cfg2)
            
    # read it in:
    cfg3.load_files(cfgfile, 'data.dat')
    assert_equal(cfg, cfg3)

    # clean up:
    os.remove(cfgfile)
