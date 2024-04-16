import pytest
import thunderfish.configfile as cf
import thunderfish.bestwindow as bw
import os


def test_config_file():
    cfg = cf.ConfigFile()
    bw.add_clip_config(cfg)
    bw.add_best_window_config(cfg)

    bwa = bw.best_window_args(cfg)

    cfgfile = 'test.cfg'
    cfgdifffile = 'testdiff.cfg'

    # manipulate some values:
    cfg2 = cf.ConfigFile(cfg)
    cfg2.set('windowSize', 100.0)
    cfg2.set('weightCVAmplitude', 20.0)
    cfg2.set('clipBins', 300)
    cfg3 = cf.ConfigFile(cfg2)

    assert 'windowSize' in cfg2, '__contains__'
    assert len(cfg2['windowSize']) == 4, '__getitem__'

    with pytest.raises(IndexError):
        cfg2.set('xyz', 20)

    # write configurations to files:
    cfg.dump(cfgfile, 'header', maxline=50)
    cfg2.dump(cfgdifffile, diff_only=True)

    # test modified configuration:
    assert cfg != cfg2, 'cfg and cfg2 should differ'

    # read it in:
    cfg2.load(cfgfile)
    assert cfg == cfg2, 'cfg and cfg2 should be the same'

    # read manipulated values:
    cfg2.load(cfgdifffile)
    assert cfg2 == cfg3, 'cfg2 and cfg3 should be the same'

    # read it in:
    cfg3.load_files(cfgfile, 'data.dat', verbose=10)
    assert cfg == cfg3, 'cfg and cfg3 should be the same'

    # clean up:
    os.remove(cfgfile)
    os.remove(cfgdifffile)


def test_main():
    cf.main()

