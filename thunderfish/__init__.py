"""
Algorithms and programs for analysing electric field recordings of
weakly electric fish.
"""

import sys

# avoid double inclusion of audioio modules if called as modules,
# e.g. python -m thunderfish.datawriter`:
if len(sys.argv) > 0 and sys.argv[0] != '-m':

    from .version import __version__

    # somehow pdoc3 gets confused by this:
    #__all__ = ['thunderfish',
    #           'dataloader',
    #           'datawriter',
    #           'tabledata',
    #           'configfile',
    #           'eventdetection',
    #           'bestwindow',
    #           'powerspectrum',
    #           'harmonics',
    #           'checkpulse',
    #           'consistentfishes',
    #           'eodanalysis',
    #           'voronoi',
    #           'fakefish']
