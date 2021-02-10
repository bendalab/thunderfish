"""
Check for pulse-type weakly electric fish

Check whether a pulse-type or a wave-type weakly electric fish is present in a recording.

## Main functions

- `check_pulse()`: checks for pulse-type fish.

## Configuration parameter

- `add_check_pulse_config()`: add parameters for `check_pulse()` to configuration.
- `check_pulse_args()`: retrieve parameters for `check_pulse()` from configuration.

"""

import numpy as np
from .eventdetection import percentile_threshold, detect_peaks, trim


def check_pulse(data, sem, samplerate, thresh_fac=0.8, percentile=0.0,
                sem_fac=0.05, pulse_thresh=0.15, verbose=0):
    """Detects if a fish is pulse- or wave-type based on the proportion of the time distance
    between a peak and its following trough, relative to the time between consecutive peaks.


    Parameters
    ----------
    data: 1-D array
         The data to be analyzed.
    sem: 1-D array or None
         Standard error of the mean corresponding to data.
    samplerate: float
         Sampling rate of the data in Hertz.
    percentile: float
         The interpercentile range is computed at percentile and 100.0-percentile.
         If zero take maxmimum minus minimum.
    thresh_fac: float
         The threshold for peak detection is the inter-percentile-range
         multiplied by this factor.
    sem_fac: float
         Base analysis on peaks and troughs with the corresponding standard error of the mean
         of the data below `sem_fac` times the interpercentile range.
    pulse_thresh: float
         A positive number setting the minimum distance between peaks and troughs.
    verbose: int
         If > 1, print information in the command line.

    Returns
    -------
    pulse_fish: bool
        True if algorithm suggests a pulse-type fish.
    peak_ratio: float
        Returns a float between 0. and 1. which gives the proportion of peak-2-trough,
        from peak-2-peak time distance, i.e. pulse width relative to pulse interval.
    """

    def ratio(peak_idx, trough_idx):
        if len(peak_idx) < 2:
            return 1.0 if len(peak_idx)+len(trough_idx) < 1 else 0.0
        # ratio of peak-to-trough to peak-to-peak time distances:
        ratios = np.abs((trough_idx - peak_idx))[:-1].astype(np.float) / np.diff(peak_idx)
        # fix for cases where trough of eod comes before peak:
        ratios[ratios > 0.5] = 1.0 - ratios[ratios > 0.5]
        return np.median(ratios)

    # threshold for peak detection:
    pp_ampl = percentile_threshold(data, thresh_fac=1.0, percentile=percentile)
    threshold = thresh_fac*pp_ampl
    if verbose > 1:
        print('  check_pulse(): amplitude threshold for peak detection is %g' % threshold)
    # detect large peaks and troughs:
    peak_idx, trough_idx = detect_peaks(data, threshold)
    # XXX check here for less than 2 peaks + troughs XXX
    if verbose > 1:
        print('  check_pulse(): detected %d peaks %d troughs' %
              (len(peak_idx), len(trough_idx)))
    peak_idx, trough_idx = trim(peak_idx, trough_idx)
    if verbose > 1:
        print('  check_pulse(): trim %d peaks %d troughs' %
              (len(peak_idx), len(trough_idx)))
    # take only peaks and troughs of reliable sem:
    if sem is not None:
        athresh = sem_fac*pp_ampl
        sel = (sem[peak_idx]<athresh) & (sem[trough_idx]<athresh)
        peak_idx = peak_idx[sel]
        trough_idx = trough_idx[sel]
        if verbose > 1:
            print('  check_pulse(): sem left %d peaks %d troughs' %
                  (len(peak_idx), len(trough_idx)))
    # compute ratios:
    peak_ratio = np.mean([ratio(peak_idx, trough_idx), ratio(trough_idx, peak_idx)])
    pulse_fish = peak_ratio < pulse_thresh
    if verbose > 0:
        print('  check_pulse(): classified as %s fish: pulse-width-ratio is %.3f' %
              ('pulse' if pulse_fish else 'wave', peak_ratio))
    return pulse_fish, peak_ratio


def add_check_pulse_config(cfg, thresh_fac=0.4, percentile=0.0,
                           sem_fac=0.05, pulse_thresh=0.1):
    """ Add parameter needed for `check_pulse()` as a new section to a configuration.

    Parameters
    ----------
    cfg: ConfigFile
        The configuration.
    See check_pulse() for details on the remaining arguments.
    """

    cfg.add_section('Classify waveforms:')
    cfg.add('pulseWidthPercentile', percentile, '%', 'Threshold for detecing peaks is based on interpercentile range. If zero use maximum minus minimum amplitude.')
    cfg.add('pulseWidthThresholdFactor', thresh_fac, '', 'The threshold for peak detection is this factor multiplied with the interpercentile range. Should be smaller than 0.5 if only a single pulse is contained in the data.')
    cfg.add('pulseWidthSEMFactor', sem_fac, '', 'Peaks and troughs where the standarad error of the mean of the data is larger than the interpercentile range times this factor are discarded from the analysis.')
    cfg.add('pulseWidthThresholdRatio', pulse_thresh, '', 'A pulsefish is detected if the width of the pulses relative to the intervals is smaller than this threshold.')


def check_pulse_args(cfg):
    """ Translates a configuration to the
    respective parameter names of the function `check_pulse()`.
    The return value can then be passed as key-word arguments to this function.

    Parameters
    ----------
    cfg: ConfigFile
        The configuration.

    Returns
    -------
    a: dict
        Dictionary with names of arguments of the check_pulse() function
        and their values as supplied by `cfg`.
    """
    a = cfg.map({'thresh_fac': 'pulseWidthThresholdFactor',
                 'percentile': 'pulseWidthPercentile',
                 'sem_fac': 'pulseWidthSEMFactor',
                 'pulse_thresh': 'pulseWidthThresholdRatio'})
    return a

    
if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt
    from .bestwindow import best_window
    from .fakefish import wavefish_eods, pulsefish_eods

    print("\nChecking checkpulse module ...\n")
    
    # generate data:
    rate = 44100.0
    if len(sys.argv) < 2:
        data = wavefish_eods('Eigenmannia', 80.0, rate, 8.0)
    elif sys.argv[1] == '-w':
        data = wavefish_eods('Alepto', 600.0, rate, 8.0)
    elif sys.argv[1] == '-m':
        data = pulsefish_eods('monophasic', 80.0, rate, 8.0)
    elif sys.argv[1] == '-b':
        data = pulsefish_eods('biphasic', 80.0, rate, 8.0)
    elif sys.argv[1] == '-t':
        data = pulsefish_eods('triphasic', 80.0, rate, 8.0)
    else:  # load data given by the user
        from .dataloader import load_data

        file_path = sys.argv[1]
        print("loading %s ...\n" % file_path)
        rawdata, rate, unit = load_data(sys.argv[1], 0)
        data, _ = best_window(rawdata, rate)
        
    # run pulse-width-based detector:
    pulse_fish, ratio = check_pulse(data, None, rate)
