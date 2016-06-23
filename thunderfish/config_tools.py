from collections import OrderedDict


def get_config_dict():
    cfg = OrderedDict()
    cfgsec = dict()

    cfgsec['minPSDAverages'] = 'Power spectrum estimation:'
    cfg['minPSDAverages'] = [3, '', 'Minimum number of fft averages for estimating the power spectrum.']
    cfg['initialFrequencyResolution'] = [1.0, 'Hz', 'Initial frequency resolution of the power spectrum.']

    cfgsec['lowThreshold'] = 'Thresholds for peak detection in power spectra:'
    cfg['lowThreshold'] = [0.0, 'dB', 'Threshold for all peaks.\n If 0.0 estimate threshold from histogram.']
    cfg['highThreshold'] = [0.0, 'dB', 'Threshold for good peaks. If 0.0 estimate threshold from histogram.']
    # cfg['lowThreshold'][0] = 12. # panama
    # cfg['highThreshold'][0] = 18. # panama

    cfgsec[
        'noiseFactor'] = 'Threshold estimation:\nIf no thresholds are specified they are estimated from the histogram of the decibel power spectrum.'
    cfg['noiseFactor'] = [6.0, '', 'Factor for multiplying std of noise floor for lower threshold.']
    cfg['peakFactor'] = [0.5, '', 'Fractional position of upper threshold above lower threshold.']

    cfgsec['maxPeakWidthFac'] = 'Peak detection in decibel power spectrum:'
    cfg['maxPeakWidthFac'] = [3.5, '',
                              'Maximum width of peaks at 0.75 hight in multiples of frequency resolution (might be increased)']
    cfg['minPeakWidth'] = [1.0, 'Hz', 'Peaks do not need to be narrower than this.']

    cfgsec['mainsFreq'] = 'Harmonic groups:'
    cfg['mainsFreq'] = [60.0, 'Hz', 'Mains frequency to be excluded.']
    cfg['maxDivisor'] = [4, '', 'Maximum ratio between the frequency of the largest peak and its fundamental']
    cfg['freqTolerance'] = [0.7, '',
                            'Harmonics need be within this factor times the frequency resolution of the power spectrum. Needs to be higher than 0.5!']
    cfg['maxUpperFill'] = [1, '',
                           'As soon as more than this number of harmonics need to be filled in conescutively stop searching for higher harmonics.']
    cfg['maxFillRatio'] = [0.25, '',
                           'Maximum fraction of filled in harmonics allowed (usefull values are smaller than 0.5)']
    cfg['maxDoubleUseHarmonics'] = [8, '', 'Maximum harmonics up to which double uses are penalized.']
    cfg['maxDoubleUseCount'] = [1, '', 'Maximum overall double use count allowed.']
    cfg['powerNHarmonics'] = [10, '', 'Compute total power over the first # harmonics.']

    cfgsec['minimumGroupSize'] = 'Acceptance of best harmonic groups:'
    cfg['minimumGroupSize'] = [3, '',
                               'Minimum required number of harmonics (inclusively fundamental) that are not filled in and are not used by other groups.']
    # cfg['minimumGroupSize'][0] = 2 # panama
    cfg['minimumFrequency'] = [20.0, 'Hz', 'Minimum frequency allowed for the fundamental.']
    cfg['maximumFrequency'] = [2000.0, 'Hz', 'Maximum frequency allowed for the fundamental.']
    cfg['maximumWorkingFrequency'] = [4000.0, 'Hz',
                                      'Maximum frequency to be used to search for harmonic groups and to adjust fundamental frequency.']

    cfg['maxHarmonics'] = [0, '', '0: keep all, >0 only keep the first # harmonics.']

    cfgsec['displayHelp'] = 'Items to display:'
    cfg['displayHelp'] = [False, '', 'Display help on key bindings']
    cfg['labelFrequency'] = [True, '', 'Display the frequency of the peak']
    cfg['labelHarmonic'] = [True, '', 'Display the harmonic of the peak']
    cfg['labelPower'] = [True, '', 'Display the power of the peak']
    cfg['labelWidth'] = [True, '', 'Display the width of the peak']
    cfg['labelDoubleUse'] = [True, '', 'Display double-use count of the peak']

    cfgsec['verboseLevel'] = 'Debugging:'
    cfg['verboseLevel'] = [0, '', '0=off upto 4 very detailed']
    return cfg
