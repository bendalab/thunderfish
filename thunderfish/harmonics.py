"""
Extract and analyze harmonic frequencies from power spectra.

## Harmonic group extraction

- `harmonic_groups()`: detect peaks in power spectrum and group them
                       according to their harmonic structure.
- `expand_group()`: add more harmonics to harmonic group. 
- `extract_fundamentals()`: collect harmonic groups from
                            lists of power spectrum peaks.
- `threshold_estimate()`: estimates thresholds for peak detection
                          in a power spectrum.
                            
## Helper functions for harmonic group extraction

- `build_harmonic_group()`: find all the harmonics belonging to the largest peak in the power spectrum.
- `retrieve_harmonic_group()`: find all the harmonics belonging to a given fundamental.
- `group_candidate()`: candidate harmonic frequencies belonging to a fundamental frequency.
- `update_group()`: update frequency lists and harmonic group.

## Handling of lists of harmonic groups

- `fundamental_freqs()`: extract fundamental frequencies from
                         lists of harmonic groups.
- `fundamental_freqs_and_power()`: extract fundamental frequencies and their
                                   power in dB from lists of harmonic groups.

## Handling of lists of fundamental frequencies

- `add_relative_power()`: add a column with relative power.
- `add_power_ranks()`: add a column with power ranks.
- `similar_indices()`: indices of similar frequencies.
- `unique_mask()`: mark similar frequencies from different recordings as dublicate.
- `unique()`: remove similar frequencies from different recordings.

## Visualization

- `colors_markers()`: Generate a list of colors and markers for plotting.
- `plot_harmonic_groups()`: Mark decibel power of fundamentals and their
                            harmonics.
- `plot_psd_harmonic_groups()`: Plot decibel power-spectrum with detected peaks,
                                harmonic groups, and mains frequencies.

## Configuration

- `add_psd_peak_detection_config()`: add parameters for the detection of
                                     peaks in power spectra to configuration.
- `psd_peak_detection_args()`: retrieve parameters for the detection of peaks
                               in power spectra from configuration.
- `add_harmonic_groups_config()`: add parameters for the detection of
                                  harmonic groups to configuration.
- `harmonic_groups_args()`: retrieve parameters for the detection of
                            harmonic groups from configuration.
"""

from __future__ import print_function
import math as m
import numpy as np
import scipy.signal as sig
from .eventdetection import detect_peaks, trim, hist_threshold
from .powerspectrum import decibel, power, plot_decibel_psd
try:
    import matplotlib.cm as cm
    import matplotlib.colors as mc
except ImportError:
    pass


def group_candidate(good_freqs, all_freqs, freq, divisor,
                    freq_tol, max_freq_tol, min_group_size, verbose):
    """ Candidate harmonic frequencies belonging to a fundamental frequency.

    Parameters
    ----------
    good_freqs: 2-D array
        Frequency, power, and use count (columns) of strong peaks detected
        in a power spectrum.
    all_freqs: 2-D array
        Frequency, power, and use count (columns) of all peaks detected
        in a power spectrum.
    freq: float
        Fundamental frequency for which a harmonic group should be assembled.
    divisor: int
        Fundamental frequency was obtained from a frequency divided by divisor.
        0: Fundamental frequency is given as is.
    freq_tol: float
        Harmonics should fall within this frequency tolerance.
        This should be in the range of the frequency resolution
        and should not be smaller than half of the frequency resolution.
    max_freq_tol: float
        Maximum deviation of harmonics from their expected frequency.
        Peaks with frequencies between `freq_tol` and `max_freq_tol`
        get penalized the further away from their expected frequency.
    min_group_size: int
        Minimum number of harmonics of a harmonic group.
        The harmonics from min_group_size/3 to max(min_group_size, divisor)
        need to be in good_freqs.
    verbose: int
        Verbosity level.
    
    Returns
    -------
    new_group: 1-D aray of indices
        Frequencies in all_freqs belonging to the candidate group.
    fzero: float
        Adjusted fundamental frequency. If negative, no group was found.
    fzero_harmonics: int
        The highest harmonics that was used to recompute the fundamental frequency.
    """
    if verbose > 0:
        print('')
        
    dvs = True
    if divisor <= 0:
        divisor = 1
        dvs = False
        
    # 1. find harmonics in good_freqs and adjust fzero accordingly:
    fzero = freq
    fzero_harmonics = 1
    if len(good_freqs[:,0]) > 0:
        prev_freq = divisor * freq
        for h in range(divisor+1, 2*min_group_size+1):
            idx = np.argmin(np.abs(good_freqs[:,0]/h - fzero))
            ff = good_freqs[idx,0]
            if m.fabs(ff/h - fzero) > freq_tol:
                continue
            df = ff - prev_freq
            dh = np.round(df/fzero)
            fe = m.fabs(df/dh - fzero)
            if fe > 2.0*freq_tol:
                if h > min_group_size:
                    break
                continue
            # update fzero:
            prev_freq = ff
            fzero_harmonics = h
            fzero = ff/fzero_harmonics
            if verbose > 1:
                print('adjusted fzero to %.2fHz from %d-th harmonic' % (fzero, h))

    if verbose > 0:
        ds =  'divisor: %d, ' % divisor if dvs else ''
        print('# %sfzero=%7.2fHz adjusted from harmonics %d'
              % (ds, fzero, fzero_harmonics))

    # 2. check fzero:
    # freq might not be in our group anymore, because fzero was adjusted:
    if np.abs(freq - fzero) > freq_tol:
        if verbose > 0:
            print('  discarded: lost frequency')
        return [], -1.0, fzero_harmonics

    # 3. collect harmonics from all_freqs:
    new_group = -np.ones(min_group_size, dtype=np.int)
    new_penalties = np.ones(min_group_size)
    freqs = []
    prev_h = 0
    prev_fe = 0.0
    for h in range(1, min_group_size+1):
        penalty = 0
        i = np.argmin(np.abs(all_freqs[:,0]/h - fzero))
        f = all_freqs[i,0]
        if verbose > 2:
            print('    check %7.2fHz as %d. harmonics' % (f, h))
        fac = 1.0 if h >= divisor else 2.0
        fe = m.fabs(f/h - fzero)
        if fe > fac*max_freq_tol:
            if verbose > 1 and fe < 2*fac*max_freq_tol:
                print('    %d. harmonics at %7.2fHz is off by %7.2fHz (max between %5.2fHz and %5.2fHz) from %7.2fHz'
                      % (h, f, h*fe, h*fac*freq_tol, h*fac*max_freq_tol, h*fzero))
            continue
        if fe > fac*freq_tol:
            penalty = np.interp(fe, [fac*freq_tol, fac*max_freq_tol], [0.0, 1.0])
        if len(freqs) > 0:
            pf = freqs[-1]
            df = f - pf
            if df < 0.5*fzero:
                if len(freqs)>1:
                    pf = freqs[-2]
                    df = f - pf
                else:
                    pf = 0.0
                    df = h*fzero
            dh = m.floor(df/fzero + 0.5)
            fe = m.fabs(df/dh - fzero)
            if fe > 2*dh*fac*max_freq_tol:
                if verbose > 1:
                    print('    %d. harmonics at %7.2fHz is off by %7.2fHz (max between %5.2fHz and %5.2fHz) from previous harmonics at %7.2fHz'
                          % (h, f, dh*fe, 2*fac*dh*freq_tol, 2*fac*dh*max_freq_tol, pf))
                continue
            if fe > 2*dh*fac*freq_tol:
                penalty = np.interp(fe, [2*dh*fac*freq_tol, 2*dh*fac*max_freq_tol], [0.0, 1.0])
        else:
            fe = 0.0
        if h > prev_h or fe < prev_fe:
            if prev_h > 0 and h - prev_h > 1:
                if verbose > 1:
                    print('    previous harmonics %d more than 1 away from %d. harmonics at %7.2fHz'
                          % (prev_h, h, f))
                break
            if h == prev_h and len(freqs) > 0:
                freqs.pop()
            freqs.append(f)
            new_group[int(h)-1] = i
            new_penalties[int(h)-1] = penalty
            prev_h = h
            prev_fe = fe
            if verbose > 1:
                print('    %d. harmonics at %7.2fHz has been taken (peak %2d) with penalty %3.1f' % (h, f, i, penalty))

    # 4. check new group:

    # almost all harmonics in min_group_size required:
    max_penalties = min_group_size/3
    if np.sum(new_penalties) > max_penalties:
        if verbose > 0:
            print('  discarded group because sum of penalties %3.1f is more than %3.1f: indices' %
                  (np.sum(new_penalties), max_penalties), new_group, ' penalties', new_penalties)
        return [], -1.0, fzero_harmonics
    new_group = new_group[new_group>=0]
    
    # check use count of frequencies:
    double_use = np.sum(all_freqs[new_group, 2]>0)
    if double_use >= 2:    # XXX make this a parameter?
        if verbose > 0:
            print('  discarded group because of use count = %2d >= 2' % double_use)
        return [], -1.0, fzero_harmonics

    # 5. return results:
    return new_group, fzero, fzero_harmonics


def update_group(good_freqs, all_freqs, new_group, fzero,
                 freq_tol, verbose, group_str):
    """ Update good frequencies and harmonic group.

    Remove frequencies from good_freqs, add missing fundamental to group.

    Parameters
    ----------
    good_freqs: 2-D array
        Frequency, power, and use count (columns) of strong peaks detected
        in a power spectrum.
    all_freqs: 2-D array
        Frequency, power, and use count (columns) of all peaks detected
        in a power spectrum.
    new_group: 1-D aray of indices
        Frequencies in all_freqs of an harmonic group.
    fzero: float
        Fundamental frequency for which frequencies are collected in good_freqs.
    freq_tol: float
        Harmonics need to fall within this frequency tolerance.
        This should be in the range of the frequency resolution
        and not be smaller than half of the frequency resolution.
    verbose: int
        Verbosity level.
    group_str: string
        String for debug message.
    
    Returns
    -------
    good_freqs: 2-D array
        Frequency, power, and use count (columns) of strong peaks detected
        in a power spectrum with frequencies for harmonic group
        of fundamental frequency fzero removed.
    group: 2-D array
        Frequency, power, and use count (columns) of harmonic group
        for fundamental frequency fzero.
    """
    # initialize group:
    group = all_freqs[new_group,:]

    # indices of group in good_freqs:
    freq_tol *= 1.1
    indices = []
    for f in group[:,0]:
        idx = np.argmin(np.abs(good_freqs[:,0]-f))
        if np.abs(good_freqs[idx,0]-f) <= freq_tol:
            indices.append(idx)
    indices = np.asarray(indices, dtype=np.int)

    # harmonics in good_freqs:
    nharm = np.round(good_freqs[:,0]/fzero)
    idxs = np.where(np.abs(good_freqs[:,0] - nharm*fzero) <= freq_tol)[0]
    indices = np.unique(np.concatenate((indices, idxs)))

    # report:
    if verbose > 1:
        print('#     good freqs: ', indices,
              '[', ', '.join(['%.2f' % f for f in good_freqs[indices,0]]), ']')
        print('#     all freqs : ', new_group,
              '[', ', '.join(['%.2f' % f for f in all_freqs[new_group,0]]), ']')
    if verbose > 0:
        refi = np.argmax(group[:,1] > 0.0)
        print('')
        print(group_str)
        for i in range(len(group)):
            print('f=%8.2fHz n=%5.2f: power=%9.3g power/pmax=%6.4f=%5.1fdB'
                  % (group[i,0], group[i,0]/fzero,
                     group[i,1], group[i,1]/group[refi,1], decibel(group[i,1], group[refi,1])))
        print('')
            
    # erase group from good_freqs:
    good_freqs = np.delete(good_freqs, indices, axis=0)

    # adjust frequencies to fzero:
    group[:,0] = np.round(group[:,0]/fzero)*fzero

    # insert missing fzero:
    if np.round(group[0,0]/fzero) != 1.0:
        group = np.vstack(((fzero, group[0,1], -2.0), group))

    return good_freqs, group


def build_harmonic_group(good_freqs, all_freqs, freq_tol, max_freq_tol,
                         verbose=0, min_group_size=3, max_divisor=4):
    """Find all the harmonics belonging to the largest peak in a list of frequency peaks.

    Parameters
    ----------
    good_freqs: 2-D array
        Frequency, power, and use count (columns) of strong peaks detected
        in a power spectrum.
    all_freqs: 2-D array
        Frequency, power, and use count (columns) of all peaks detected
        in a power spectrum.
    freq_tol: float
        Harmonics should fall within this frequency tolerance.
        This should be in the range of the frequency resolution
        and should not be smaller than half of the frequency resolution.
    max_freq_tol: float
        Maximum deviation of harmonics from their expected frequency.
        Peaks with frequencies between `freq_tol` and `max_freq_tol`
        get penalized the further away from their expected frequency.
    verbose: int
        Verbosity level.
    min_group_size: int
        Minimum number of harmonics of a harmonic group.
        The harmonics from min_group_size/3 to max(min_group_size, divisor)
        need to be in good_freqs.
    max_divisor: int
        Maximum divisor used for checking for sub-harmonics.
    
    Returns
    -------
    good_freqs: 2-D array
        Frequency, power, and use count (columns) of strong peaks detected
        in a power spectrum with frequencies of the returned harmonic group removed.
    group: 2-D array
        The detected harmonic group. Might be empty.
    indices: 1-D array of indices
        Indices of the harmonic group in all_freqs.
    best_fzero_harmonics: int
        The highest harmonics that was used to recompute
        the fundamental frequency.
    fmax: float
        The frequency of the largest peak in good_freqs
        for which the harmonic group was detected.
    """
    # select strongest frequency for building the harmonic group:
    fmaxinx = np.argmax(good_freqs[:,1])
    fmax = good_freqs[fmaxinx,0]
    if verbose > 0:
        print('')
        print('%s build_harmonic_group %s' % (10*'#', 38*'#'))
        print('%s fmax=%7.2fHz, power=%9.3g %s'
              % (10*'#', fmax, good_freqs[fmaxinx,1], 27*'#'))
        print('good_freqs: ', '[',
              ', '.join(['%.2f' % f for f in good_freqs[:,0]]), ']')

    # container for harmonic groups:
    best_group = []
    best_value = -1e6
    best_divisor = 0
    best_fzero = 0.0
    best_fzero_harmonics = 0

    # check for integer fractions of the frequency:
    for divisor in range(1, max_divisor + 1):
        # 1. hypothesized fundamental:
        freq = fmax / divisor
        
        # 2. find harmonics in good_freqs and adjust fzero accordingly:
        group_size = min_group_size if divisor <= min_group_size else divisor
        new_group, fzero, fzero_harmonics = group_candidate(good_freqs, all_freqs,
                                                            freq, divisor,
                                                            freq_tol, max_freq_tol,
                                                            group_size, verbose)
        # no group found:
        if fzero < 0.0:
            continue

        # 3. compare new group to best group:
        peaksum = decibel(np.sum(all_freqs[new_group, 1])*min_group_size/len(new_group))
        diff = np.std(np.diff(decibel(all_freqs[new_group, 1])))
        new_group_value = peaksum - diff
        counts = np.sum(all_freqs[new_group, 2])
        if verbose > 0:
            print('  new group:                 fzero=%7.2fHz, nharmonics=%d, value=%6.1fdB, peaksum=%5.1fdB, diff=%6.1fdB, use count=%2d, peaks:'
                  % (fzero, len(new_group), new_group_value, peaksum, diff, counts), new_group)
            if verbose > 1:
                print('  best group:     divisor=%d, fzero=%7.2fHz, nharmonics=%d, value=%6.1fdB, peaks:'
                      % (best_divisor, best_fzero, len(best_group), best_value), best_group)
        # select new group if sum of peak power minus diff is larger:
        if len(new_group) >= len(best_group) and new_group_value >= best_value:
            best_value = new_group_value
            best_group = new_group
            best_divisor = divisor
            best_fzero = fzero
            best_fzero_harmonics = fzero_harmonics
            if verbose > 1:
                print('  new best group: divisor=%d, fzero=%7.2fHz, value=%6.1fdB, peaks:'
                      % (best_divisor, best_fzero, best_value), best_group)
            elif verbose > 0:
                print('  took as new best group')
                
    # no group found:
    if len(best_group) == 0:
        # erase freq:
        good_freqs = np.delete(good_freqs, fmaxinx, axis=0)
        group = np.zeros((0, 3))
        return good_freqs, group, best_group, 1, fmax

    # update frequencies and group:
    if verbose > 1:
        print('')
        print('# best group found for fmax=%.2fHz, fzero=%.2fHz, divisor=%d:'
              % (fmax, best_fzero, best_divisor))
    group_str = '%s resulting harmonic group for fmax=%.2fHz' % (10*'#', fmax)
    good_freqs, group = update_group(good_freqs, all_freqs, best_group, best_fzero, freq_tol, verbose, group_str)

    # good_freqs: removed all frequencies of bestgroup
    return good_freqs, group, best_group, best_fzero_harmonics, fmax


def retrieve_harmonic_group(freq, good_freqs, all_freqs,
                            freq_tol, max_freq_tol, verbose=0,
                            min_group_size=3):
    """Find all the harmonics belonging to a given fundamental.

    Parameters
    ----------
    freq: float
        Fundamental frequency for which harmonics are to be retrieved.
    good_freqs: 2-D array
        Frequency, power, and use count (columns) of strong peaks detected
        in a power spectrum. All harmonics of `freq` will be
        removed from `good_freqs`.
    all_freqs: 2-D array
        Frequency, power, and use count (columns) of all peaks detected
        in a power spectrum.
    freq_tol: float
        Harmonics should fall within this frequency tolerance.
        This should be in the range of the frequency resolution
        and should not be smaller than half of the frequency resolution.
    max_freq_tol: float
        Maximum deviation of harmonics from their expected frequency.
        Peaks with frequencies between `freq_tol` and `max_freq_tol`
        get penalized the further away from their expected frequency.
    verbose: int
        Verbosity level.
    min_group_size: int
        Minimum number of harmonics of a harmonic group.
        The harmonics from min_group_size/3 to max(min_group_size, divisor)
        need to be in good_freqs.
    
    Returns
    -------
    good_freqs: 2-D array
        Frequency, power, and use count (columns) of strong peaks detected
        in a power spectrum with frequencies of the returned harmonic group removed.
    group: 2-D array
        The detected harmonic group. Might be empty.
    indices: 1-D array of indices
        Indices of the harmonic group in all_freqs.
    fzero_harmonics: int
        The highest harmonics that was used to recompute
        the fundamental frequency.
    """
    if verbose > 0:
        print('')
        print('%s retrieve harmonic group %s' % (10*'#', 35*'#'))
        print('%s freq=%7.2fHz %s' % (10*'#', freq, 44*'#'))
        print('good_freqs: ', '[',
              ', '.join(['%.2f' % f for f in good_freqs[:,0]]), ']')

    # find harmonics in good_freqs and adjust fzero accordingly:
    new_group, fzero, fzero_harmonics = group_candidate(good_freqs, all_freqs,
                                                        freq, 0,
                                                        freq_tol, max_freq_tol,
                                                        min_group_size, verbose)

    # no group found:
    if fzero < 0.0:
        return good_freqs, np.zeros((0, 2)), np.zeros(0), fzero_harmonics

    if verbose > 1:
        print('')
        print('# group found for freq=%.2fHz, fzero=%.2fHz:'
              % (freq, fzero))
    group_str = '#### resulting harmonic group for freq=%.2fHz' % freq
    good_freqs, group = update_group(good_freqs, all_freqs, new_group,
                                     fzero, freq_tol, verbose, group_str)

    # good_freqs: removed all frequencies of bestgroup
    return good_freqs, group, new_group, fzero_harmonics


def expand_group(group, indices, freqs, freq_tol, max_harmonics=0):
    """ Add more harmonics to harmonic group.
    
    Parameters
    ----------
    group: 2-D array
        Group of fundamental frequency and harmonics
        as returned by build_harmonic_group.
    indices: 1-D array of indices
        Indices of the harmonics in group in `freqs`.
    freqs: 2-D array
        Frequency, power, and use count (columns) of all peaks detected
        in a power spectrum.
    freq_tol: float
        Harmonics need to fall within this frequency tolerance.
        This should be in the range of the frequency resolution
        and not be smaller than half of the frequency resolution.
    max_harmonics: int
        Maximum number of harmonics to be returned for each group.

    Returns
    -------
    group: 2-D array
        Expanded group of fundamental frequency and harmonics.
    indices: 1-D array of indices
        Indices of the harmonics in the expanded group in all_freqs.
    """
    if len(group) == 0:
        return group, indices
    fzero = group[0,0]
    if max_harmonics <= 0:
        max_harmonics = m.floor(freqs[-1,0]/fzero + 0.5) # round
    if max_harmonics <= len(group):
        return group, indices
    group_freqs = list(group[:,0])
    indices = list(indices)
    last_h = m.floor(group_freqs[-1]/fzero + 0.5) # round
    for h in range(last_h+1, max_harmonics+1):
        i = np.argmin(np.abs(freqs[:,0]/h - fzero))
        f = freqs[i,0]
        if m.fabs(f/h - fzero) > freq_tol:
            continue
        df = f - group_freqs[-1]
        dh = m.floor(df/fzero + 0.5) # round
        fe = m.fabs(df/dh - fzero)
        if fe > 2*freq_tol:
            continue
        group_freqs.append(f)
        indices.append(i)
    # assemble group:
    new_group = freqs[indices,:group.shape[1]]
    # keep filled in fundamental:
    if group[0,2] == -2:
        new_group = np.vstack((group[0,:], new_group))
    return new_group, np.array(indices, dtype=np.int)
            

def extract_fundamentals(good_freqs, all_freqs, freq_tol, max_freq_tol,
                         verbose=0, check_freqs=[],
                         mains_freq=60.0, mains_freq_tol=1.0,
                         min_freq=0.0, max_freq=2000.0, max_db_diff=20.0,
                         max_divisor=4, min_group_size=3, max_harmonics_db=-5.0,
                         max_harmonics=0, max_groups=0, **kwargs):
    """Extract fundamental frequencies from power-spectrum peaks.
                         
    Parameters
    ----------
    good_freqs: 2-D array
        Frequency, power, and use count (columns) of strong peaks detected
        in a power spectrum.
    all_freqs: 2-D array
        Frequency, power, and use count (columns) of all peaks detected
        in a power spectrum.
    freq_tol: float
        Harmonics should fall within this frequency tolerance.
        This should be in the range of the frequency resolution
        and should not be smaller than half of the frequency resolution.
    max_freq_tol: float
        Maximum deviation of harmonics from their expected frequency.
        Peaks with frequencies between `freq_tol` and `max_freq_tol`
        get penalized the further away from their expected frequency.
    verbose: int
        Verbosity level.
    check_freqs: list of float
        List of fundamental frequencies that will be checked
        first for being present and valid harmonic groups in the peak frequencies
        of a power spectrum.
    mains_freq: float
        Frequency of the mains power supply.
    mains_freq_tol: float
        Tolarerance around harmonics of the mains frequency,
        within which peaks are removed.
    min_freq: float
        Minimum frequency accepted as a fundamental frequency.
    max_freq: float
        Maximum frequency accepted as a fundamental frequency.
    max_db_diff: float
        If larger than zero, maximum standard deviation of differences between
        logarithmic powers of harmonics in decibel.
        Low values enforce smoother power spectra.
    max_divisor: int
        Maximum divisor used for checking for sub-harmonics.
    min_group_size: int
        Minimum number of harmonics of a harmonic group.
        The harmonics from min_group_size/3 to max(min_group_size, divisor)
        need to be in good_freqs.
    max_harmonics_db: float
        Maximum allowed power of the `min_group_size`-th and higher harmonics
        after the peak (in decibel relative to peak power withn the first
        `min_group_size` harmonics, i.e. if harmonics are required to be
        smaller than fundamental then this is a negative number).
        Make it a large positive number to effectively not check for relative power.
    max_harmonics: int
        Maximum number of harmonics to be returned for each group.
    max_groups: int
        If not zero the maximum number of most powerful harmonic groups.

    Returns
    -------
    group_list: list of 2-D arrays
        List of all harmonic groups found sorted by fundamental frequency.
        Each harmonic group is a 2-D array with the first dimension the harmonics
        and the second dimension containing frequency and power of each harmonic.
        If the power is zero, there was no corresponding peak in the power spectrum.
    fzero_harmonics_list: list of int
        The harmonics from which the fundamental frequencies were computed.
    mains_freqs: 2-d array
        Array of mains peaks found in all_freqs (frequency, power).
    """
    if verbose > 0:
        print('')

    # set use count to zero:
    all_freqs[:,2] = 0.0

    # remove power line harmonics from good_freqs:
    if mains_freq > 0.0:
        indices = np.where(np.abs(good_freqs[:,0] - np.round(good_freqs[:,0]/mains_freq)*mains_freq) < mains_freq_tol)[0]
        if len(indices)>0:
            if verbose > 1:
                print('remove power line frequencies',
                      ', '.join(['%.1f' % f for f in good_freqs[indices,0]]))
            good_freqs = np.delete(good_freqs, indices, axis=0)

    if verbose > 1:
        print('all_freqs:  ', '[',
              ', '.join(['%.2f' % f for f in all_freqs[:,0] if f < 3000.0]), ']')

    group_list = []
    fzero_harmonics_list = []
    first = True
    # as long as there are frequencies left in good_freqs:
    fi = 0
    while len(good_freqs) > 0:
        if fi < len(check_freqs):
            # check for harmonic group of a given fundamental frequency:
            fmax = check_freqs[fi]
            f0s = 'freq'
            good_freqs, harm_group, harm_indices, fzero_harmonics = \
                retrieve_harmonic_group(fmax, good_freqs, all_freqs,
                                        freq_tol, max_freq_tol,
                                        verbose-1, min_group_size)
            fi += 1
        else:
            # check for harmonic groups:
            f0s = 'fmax'
            good_freqs, harm_group, harm_indices, fzero_harmonics, fmax = \
                build_harmonic_group(good_freqs, all_freqs,
                                     freq_tol, max_freq_tol, verbose-1,
                                     min_group_size, max_divisor)

        # nothing found:
        if len(harm_group) == 0:
            if verbose > 1 - first:
                s = ' (largest peak)' if first else ''
                print('No harmonic group for %7.2fHz%s' % (fmax, s))
            first = False
            continue
        first = False

        # fill up harmonic group:
        harm_group, harm_indices = expand_group(harm_group, harm_indices,
                                                all_freqs, freq_tol,
                                                max_harmonics)

        # check whether fundamental was filled in:
        first_h = 0 if harm_group[0,2] > -2 else 1
        
        # increment use count:
        all_freqs[harm_indices,2] += 1
        if harm_group[0,2] == -1:
            harm_group[0,2] = -2

        # check frequency range of fundamental:
        fundamental_ok = (harm_group[0, 0] >= min_freq and
                          harm_group[0, 0] <= max_freq)
        # check power hum:
        mains_ok = ((mains_freq <= 0.0) or
                    (m.fabs(harm_group[0,0] - mains_freq) > freq_tol))
        # check smoothness:
        db_powers = decibel(harm_group[first_h:,1])
        diff = np.std(np.diff(db_powers))
        smooth_ok = max_db_diff <= 0.0 or diff < max_db_diff
        # check relative power of higher harmonics:
        p_max = np.argmax(db_powers[:min_group_size])
        db_powers -= db_powers[p_max]
        amplitude_ok = len(db_powers[p_max+min_group_size:]) == 0
        if amplitude_ok:
            pi = len(db_powers) - 1
        else:
            amplitude_ok = np.all((db_powers[p_max+min_group_size:] < max_harmonics_db) |
                                  (harm_group[first_h+p_max+min_group_size:,2] > 1))
            if amplitude_ok:
                pi = p_max+min_group_size-1
            else:
                pi = np.where((db_powers[p_max+min_group_size:] >= max_harmonics_db) &
                              (harm_group[first_h+p_max+min_group_size:,2] <= 1))[0][0]
                             
        # check:
        if fundamental_ok and mains_ok and smooth_ok and amplitude_ok:
            if verbose > 0:
                print('Accepting  harmonic group from %s=%7.2fHz: %7.2fHz power=%9.3g nharmonics=%2d, use count=%d'
                      % (f0s, fmax, harm_group[0,0], np.sum(harm_group[first_h:,1]),
                         len(harm_group), np.sum(harm_group[first_h:,2])))
            group_list.append(harm_group[:,0:2])
            fzero_harmonics_list.append(fzero_harmonics)
        else:
            if verbose > 0:
                fs = 'is ' if fundamental_ok else 'NOT'
                ms = 'not ' if mains_ok else 'IS'
                ss = 'smaller' if smooth_ok else 'LARGER '
                ps = 'smaller' if amplitude_ok else 'LARGER '
                print('Discarded  harmonic group from %s=%7.2fHz: %7.2fHz power=%9.3g: %s in frequency range, %s mains frequency, smooth=%4.1fdB %s than %4.1fdB, relpower[%d]=%5.1fdB %s than %5.1fdB'
                      % (f0s, fmax, harm_group[0,0], np.sum(harm_group[first_h:,1]),
                         fs, ms, diff, ss, max_db_diff,
                         pi, db_powers[pi], ps, max_harmonics_db))
                
    # select most powerful harmonic groups:
    if max_groups > 0 and len(group_list) > max_groups:
        n = len(group_list)
        powers = [np.sum(group[:,1]) for group in group_list]
        powers_inx = np.argsort(powers)
        group_list = [group_list[pi] for pi in powers_inx[-max_groups:]]
        fzero_harmonics_list = [fzero_harmonics_list[pi] for pi in powers_inx[-max_groups:]]
        if verbose > 0:
            print('Selected from %d groups the %d most powerful groups.' % (n, max_groups))
        
    # sort groups by fundamental frequency:
    freqs = [group[0,0] for group in group_list]
    freq_inx = np.argsort(freqs)
    group_list = [group_list[fi] for fi in freq_inx]
    fzero_harmonics_list = [fzero_harmonics_list[fi] for fi in freq_inx]

    if verbose > 1:
        print('')
        if len(group_list) > 0:
            print('## FUNDAMENTALS FOUND: ##')
            for group, fzero_h in zip(group_list, fzero_harmonics_list):
                print('%7.2fHz: power=%9.3g fzero-h=%2d'
                      % (group[0,0], np.sum(group[:,1]), fzero_h))
        else:
            print('## NO FUNDAMENTALS FOUND ##')

    # assemble mains frequencies from all_freqs:
    if mains_freq > 0.0:
        mains_freqs = all_freqs[np.abs(all_freqs[:,0] - np.round(all_freqs[:,0]/mains_freq)*mains_freq) < mains_freq_tol,:2]
    else:
        mains_freqs = np.zeros((0, 2))
                
    return group_list, fzero_harmonics_list, mains_freqs

            
def threshold_estimate(psd_data, low_thresh_factor=6.0, high_thresh_factor=10.0,
                       nbins=100, hist_height=1.0/ np.sqrt(np.e)):
    """Estimate thresholds for peak detection from histogram of power spectrum.

    The standard deviation of the noise floor without peaks is estimated from
    the width of the histogram of the power spectrum at `hist_height` relative height.
    The histogram is computed in the third quarter of the linearly detrended power spectrum.

    Parameters
    ----------
    psd_data: 1-D array
        The power spectrum from which to estimate the thresholds.
    low_thresh_factor: float
        Factor by which the estimated standard deviation of the noise floor
        is multiplied to set the `low_threshold`.
    high_thresh_factor: float
        Factor by which the estimated standard deviation of the noise floor
        is multiplied to set the `high_threshold`.
    nbins: int or list of floats
        Number of bins or the bins for computing the histogram.
    hist_height: float
        Height between 0 and 1 at which the standard deviation of the histogram is estimated.

    Returns
    -------
    low_threshold: float
        The threshold for peaks just above the noise floor.
    high_threshold: float
        The threshold for distinct peaks.
    center: float
        The baseline level of the power spectrum.
    """
    n = len(psd_data)
    psd_data_seg = psd_data[n//2:n*3//4]
    psd_data_seg = psd_data_seg[~np.isinf(psd_data_seg)]
    psd_data_seg = np.mean(psd_data_seg) + \
      sig.detrend(psd_data_seg, type='linear')
    noise_std, center = hist_threshold(psd_data_seg, thresh_fac=1.0, nbins=nbins)
    low_threshold = noise_std * low_thresh_factor
    high_threshold = noise_std * high_thresh_factor
    return low_threshold, high_threshold, center


def harmonic_groups(psd_freqs, psd, verbose=0, check_freqs=[],
                    low_threshold=0.0, high_threshold=0.0, thresh_bins=100,
                    low_thresh_factor=6.0, high_thresh_factor=10.0,
                    freq_tol_fac=1.0, max_freq_tol=1.0,
                    mains_freq=60.0, mains_freq_tol=1.0,
                    min_freq=0.0, max_freq=2000.0, max_db_diff=20.0, max_divisor=4,
                    min_group_size=3, max_harmonics_db=-5.0,
                    max_harmonics=0, max_groups=0, **kwargs):
    """Detect peaks in power spectrum and group them according to their harmonic structure.

    Parameters
    ----------
    psd_freqs: 1-D array
        Frequencies of the power spectrum.
    psd: 1-D array
        Power spectrum (linear, not decible).
    verbose: int
        Verbosity level.
    check_freqs: list of float
        List of fundamental frequencies that will be checked first for being
        present and valid harmonic groups in the peak frequencies
        of the power spectrum.
    low_threshold: float
        The relative threshold for detecting all peaks in the decibel spectrum.
    high_threshold: float
        The relative threshold for detecting good peaks in the decibel spectrum.
    thresh_bins: int or list of floats
        Number of bins or the bins for computing the histogram from
        which the standard deviation of the noise level in the `psd` is estimated.
    low_thresh_factor: float
        Factor by which the estimated standard deviation of the noise floor
        is multiplied to set the `low_threshold`.
    high_thresh_factor: float
        Factor by which the estimated standard deviation of the noise floor
        is multiplied to set the `high_threshold`.
    freq_tol_fac: float
        Harmonics should fall within `deltaf*freq_tol_fac`.
    max_freq_tol: float
        Maximum absolute frequency deviation of harmonics in Hertz..
    mains_freq: float
        Frequency of the mains power supply.
    mains_freq_tol: float
        Tolarerance around harmonics of the mains frequency,
        within which peaks are removed.
    min_freq: float
        Minimum frequency accepted as a fundamental frequency.
    max_freq: float
        Maximum frequency accepted as a fundamental frequency.
    max_db_diff: float
        If larger than zero, maximum standard deviation of differences between
        logarithmic powers of harmonics in decibel (larger than zero).
        Low values enforce smoother power spectra.
    max_divisor: int
        Maximum divisor used for checking for sub-harmonics.
    min_group_size: int
        Minimum number of harmonics of a harmonic group.
        The harmonics from min_group_size/3 to max(min_group_size, divisor)
        need to be in good_freqs.
    max_harmonics_db: float
        Maximum allowed power of the `min_group_size`-th and higher harmonics
        after the peak (in decibel relative to peak power withn the first
        `min_group_size` harmonics, i.e. if harmonics are required to be
        smaller than fundamental then this is a negative number).
        Make it a large positive number to effectively not check for relative power.
    max_harmonics: int
        Maximum number of harmonics to be returned for each group.
    max_groups: int
        If not zero the maximum number of most powerful harmonic groups.

    Returns
    -------
    group_list: list of 2-D arrays
        List of all extracted harmonic groups, sorted by fundamental frequency.
        Each harmonic group is a 2-D array with the first dimension the harmonics
        and the second dimension containing frequency and power of each harmonic.
        If the power is zero, there was no corresponding peak in the power spectrum.
    fzero_harmonics: list of ints
        The harmonics from which the fundamental frequencies were computed.
    mains: 2-d array
        Frequencies and power of multiples of the mains frequency found in the power spectrum.
    all_freqs: 2-D array
        Frequency, power, and use count (columns) of all peaks detected
        in the power spectrum.
    good_freqs: 1-D array
        Frequencies of strong peaks detected in the power spectrum.
    low_threshold: float
        The relative threshold for detecting all peaks in the decibel spectrum.
    high_threshold: float
        The relative threshold for detecting good peaks in the decibel spectrum.
    center: float
        The baseline level of the power spectrum.
    """
    if verbose > 0:
        print('')
        if verbose > 1:
            print(70*'#')
        print('##### harmonic_groups', 48*'#')

    # decibel power spectrum:
    log_psd = decibel(psd)
    max_idx = np.argmax(~np.isfinite(log_psd))
    if max_idx > 0:
        log_psd = log_psd[:max_idx]
    delta_f = psd_freqs[1] - psd_freqs[0]

    # thresholds:
    center = np.NaN
    if low_threshold <= 0.0 or high_threshold <= 0.0:
        low_th, high_th, center = threshold_estimate(log_psd, low_thresh_factor,
                                                     high_thresh_factor,
                                                     thresh_bins)
        if low_threshold <= 0.0:
            low_threshold = low_th
        if high_threshold <= 0.0:
            high_threshold = high_th
        if verbose > 1:
            print('')
            print('low_threshold =%4.1fdB, center+low_threshold =%6.1fdB' % (low_threshold, center + low_threshold))
            print('high_threshold=%4.1fdB, center+high_threshold=%6.1fdB' % (high_threshold, center + high_threshold))
            print('                                       center=%6.1fdB' % center)

    # detect peaks in decibel power spectrum:
    peaks, troughs = detect_peaks(log_psd, low_threshold)
    peaks, troughs = trim(peaks, troughs)
    all_freqs = np.zeros((len(peaks), 3))
    all_freqs[:,0] = psd_freqs[peaks]
    all_freqs[:,1] = psd[peaks]

    if len(all_freqs) == 0:
        return [], [], [], np.zeros((0, 3)), [], low_threshold, high_threshold, center
        
    # select good peaks:
    good_freqs = all_freqs[(log_psd[peaks] - log_psd[troughs] > high_threshold) &
                           (all_freqs[:,0] >= min_freq) &
                           (all_freqs[:,0] < max_freq*max_divisor),:]

    # detect harmonic groups:
    freq_tol = delta_f*freq_tol_fac
    if max_freq_tol < 1.1*freq_tol:
        max_freq_tol = 1.1*freq_tol
    groups, fzero_harmonics, mains = \
      extract_fundamentals(good_freqs, all_freqs,
                           freq_tol, max_freq_tol,
                           verbose, check_freqs, mains_freq, mains_freq_tol,
                           min_freq, max_freq, max_db_diff, max_divisor, min_group_size,
                           max_harmonics_db, max_harmonics, max_groups)

    return (groups, fzero_harmonics, mains, all_freqs, good_freqs[:,0],
            low_threshold, high_threshold, center)


def fundamental_freqs(group_list):
    """
    Extract fundamental frequencies from lists of harmonic groups.

    The inner list of 2-D arrays of the input argument is transformed into
    a 1-D array containig the fundamental frequencies extracted from
    the 2-D arrays.

    Parameters
    ----------
    group_list: (list of (list of ...)) list of 2-D arrays
        Arbitrarily nested lists of harmonic groups as returned by
        extract_fundamentals() and harmonic_groups() with the element
        [0, 0] being the fundamental frequency.

    Returns
    -------
    fundamentals: (list of (list of ...)) 1-D array
        Nested list (corresponding to `group_list`) of 1-D arrays
        with the fundamental frequencies extracted from the harmonic groups.
    """
    if len(group_list) == 0:
        return np.array([])

    # check whether group_list is list of harmonic groups:
    list_of_groups = True
    for group in group_list:
        if not ( hasattr(group, 'shape') and len(group.shape) == 2 ):
            list_of_groups = False
            break

    if list_of_groups:
        fundamentals = np.array([group[0, 0] for group in group_list if len(group) > 0])
    else:
        fundamentals = []
        for groups in group_list:
            f = fundamental_freqs(groups)
            fundamentals.append(f)
    return fundamentals


def fundamental_freqs_and_power(group_list, power=False,
                                ref_power=1.0, min_power=1e-20):
    """
    Extract fundamental frequencies and their power in dB from lists of harmonic groups.

    The inner list of 2-D arrays of the input argument is transformed
    into a 2-D array containig for each fish (1st dimension) the
    fundamental frequencies and powers (summed over all harmonics)
    extracted from the 2-D arrays.
    
    Parameters
    ----------
    group_list: (list of (list of ...)) list of 2-D arrays
        Arbitrarily nested lists of harmonic groups as returned by
        extract_fundamentals() and harmonic_groups() with the element
        [0, 0] being the fundamental frequency and the elements [:,1] being
        the powers of each harmonics.
    power: boolean
        If `False` convert the power into decibel using the
        powerspectrum.decibel() function.
    ref_power: float
        Reference power for computing decibel.
        If set to `None` the maximum power is used.
    min_power: float
        Power values smaller than `min_power` are set to `-np.inf`.

    Returns
    -------
    fundamentals: (list of (list of ...)) 2-D array
        Nested list (corresponding to `group_list`) of 2-D arrays
        with fundamental frequencies in first column and
        corresponding power in second column.
    """

    if len(group_list) == 0:
        return np.array([])

    # check whether group_list is list of harmonic groups:
    list_of_groups = True
    for group in group_list:
        if not ( hasattr(group, 'shape') and len(group.shape) == 2 ):
            list_of_groups = False
            break
        
    if list_of_groups:
        fundamentals = np.array([[group[0, 0], np.sum(group[:, 1])]
                            for group in group_list if len(group) > 0])
        if not power:
            fundamentals[:, 1] = decibel(fundamentals[:, 1],
                                         ref_power, min_power)
    else:
        fundamentals = []
        for groups in group_list:
            f = fundamental_freqs_and_power(groups, power,
                                            ref_power, min_power)
            fundamentals.append(f)
    return fundamentals


def add_relative_power(freqs):
    """ Add a column with relative power.

    For each element in `freqs`, its maximum power is subtracted
    from all powers.

    Parameters
    ----------
    freqs: list of 2-D arrays
        First column in the ndarrays is fundamental frequency and
        second column the corresponding power.
        Further columns are optional and kept in the returned list.
        fundamental_freqs_and_power() returns such a list.

    Returns
    -------
    power_freqs: list of 2-D arrays
        Same as freqs, but with an added column containing the relative power.
    """
    return [np.column_stack((f, f[:,1] - np.max(f[:,1]))) for f in freqs]


def add_power_ranks(freqs):
    """ Add a column with power ranks.

    Parameters
    ----------
    freqs: list of 2-D arrays
        First column in the arrays is fundamental frequency and
        second column the corresponding power.
        Further columns are optional and kept in the returned list.
        fundamental_freqs_and_power() returns such a list.

    Returns
    -------
    rank_freqs: list of 2-D arrays
        Same as freqs, but with an added column containing the ranks.
        The highest power is assinged to zero,
        lower powers are assigned negative integers.
    """
    rank_freqs = []
    for f in freqs:
        i = np.argsort(f[:,1])[::-1]
        ranks = np.empty_like(i)
        ranks[i] = -np.arange(len(i))
        rank_freqs.append(np.column_stack((f, ranks)))
    return rank_freqs


def similar_indices(freqs, df_thresh, nextfs=0):
    """ Indices of similar frequencies.

    If two frequencies from different elements in the inner lists of `freqs` are
    reciprocally the closest to each other and closer than `df_thresh`,
    then two indices (element, frequency) of the respective other frequency
    are appended.

    Parameters
    ----------
    freqs: (list of (list of ...)) list of 2-D arrays
        First column in the arrays is fundamental frequency.
    df_thresh: float
        Fundamental frequencies closer than this threshold are considered
        equal.
    nextfs: int
        If zero, compare all elements in freqs with each other. Otherwise,
        only compare with the `nextfs` next elements in freqs.

    Returns
    -------
    indices: (list of (list of ...)) list of list of two-tuples of int
        For each frequency of each element in `freqs` a list of two tuples containing
        the indices of elements and frequencies that are similar.
    """
    if len(freqs) == 0:
        return []
    
    # check whether freqs is list of fundamental frequencies and powers:
    list_of_freq_power = True
    for group in freqs:
        if not (hasattr(group, 'shape') and len(group.shape) == 2):
            list_of_freq_power = False
            break

    if list_of_freq_power:
        indices = [ [[] for j in range(len(freqs[i]))] for i in range(len(freqs))]
        for j in range(len(freqs)-1):
            freqsj = np.asarray(freqs[j])
            for m in range(len(freqsj)):
                freq1 = freqsj[m]
                nn = len(freqs) if nextfs == 0 else j+1+nextfs
                if nn > len(freqs):
                    nn = len(freqs)
                for k in range(j+1, nn):
                    freqsk = np.asarray(freqs[k])
                    if len(freqsk) == 0:
                        continue
                    n = np.argmin(np.abs(freqsk[:,0] - freq1[0]))
                    freq2 = freqsk[n]
                    if np.argmin(np.abs(freqsj[:,0] - freq2[0])) != m:
                        continue
                    if np.abs(freq1[0] - freq2[0]) < df_thresh:
                        indices[k][n].append((j, m))
                        indices[j][m].append((k, n))
    else:
        indices = []
        for groups in freqs:
            indices.append(similar_indices(groups, df_thresh, nextfs))
    return indices


def unique_mask(freqs, df_thresh, nextfs=0):
    """ Mark similar frequencies from different recordings as dublicate.

    If two frequencies from different elements in `freqs` are
    reciprocally the closest to each other and closer than `df_thresh`,
    then the one with the smaller power is marked for removal.

    Parameters
    ----------
    freqs: list of 2-D arrays
        First column in the arrays is fundamental frequency and
        second column the corresponding power or equivalent.
        If values in the second column are equal (e.g. they are the same ranks),
        and there is a third column (e.g. power),
        the third column is used to decide, which element should be removed.
    df_thresh: float
        Fundamental frequencies closer than this threshold are considered
        equal.
    nextfs: int
        If zero, compare all elements in freqs with each other. Otherwise,
        only compare with the `nextfs` next elements in freqs.

    Returns
    -------
    mask: list of boolean arrays
        For each element in `freqs` True if that frequency should be kept.
    """
    mask = [np.ones(len(freqs[i]), dtype=bool) for i in range(len(freqs))]
    for j in range(len(freqs)-1):
        freqsj = np.asarray(freqs[j])
        for m in range(len(freqsj)):
            freq1 = freqsj[m]
            nn = len(freqs) if nextfs == 0 else j+1+nextfs
            if nn > len(freqs):
                nn = len(freqs)
            for k in range(j+1, nn):
                freqsk = np.asarray(freqs[k])
                if len(freqsk) == 0:
                    continue
                n = np.argmin(np.abs(freqsk[:,0] - freq1[0]))
                freq2 = freqsk[n]
                if np.argmin(np.abs(freqsj[:,0] - freq2[0])) != m:
                    continue
                if np.abs(freq1[0] - freq2[0]) < df_thresh:
                    if freq1[1] > freq2[1]:
                        mask[k][n] = False
                    elif freq1[1] < freq2[1]:
                        mask[j][m] = False
                    elif len(freq1) > 2:
                        if freq1[2] > freq2[2]:
                            mask[k][n] = False
                        else:
                            mask[j][m] = False
                    else:
                        mask[j][m] = False
    return mask


def unique(freqs, df_thresh, mode='power', nextfs=0):
    """ Remove similar frequencies from different recordings.

    If two frequencies from different elements in the inner lists of `freqs`
    are reciprocally the closest to each other and closer than `df_thresh`,
    then the one with the smaller power is removed. As power, either the
    absolute power as provided in the second column of the data elements
    in `freqs` is taken (mode=='power'), or the relative power
    (mode='relpower'), or the power rank (mode='rank').

    Parameters
    ----------
    freqs: (list of (list of ...)) list of 2-D arrays
        First column in the arrays is fundamental frequency and
        second column the corresponding power, as returned by
        fundamental_freqs_and_power().
    df_thresh: float
        Fundamental frequencies closer than this threshold are considered
        equal.
    mode: string
        - 'power': use second column of freqs elements as power.
        - 'relpower': use relative power computed from the second column
          of freqs elements for deciding which frequency to delete.
        - 'rank': use rank of second column of freqs elements
                  for deciding which frequency to delete.
    nextfs: int
        If zero, compare all elements in freqs with each other. Otherwise,
        only compare with the `nextfs` next elements in freqs.

    Returns
    -------
    uniqe_freqs: (list of (list of ...)) list of 2-D arrays
        Same as `freqs` but elements with similar fundamental frequencies
        removed.
    """
    if len(freqs) == 0:
        return []
    
    # check whether freqs is list of fundamental frequencies and powers:
    list_of_freq_power = True
    for group in freqs:
        if not (hasattr(group, 'shape') and len(group.shape) == 2):
            list_of_freq_power = False
            break

    if list_of_freq_power:
        if mode == 'power':
            mask = unique_mask(freqs, df_thresh, nextfs)
        elif mode == 'relpower':
            power_freqs = [f[:,[0, 2, 1]] for f in add_relative_power(freqs)]
            mask = unique_mask(power_freqs, df_thresh, nextfs)
        elif mode == 'rank':
            rank_freqs = [f[:,[0, 2, 1]] for f in add_power_ranks(freqs)]
            mask = unique_mask(rank_freqs, df_thresh, nextfs)
        else:
            raise ValueError('%s is not a valid mode for unique(). Choose one of "power" or "rank"')
        unique_freqs = []
        for f, m in zip(freqs, mask):
            unique_freqs.append(f[m])
    else:
        unique_freqs = []
        for groups in freqs:
            unique_freqs.append(unique(groups, df_thresh, mode, nextfs))
    return unique_freqs


def colors_markers():
    """
    Generate a list of colors and markers for plotting.

    Returns
    -------
    colors: list
        list of colors
    markers: list
        list of markers
    """
    # color and marker range:
    colors = []
    markers = []
    mr2 = []

    # first color range:
    cc0 = cm.gist_rainbow(np.linspace(0.0, 1.0, 8))

    # shuffle it:
    for k in range((len(cc0) + 1) // 2):
        colors.extend(cc0[k::(len(cc0) + 1) // 2])
    markers.extend(len(cc0) * 'o')
    mr2.extend(len(cc0) * 'v')
    # second darker color range:
    cc1 = cm.gist_rainbow(np.linspace(0.33 / 7.0, 1.0, 7))
    cc1 = mc.hsv_to_rgb(mc.rgb_to_hsv(np.array([cc1[:, :3]])) * np.array([1.0, 0.9, 0.7]))[0]
    cc1 = np.hstack((cc1, np.ones((len(cc1),1))))
    # shuffle it:
    for k in range((len(cc1) + 1) // 2):
        colors.extend(cc1[k::(len(cc1) + 1) // 2])
    markers.extend(len(cc1) * '^')
    mr2.extend(len(cc1) * '*')
    # third lighter color range:
    cc2 = cm.gist_rainbow(np.linspace(0.67 / 6.0, 1.0, 6))
    cc2 = mc.hsv_to_rgb(mc.rgb_to_hsv(np.array([cc1[:, :3]])) * np.array([1.0, 0.5, 1.0]))[0]
    cc2 = np.hstack((cc2, np.ones((len(cc2),1))))
    # shuffle it:
    for k in range((len(cc2) + 1) // 2):
        colors.extend(cc2[k::(len(cc2) + 1) // 2])
    markers.extend(len(cc2) * 'D')
    mr2.extend(len(cc2) * 'x')
    markers.extend(mr2)
    return colors, markers


def plot_harmonic_groups(ax, group_list, indices=None, max_groups=0,
                         sort_by_freq=True, label_power=False,
                         colors=None, markers=None, legend_rows=8, **kwargs):
    """
    Mark decibel power of fundamentals and their harmonics in a plot.

    Parameters
    ----------
    ax: axis for plot
        Axis used for plotting.
    group_list: list of 2-D arrays
        Lists of harmonic groups as returned by extract_fundamentals() and
        harmonic_groups() with the element [0, 0] of the harmonic groups being
        the fundamental frequency, and element[0, 1] being the corresponding power.
    indices: list of int or None
        If smaller than zero then set the legend label of the corresponding group in brackets.
    max_groups: int
            If not zero plot only the max_groups most powerful groups.
    sort_by_freq: boolean
            If True sort legend by frequency, otherwise by power.
    label_power: boolean
        If `True` put the power in decibel in addition to the frequency into the legend.
    colors: list of colors or None
            If not None list of colors for plotting each group
    markers: list of markers or None
            If not None list of markers for plotting each group
    legend_rows: int
            Maximum number of rows to be used for the legend.
    kwargs: 
            Key word arguments for the legend of the plot.
    """

    if len(group_list) == 0:
        return
    
    # sort by power:
    powers = np.array([np.sum(group[:,1]) for group in group_list])
    max_power = np.max(powers)
    power_idx = np.argsort(powers)
    if max_groups > 0 and len(power_idx > max_groups):
        power_idx = power_idx[-max_groups:]
    idx = np.array(list(reversed(power_idx)))

    # sort by frequency:
    if sort_by_freq:
        freqs = [group_list[group][0, 0] for group in idx]
        if legend_rows > 0 and legend_rows < len(freqs):
            idx[:legend_rows] = idx[np.argsort(freqs[:legend_rows])]
        else:
            idx = idx[np.argsort(freqs)]

    # plot:
    for k, i in enumerate(idx):
        group = group_list[i]
        x = group[:,0]
        y = decibel(group[:,1])
        msize = 7.0 + 10.0*(powers[i]/max_power)**0.25
        color_kwargs = {}
        if colors is not None:
            color_kwargs = {'color': colors[k%len(colors)]}
        label = '%6.1f Hz' % group[0, 0]
        if label_power:
            label += ' %6.1f dB' % decibel(np.array([np.sum(group[:,1])]))[0]
        if indices is not None:
            if indices[i] < 0:
                label = '(' + label + ')'
            else:
                label = ' ' + label + ' '
        if legend_rows > 5 and k >= legend_rows:
            label = None
        if markers is None:
            ax.plot(x, y, 'o', ms=msize, label=label, **color_kwargs)
        else:
            if k >= len(markers):
                break
            ax.plot(x, y, linestyle='None', marker=markers[k],
                    mec=None, mew=0.0, ms=msize, label=label, **color_kwargs)

    # legend:
    if legend_rows > 0:
        if legend_rows > 5:
            ncol = 1
        else:
            ncol = (len(idx)-1) // legend_rows + 1
        ax.legend(numpoints=1, ncol=ncol, **kwargs)
    else:
        ax.legend(numpoints=1, **kwargs)


def plot_psd_harmonic_groups(ax, psd_freqs, psd, group_list,
                             mains=None, all_freqs=None, good_freqs=None,
                             log_freq=False, min_freq=0.0, max_freq=2000.0, ymarg=0.0):
    """
    Plot decibel power-spectrum with detected peaks, harmonic groups, and mains frequencies.
    
    Parameters
    ----------
    psd_freqs: array
        Frequencies of the power spectrum.
    psd: array
        Power spectrum (linear, not decible).
    group_list: list of 2-D arrays
        Lists of harmonic groups as returned by extract_fundamentals() and
        harmonic_groups() with the element [0, 0] of the harmonic groups being the fundamental frequency,
        and element[0, 1] being the corresponding power.
    mains: 2-D array
        Frequencies and power of multiples of the mains frequency found in the power spectrum.
    all_freqs: 2-D array
        Peaks in the power spectrum detected with low threshold.
    good_freqs: 1-D array
        Frequencies of peaks detected with high threshold.
    log_freq: boolean
        Logarithmic (True) or linear (False) frequency axis.
    min_freq: float
        Limits of frequency axis are set to `(min_freq, max_freq)`
        if `max_freq` is greater than zero
    max_freq: float
        Limits of frequency axis are set to `(min_freq, max_freq)`
        and limits of power axis are computed from powers below max_freq
        if `max_freq` is greater than zero
    ymarg: float
        Add this to the maximum decibel power for setting the ylim.
    """
    
    # mark all and good psd peaks:
    pmin, pmax = ax.get_ylim()
    doty = pmax - 5.0
    if all_freqs is not None:
        ax.plot(all_freqs[:, 0], np.zeros(len(all_freqs[:, 0])) + doty, 'o', color='#ffffff')
    if good_freqs is not None:
        ax.plot(good_freqs, np.zeros(len(good_freqs)) + doty, 'o', color='#888888')
    # mark mains frequencies:
    if mains is not None and len(mains) > 0:
        fpeaks = mains[:, 0]
        fpeakinx = [int(np.round(fp/(psd_freqs[1]-psd_freqs[0]))) for fp in fpeaks if fp < psd_freqs[-1]]
        ax.plot(fpeaks[:len(fpeakinx)], decibel(psd[fpeakinx]), linestyle='None',
                marker='.', color='k', ms=10, mec=None, mew=0.0,
                label='%3.0f Hz mains' % mains[0, 0])
    # mark harmonic groups:
    colors, markers = colors_markers()
    plot_harmonic_groups(ax, group_list, max_groups=0, sort_by_freq=True,
                         colors=colors, markers=markers, legend_rows=8,
                         loc='upper right')
    # plot power spectrum:
    plot_decibel_psd(ax, psd_freqs, psd, log_freq=log_freq, min_freq=min_freq,
                     max_freq=max_freq, ymarg=ymarg, color='blue')

    
def add_psd_peak_detection_config(cfg, low_threshold=0.0, high_threshold=0.0,
                                  thresh_bins=100,
                                  low_thresh_factor=6.0, high_thresh_factor=10.0):
    """ Add parameter needed for detection of peaks in power spectrum used by
    harmonic_groups() as a new section to a configuration.

    Parameters
    ----------
    cfg: ConfigFile
        The configuration.
    """

    cfg.add_section('Thresholds for peak detection in power spectra:')
    cfg.add('lowThreshold', low_threshold, 'dB', 'Threshold for all peaks.\n If 0.0 estimate threshold from histogram.')
    cfg.add('highThreshold', high_threshold, 'dB', 'Threshold for good peaks. If 0.0 estimate threshold from histogram.')
    
    cfg.add_section('Threshold estimation:\nIf no thresholds are specified they are estimated from the histogram of the decibel power spectrum.')
    cfg.add('thresholdBins', thresh_bins, '', 'Number of bins used to compute the histogram used for threshold estimation.')
    cfg.add('lowThresholdFactor', low_thresh_factor, '', 'Factor for multiplying standard deviation of noise floor for lower threshold.')
    cfg.add('highThresholdFactor', high_thresh_factor, '', 'Factor for multiplying standard deviation of noise floor for higher threshold.')


def psd_peak_detection_args(cfg):
    """ Translates a configuration to the respective parameter names for the
    detection of peaks in power spectrum used by harmonic_groups().
    The return value can then be passed as key-word arguments to this function.

    Parameters
    ----------
    cfg: ConfigFile
        The configuration.

    Returns
    -------
    a: dict
        Dictionary with names of arguments of the `harmonic-group()` function
        and their values as supplied by `cfg`.
    """
    return cfg.map({'low_threshold': 'lowThreshold',
                    'high_threshold': 'highThreshold',
                    'thresh_bins': 'thresholdBins',
                    'low_thresh_factor': 'lowThresholdFactor',
                    'high_thresh_factor': 'highThresholdFactor'})


def add_harmonic_groups_config(cfg, mains_freq=60.0, mains_freq_tol=1.0,
                               max_divisor=4, freq_tol_fac=1.0,
                               max_freq_tol=1.0, min_group_size=3,
                               min_freq=20.0, max_freq=2000.0, max_db_diff=20.0,
                               max_harmonics_db=-5.0, max_harmonics=0, max_groups=0):
    """ Add parameter needed for detection of harmonic groups as
    a new section to a configuration.

    Parameters
    ----------
    cfg: ConfigFile
        The configuration.
    """
    
    cfg.add_section('Harmonic groups:')
    cfg.add('mainsFreq', mains_freq, 'Hz', 'Mains frequency to be excluded.')
    cfg.add('mainsFreqTolerance', mains_freq_tol, 'Hz', 'Exclude peaks within this tolerance around multiples of the mains frequency.')
    cfg.add('minimumGroupSize', min_group_size, '',
'Minimum number of harmonics (inclusively fundamental) that make up a harmonic group.')
    cfg.add('maxDivisor', max_divisor, '', 'Maximum ratio between the frequency of the largest peak and its fundamental.')
    cfg.add('freqTolerance', freq_tol_fac, '',
            'Harmonics should be within this factor times the frequency resolution of the power spectrum. Needs to be higher than 0.5!')
    cfg.add('maximumFreqTolerance', max_freq_tol, 'Hz',
            'Maximum deviation of harmonics from their expected value.')
    
    cfg.add_section('Acceptance of best harmonic groups:')
    cfg.add('minimumFrequency', min_freq, 'Hz', 'Minimum frequency allowed for the fundamental.')
    cfg.add('maximumFrequency', max_freq, 'Hz', 'Maximum frequency allowed for the fundamental.')
    cfg.add('maximumPowerDifference', max_db_diff, 'dB', 'If larger than zero, maximum standard deviation allowed for difference in logarithmic power between successive harmonics. Smaller values enforce smoother spectra.')
    cfg.add('maximumHarmonicsPower', max_harmonics_db, 'dB', 'Maximum allowed power of the minimumGroupSize-th and higher harmonics relative to peak power.')
    cfg.add('maximumHarmonics', max_harmonics, '', '0: keep all, >0 only keep the first # harmonics.')
    cfg.add('maximumGroups', max_groups, '', 'Maximum number of harmonic groups. If 0 process all.')


def harmonic_groups_args(cfg):
    """ Translates a configuration to the
    respective parameter names of the harmonic-group detection functions.
    The return value can then be passed as key-word arguments to this function.

    Parameters
    ----------
    cfg: ConfigFile
        The configuration.

    Returns
    -------
    a: dict
        Dictionary with names of arguments of the harmonic-group detection functions
        and their values as supplied by `cfg`.
    """
    return cfg.map({'mains_freq': 'mainsFreq',
                    'mains_freq_tol': 'mainsFreqTolerance',
                    'freq_tol_fac': 'freqTolerance',
                    'max_freq_tol': 'maximumFreqTolerance',
                    'max_divisor': 'maxDivisor',
                    'min_group_size': 'minimumGroupSize',
                    'min_freq': 'minimumFrequency',
                    'max_freq': 'maximumFrequency',
                    'max_db_diff': 'maximumPowerDifference',
                    'max_harmonics_db': 'maximumHarmonicsPower',                     
                    'max_harmonics': 'maximumHarmonics',
                    'max_groups': 'maximumGroups'})


if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt
    from .fakefish import wavefish_eods
    from .powerspectrum import psd

    if len(sys.argv) < 2:
        # generate data:
        title = 'simulation'
        samplerate = 44100.0
        eodfs = [123.0, 333.0, 666.0, 666.5]
        fish1 = 0.5*wavefish_eods('Eigenmannia', eodfs[0], samplerate, duration=8.0, noise_std=0.01)
        fish2 = 1.0*wavefish_eods('Eigenmannia', eodfs[1], samplerate, duration=8.0, noise_std=0.01)
        fish3 = 10.0*wavefish_eods('Alepto', eodfs[2], samplerate, duration=8.0, noise_std=0.01)
        fish4 = 6.0*wavefish_eods('Alepto', eodfs[3], samplerate, duration=8.0, noise_std=0.01)
        data = fish1 + fish2 + fish3 + fish4
    else:
        from .dataloader import load_data
        print("load %s ..." % sys.argv[1])
        data, samplerate, unit = load_data(sys.argv[1], 0)
        title = sys.argv[1]

    # retrieve fundamentals from power spectrum:
    psd_data = psd(data, samplerate, freq_resolution=0.5)
    groups, _, mains, all_freqs, good_freqs, _, _, _ = harmonic_groups(psd_data[0], psd_data[1], verbose=0, check_freqs=[123.0, 666.0])
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plot_psd_harmonic_groups(ax, psd_data[0], psd_data[1], groups, mains,
                             all_freqs, good_freqs, max_freq=3000.0)
    ax.set_title(title)
    plt.show()
    #exit()
    # unify fundamental frequencies:
    fundamentals = fundamental_freqs(groups)
    np.set_printoptions(formatter={'float': lambda x: '%5.1f' % x})
    print('fundamental frequencies extracted from power spectrum:')
    print(fundamentals)
    print('')
    freqs = fundamental_freqs_and_power([groups])
    freqs.append(np.array([[44.0, -20.0], [44.2, -10.0], [320.5, 2.5], [665.5, 5.0], [666.2, 10.0]]))
    freqs.append(np.array([[123.3, 1.0], [320.2, -2.0], [668.4, 2.0]]))
    rank_freqs = add_relative_power(freqs)
    rank_freqs = add_power_ranks(rank_freqs)
    print('all frequencies (frequency, power, relpower, rank):')
    print('\n'.join(( str(f) for f in rank_freqs)))
    print('')
    indices = similar_indices(freqs, 1.0)
    print('similar indices:')
    print('\n'.join(( ('\n  '.join((str(f) for f in g)) for g in indices))))
    print('')
    unique_freqs = unique(freqs, 1.0, 'power')
    print('unique power:')
    print('\n'.join(( str(f) for f in unique_freqs)))
    print('')
    unique_freqs = unique(freqs, 1.0, 'relpower')
    print('unique relative power:')
    print('\n'.join(( str(f) for f in unique_freqs)))
    print('')
    unique_freqs = unique(freqs, 1.0, 'rank')
    print('unique rank:')
    print('\n'.join(( str(f) for f in unique_freqs)))
    print('')
    unique_freqs = unique(freqs, 1.0, 'rank', 1)
    print('unique rank for next neighor only:')
    print('\n'.join(( str(f) for f in unique_freqs)))
    print('')
