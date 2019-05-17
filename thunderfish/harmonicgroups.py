"""
# Harmonic group detection
Extract harmonic groups from power spectra.

## Harmonic group extraction
- `harmonic_groups()`: detect peaks in power spectrum and group them
                       according to their harmonic structure.
- `extract_fundamentals()`: collect harmonic groups from
                            lists of power spectrum peaks.
- `threshold_estimate()`: estimates thresholds for peak detection
                          in a power spectrum.

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

## Configuration parameter
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
import numpy as np
import scipy.signal as sig
from .eventdetection import detect_peaks, peak_size_width, hist_threshold
from .powerspectrum import decibel, power, plot_decibel_psd
try:
    import matplotlib.cm as cm
    import matplotlib.colors as mc
except ImportError:
    pass


def build_harmonic_group(freqs, more_freqs, freqtol, verbose=0,
                         min_group_size=4, max_divisor=4,
                         max_double_use_count=1, max_fill_ratio=0.25):
    """Find all the harmonics belonging to the largest peak in a list of frequency peaks.

    Parameters
    ----------
    freqs: 2-D array
        List of frequency, power, size, width and count of strong peaks in a power spectrum.
    more_freqs:
        List of frequency, power, size, width and count of all peaks in a power spectrum.
    freqtol: float
        Harmonics need to fall within this frequency tolerance.
    verbose: int
        Verbosity level.
    max_divisor: int
        Maximum divisor used for checking for sub-harmonics.
    max_double_use_count: int
        Maximum number of harmonic groups a single peak can be part of.
    max_fill_ratio: float
        Maximum allowed fraction of filled in frequencies.
    
    Returns
    -------
    freqs: 2-D array
        List of strong frequencies with the frequencies of group removed.
    more_freqs: 2-D array
        List of all frequencies with updated double-use counts.
    group: 2-D array
        The detected harmonic group. Might be empty.
    best_fzero_harmonics: int
        The highest harmonics that was used to recompute the fundamental frequency.
    fmax: float
        The frequency of the largest peak in freqs for which the harmonic group was detected.
    """
    
    # select strongest frequency for building the harmonic group:
    fmaxinx = np.argmax(freqs[:,1])
    fmax = freqs[fmaxinx,0]
    if verbose > 1:
        print('')
        print(70 * '#')
        print('freqs:     ', '[', ', '.join(['{:.2f}'.format(f) for f in freqs[:, 0]]), ']')
        print('more_freqs:', '[', ', '.join(
            ['{:.2f}'.format(f) for f in more_freqs[:, 0] if f < 3000.0]), ']')
        print('## fmax is: {0: .2f}Hz: {1:.5g} ##\n'.format(fmax, np.max(freqs[:, 1])))

    # container for harmonic groups
    best_group = list()
    best_moregroup = list()
    best_group_peaksum = 0.0
    best_divisor = 0
    best_fzero = 0.0
    best_fzero_harmonics = 0

    # check for integer fractions of the maximum frequency,
    # because the fundamental frequency does not need to be
    # the strongest harmonics.
    for divisor in range(1, max_divisor + 1):
        # hypothesized fundamental:
        fzero = fmax / divisor
        fzero_harmonics = 1
        # find harmonics in freqs and adjust fzero accordingly:
        hfreqs = [fmax]
        for h in range(divisor+1, 2*min_group_size+1):
            ff = freqs[np.abs(freqs[:,0]/h - fzero)<freqtol,0]
            if len(ff) == 0:
                if h > min_group_size:
                    break
                continue
            df = ff-hfreqs[-1]
            fe = np.abs(np.abs(df)/np.round(df/fzero) - fzero)
            idx = np.argmin(fe)
            if fe[idx] > 2.0*freqtol:
                if h > min_group_size:
                    break
                continue
            hfreqs.append(ff[idx])
            # update fzero:
            fzero_harmonics = h
            fzero = hfreqs[-1] / fzero_harmonics
            if verbose > 2:
                print('adjusted fzero to %.1fHz' % fzero)
        if verbose > 1:
            print('# divisor: %d, fzero=%.1fHz adjusted from harmonics %d'
                  % (divisor, fzero, fzero_harmonics))

        # collect harmonics from more_freqs:
        newmoregroup = []
        fill_ins = 0
        ndpre = 0.0  # difference of previous frequency
        # candidate frequencies as multiples of fzero within freqtol:
        hidx = np.round(more_freqs[:,0]/fzero)  # index of potential harmonics of each frequency
        fe = np.abs(more_freqs[:,0]-fzero*hidx) # deviation from harmonics
        # get harmonics from more_freqs:
        for j in np.where((hidx < 2*min_group_size)&(fe<freqtol*hidx))[0]:

            if verbose > 3:
                print('check more_freq %3d %8.2f ' % (j, more_freqs[j, 0]), end='')

            # IS FREQUENCY A AN INTEGRAL MULTIPLE OF FREQUENCY B?
            # divide the frequency-to-be-checked with fzero:
            # what is the multiplication factor between freq and fzero?
            n = hidx[j]

            # !! the difference between the detection, divided by the derived integer
            # , and fzero should be very very small: 1 resolution step of the fft
            # (more_freqs[j,0] / n) = should be fzero, plus minus a little tolerance,
            # which is the fft resolution
            nd = fe[j]

            # two succeeding frequencies should also differ by fzero plus/minus tolerance:
            if len(newmoregroup) > 0:
                nn = int(np.round((more_freqs[j,0] - more_freqs[newmoregroup[-1],0])/fzero))
                if nn == 0:
                    # current frequency is close to the same harmonic as the previous one:
                    if len(newmoregroup) > 1 and newmoregroup[-2] >= 0:
                        # check whether the current frequency is fzero apart from the previous harmonics:
                        nn = int(np.round((more_freqs[j,0] - more_freqs[newmoregroup[-2],0])/fzero))
                        nnd = np.abs(((more_freqs[j,0] - more_freqs[newmoregroup[-2],0]) / nn) - fzero)
                        if nnd > 2.0 * freqtol:
                            if verbose > 3:
                                print('discarded: distance to previous harmonics %2d %5.2f %5.2f %8.2f' %
                                      (nn, nnd, freqtol, fzero))
                            continue
                    if ndpre < nd:
                        # the previous frequency is closer to the harmonics, keep it:
                        if verbose > 3:
                            print('discarded: previous harmonics is closer %2d %5.2f %5.2f %5.2f %8.2f' %
                                  (n, nd, ndpre, freqtol, fzero))
                        continue
                    else:
                        # the current frequency is closer to the harmonics, remove the previous one:
                        newmoregroup.pop()
                else:
                    # check whether the current frequency is fzero apart from the previous harmonics:
                    nnd = np.abs(((more_freqs[j,0] - more_freqs[newmoregroup[-1],0])/nn) - fzero)
                    if nnd > 2.0 * freqtol:
                        if verbose > 3:
                            print('discarded: distance to previous harmonics %2d %5.2f %5.2f %8.2f' %
                                  (nn, nnd, freqtol, fzero))
                        continue
            ndpre = nd

            # fill in missing harmonics:
            while len(newmoregroup) < n - 1:  # while some harmonics are missing ...
                newmoregroup.append(-1)  # ... add marker for non-existent harmonic
                fill_ins += 1

            # take frequency:
            newmoregroup.append(j)
            if verbose > 3:
                print('append')

        newmoregroup = np.asarray(newmoregroup)

        # check double use of frequencies:
        double_use = np.sum(more_freqs[newmoregroup[newmoregroup>=0], 4])
        if double_use > max_double_use_count:
            if verbose > 1:
                print('discarded group because of double use:', double_use)
            continue

        # ratio of total fill-ins too large:
        if len(newmoregroup) == 0 or \
           float(fill_ins) / float(len(newmoregroup)) > max_fill_ratio:
            if verbose > 1:
                print('discarded group because of too many fill ins! %d from %d (%g)' %
                      (fill_ins, len(newmoregroup), float(fill_ins)/float(len(newmoregroup))), newmoregroup)
            continue

        # assemble newgroup from freqs:
        newgroup = []
        for mf in more_freqs[newmoregroup[newmoregroup>=0],0]:
            idx = np.argmin(np.abs(freqs[:,0] - mf))
            if np.abs(freqs[idx,0] - mf) < 1.0e-5:
                newgroup.append(idx)

        # fmax might not be in our group, because we adjust fzero:
        if not fmaxinx in newgroup:
            if verbose > 1:
                print("discarded: lost fmax")
            continue

        newmoregroup_peaksum = np.sum(more_freqs[newmoregroup[newmoregroup>0], 1])
        fills = np.sum(newmoregroup[:len(best_moregroup)] < 0)
        best_fills = np.sum(best_moregroup[:len(newmoregroup)] < 0)
        takes = np.sum(newmoregroup >= 0)
        best_takes = np.sum(best_moregroup >= 0)
        if verbose > 1:
            print('newgroup:      divisor=%d, fzero=%7.2fHz, peaksum=%7g, fills=%d, takes=%d'
                  % (divisor, fzero, newmoregroup_peaksum, fills, takes), newgroup)
            if verbose > 2:
                print('bestgroup:     divisor=%d, fzero=%7.2fHz, peaksum=%7g, fills=%d, takes=%d'
                      % (best_divisor, best_fzero, best_group_peaksum, best_fills, best_takes), best_group)

        # select new group if better than best group:
        # sum of peak power must be larger and
        # less fills. But if the new group has more takes,
        # this might compensate for more fills.
        if newmoregroup_peaksum > best_group_peaksum \
                and fills - best_fills <= 0.5 * (takes - best_takes):
            best_group_peaksum = newmoregroup_peaksum
            best_group = np.asarray(newgroup)
            best_moregroup = np.asarray(newmoregroup)
            best_divisor = divisor
            best_fzero = fzero
            best_fzero_harmonics = fzero_harmonics
            if verbose > 2:
                print('new bestgroup: divisor=%d, fzero=%7.2fHz, peaksum=%7g, fills=%d, takes=%d'
                      % (best_divisor, best_fzero, best_group_peaksum, best_fills, best_takes), best_group)
            elif verbose > 1:
                print('took as new best group')

    # ##############################################################

    # no group found:
    if len(best_group) == 0:
        # erase fmax:
        freqs = np.delete(freqs, fmaxinx, axis=0)
        group = np.zeros((0, 5))
        return freqs, more_freqs, group, 1, fmax

    # group found:
    if verbose > 2:
        print('')
        print('## best groups found for fmax=%.2fHz: fzero=%.2fHz, divisor=%d:'
              % (fmax, best_fzero, best_divisor))
        print('## bestgroup:     ', best_group,
              '[', ', '.join(['%.2f' % f for f in freqs[best_group,0]]), ']')
        print('## bestmoregroup: ', best_moregroup,
              '[', ', '.join(['%.2f' % f for f in more_freqs[best_moregroup,0]]), ']')

    # increment double use count:
    more_freqs[best_moregroup[best_moregroup>=0], 4] += 1.0
    # fill up group:
    group = more_freqs[best_moregroup[best_moregroup>=0],:]
    group[:,0] = np.arange(1,len(group)+1)*best_fzero

    if verbose > 1:
        refi = np.argmax(group[:,1] > 0.0)
        print('')
        print('# resulting harmonic group for fmax=', fmax)
        for i in range(len(group)):
            print('f=%8.2fHz n=%5.2f: power=%10.3g power/p0=%10.3g'
                  % (group[i,0], group[i,0]/group[0,0], group[i,1], group[i,1]/group[refi,1]))

    # erase from freqs:
    freqs = np.delete(freqs, best_group, axis=0)

    # freqs: removed all frequencies of bestgroup
    # more_freqs: updated double use count
    return freqs, more_freqs, group, best_fzero_harmonics, fmax


def build_harmonic_group_new(freqs, more_freqs, deltaf, verbose=0,
                         min_freq=20.0, max_freq=2000.0, min_group_size=4,
                         freq_tol_fac=1.0, max_divisor=4, max_upper_fill=1,
                         ax_double_use_count=1,
                         max_fill_ratio=0.25, power_n_harmonics=10, **kwargs):
    """Extract harmonics belonging to the largest peak in a list of frequency peaks.

    Parameters
    ----------
    freqs: 2-D array
        List of frequency, power, size, width and count of strong peaks in a power spectrum.
    more_freqs:
        List of frequency, power, size, width and count of all peaks in a power spectrum.
    deltaf: float
        Frequency resolution of the power spectrum.
    verbose: int
        Verbosity level.
    min_freq: float
        Minimum frequency accepted as a fundamental frequency.
    max_freq: float
        Maximum frequency accepted as a fundamental frequency.
    min_group_size: int
        Minimum number of harmonics required to be present, i.e.
        within min_group_size no harmonics are allowed to be filled in.
    freq_tol_fac: float
        Harmonics need to fall within deltaf*freq_tol_fac.
    max_divisor: int
        Maximum divisor used for checking for sub-harmonics.
    max_upper_fill: int
        Maximum number of frequencies that are allowed to be filled in
        (i.e. they are not contained in more_freqs) above the frequency of the
        largest peak in freqs for constructing a harmonic group.
    max_double_use_count: int
        Maximum number of harmonic groups a single peak can be part of.
    max_fill_ratio: float
        Maximum allowed fraction of filled in frequencies.
    power_n_harmonics: int
        Maximum number of harmonics over which the total power of the signal is computed.
    
    Returns
    -------
    freqs: 2-D array
        List of strong frequencies with the frequencies of group removed.
    more_freqs: 2-D array
        List of all frequencies with updated double-use counts.
    group: 2-D array
        The detected harmonic group. Might be empty.
    best_fzero_harmonics: int
        The highest harmonics that was used to recompute the fundamental frequency.
    fmax: float
        The frequency of the largest peak in freqs for which the harmonic group was detected.
    """

    def select_harmonics(freqs, fzero, freqtol):
        """From a list of frequencies return the ones that are closest to multiples of fzero.
        """
        if len(freqs) == 0:
            return np.array([])
        # candidate good frequencies as multiples of fzero within freqtol:
        hidx = np.round(freqs[:,0]/fzero)  # index of potential harmonics of each frequency
        fe = np.abs(freqs[:,0]-fzero*hidx) # deviation from harmonics
        sel = (fe<freqtol*hidx)            # select the ones within the tolerance
        hidx = hidx[sel]
        zfreqs = freqs[sel,0]
        if len(zfreqs) < 2:
            return zfreqs
        n = np.argmax(np.diff(hidx) > 1)
        if n > 0:
            hidx = hidx[:n+1]
            zfreqs = zfreqs[:n+1]
        # take single frequencies closest to harmonics:
        hn = int(np.ceil(zfreqs[-1]/fzero))
        hfreqs = np.array([zfreqs[hidx==h][np.argmin(fe[hidx==h])]
                           for h in range(1, hn+1) if np.sum(hidx==h)>0])
        # differences between subsequent harmonics should also be spaced from fzero by less than twice the tolerance:
        dfreqs = np.diff(hfreqs)
        hfreqs = np.hstack((hfreqs[0], hfreqs[1:][np.abs(dfreqs/np.round(dfreqs/fzero) - fzero) < 2.0 * freqtol]))
        return hfreqs
    
    def select_harmonics1(freqs, fzero, freqtol, hn=None):
        """From a list of frequencies return the ones that are closest to multiples of fzero.
        """
        newgroup = list()
        npre = -1  # previous harmonics
        ndpre = 0.0  # difference of previous frequency
        connected = True
        for j in range(freqs.shape[0]):
            # IS THE CURRENT FREQUENCY AN INTEGRAL MULTIPLE OF FZERO?
            # divide the frequency-to-be-checked by fzero
            # to get the multiplication factor between freq and fzero
            n = int(np.round(freqs[j, 0] / fzero))
            if n == 0:
                if verbose > 2:
                    print('discarded: n == 0')
                continue

            # !! the difference between the current frequency, divided by the derived integer,
            # and fzero should be very very small: 1 resolution step of the fft
            # (freqs[j,0] / n) = should be fzero, plus minus a little tolerance,
            # which is the fft resolution
            nd = np.abs((freqs[j, 0] / n) - fzero)
            # ... compare it to our tolerance
            if nd > freqtol:
                if verbose > 2:
                    print('discarded: not a harmonic n=%2d d=%5.2fHz tol=%5.2fHz' % (n, nd, freqtol))
                continue

            # two succeeding frequencies should also differ by
            # fzero plus/minus twice! the tolerance:
            if len(newgroup) > 0:
                nn = int(np.round((freqs[j, 0] - freqs[newgroup[-1], 0]) / fzero))
                if nn == 0:
                    # the current frequency is the same harmonic as the previous one
                    # print(divisor, j, freqs[j,0], freqs[newgroup[-1],0])
                    if len(newgroup) > 1:
                        # check whether the current frequency is fzero apart from the previous harmonics:
                        nn = int(np.round((freqs[j, 0] - freqs[newgroup[-2], 0]) / fzero))
                        nnd = np.abs(((freqs[j, 0] - freqs[newgroup[-2], 0]) / nn) - fzero)
                        if nnd > 2.0 * freqtol:
                            if verbose > 2:
                                print('discarded: distance to previous harmonics %2d %5.2f %5.2f %8.2f'
                                      % (nn, nnd, freqtol, fzero))
                            continue
                    if ndpre < nd:
                        # the previous frequency is closer to the harmonics, keep it:
                        if verbose > 2:
                            print('discarded: previous harmonics is closer %2d %5.2f %5.2f %5.2f %8.2f' %
                                  (n, nd, ndpre, freqtol, fzero))
                        continue
                    else:
                        # the current frequency is closer to the harmonics, remove the previous one:
                        newgroup.pop()
                else:
                    # check whether the current frequency is fzero apart from the previous harmonics:
                    nnd = np.abs(((freqs[j, 0] - freqs[newgroup[-1], 0]) / nn) - fzero)
                    if nnd > 2.0 * freqtol:
                        if verbose > 2:
                            print('discarded: distance to previous harmonics %2d %5.2f %5.2f %8.2f' %
                                  (nn, nnd, freqtol, fzero))
                        continue

            # take frequency:
            newgroup.append(j)  # append index of frequency
            if verbose > 2:
                print('append n={:.2f} d={:5.2f}Hz tol={:5.2f}Hz'.format(freqs[j, 0] / fzero, nd, freqtol))

            if npre >= 0 and n - npre > 1:
                connected = False
            npre = n
            ndpre = nd

            if connected:
                # adjust fzero as we get more information from the higher frequencies:
                fzero = freqs[j, 0] / n
                fzero_harmonics = int(n)
                if verbose > 2:
                    print('adjusted fzero to', fzero)
        return freqs[newgroup,0]

    # frequency tolerance:
    freqtol = freq_tol_fac * deltaf
    
    # select strongest frequency for building the harmonic group:
    fmaxinx = np.argmax(freqs[:, 1])
    fmax = freqs[fmaxinx, 0]
    if verbose > 1:
        print('')
        print(70 * '#')
        print('frequency tolerance: %gHz' % freqtol)
        print('freqs:     ', '[', ', '.join(['{:.2f}'.format(f) for f in freqs[:, 0]]), ']')
        print('more_freqs:', '[', ', '.join(
            ['{:.2f}'.format(f) for f in more_freqs[:, 0] if f < max_freq]), ']')
        print('## fmax is: {0: .2f}Hz @ {1:.5g}dB ##\n'.format(fmax, np.max(freqs[:, 1])))

    # container for harmonic groups:
    best_group = []
    best_moregroup = []
    best_moreharms = []
    best_group_peaksum = 0.0
    best_divisor = 0
    best_fzero = 0.0
    best_fzero_harmonics = 0

    # check for integer fractions of the maximum frequency,
    # because the fundamental frequency does not need to be
    # the strongest harmonics.
    for divisor in range(1, max_divisor + 1):
        # hypothesized fundamental:
        fzero = fmax / divisor
        fzero_harmonics = 1
        # find harmonics in freqs and adjust fzero accordingly:
        hfreqs = [fmax]
        for h in range(divisor+1, min_group_size+1):
            ff = freqs[np.abs(freqs[:,0]/h - fzero)<freqtol,0]
            if len(ff) == 0:
                continue
            df = ff-hfreqs[-1]
            fe = np.abs(np.abs(df)/np.round(df/fzero) - fzero)
            idx = np.argmin(fe)
            if fe[idx] > 2.0*freqtol:
                continue
            hfreqs.append(ff[idx])
            # update fzero:
            fzero_harmonics = h
            fzero = hfreqs[-1] / fzero_harmonics
            if verbose > 2:
                print('adjusted fzero to %.1fHz' % fzero)
        if verbose > 1:
            print('# divisor: %d, fzero=%.1fHz adjusted from harmonics %d' % (divisor, fzero, fzero_harmonics))

        # get all harmonics from more_freqs:
        mfreqs = select_harmonics(more_freqs, fzero, freqtol)
        mharms = np.round(mfreqs/fzero)
        # no missing harmonics allowed within min_group_size:
        if len(mharms) < min_group_size or mharms[min_group_size-1] < min_group_size:
            if verbose > 1:
                print('discarded group: missing harmonics below the %d-th one!' % min_group_size)
            continue
        # fraction of missing harmonics too large:
        # XXX compute only for upto
        # if more_freqs[j, 0] > fmax and n - 1 - len(newmoregroup) > max_upper_fill:
        missing = float(len(mharms))/float(mharms[-1])
        if missing > max_fill_ratio:
            if verbose > 1:
                print('discarded group: %d harmonics of %d are missing (%.1f%% > %.1f%%)!' %
                      (mharms[-1]-len(mharms), mharms[-1], 100.0*missing, 100.0*max_fill_ratio))
            continue
        # check double use of frequencies:
        double_use = np.sum([1 for mf in mfreqs[:2*min_group_size]
                             if more_freqs[np.argmin(np.abs(more_freqs[:,0] - mf)),4] > 0])
        if double_use > max_double_use_count:
            if verbose > 1:
                print('discarded group: double use count %d is larger than %d:'
                      % (double_use, max_double_use_count))
            continue
        # fmax might not be in our group, because we adjusted fzero:
        if np.min(np.abs(mfreqs - fmax)) > freqtol:
            if verbose > 1:
                print('discarded group: lost fmax (fmax=%.1fHz, closest frequency=%.1fHz)'
                      % (fmax, mfreqs[np.argmin(np.abs(mfreqs - fmax))]))
            continue

        # assemble indices to frequencies:
        newgroup = []
        newmoregroup = []
        for mf in mfreqs:
            idx = np.argmin(np.abs(freqs[:,0] - mf))
            if np.abs(freqs[idx,0] - mf) < 1.0e-5:
                newgroup.append(idx)
            idx = np.argmin(np.abs(more_freqs[:,0] - mf))
            if np.abs(more_freqs[idx,0] - mf) < 1.0e-5:
                newmoregroup.append(idx)
        newgroup = np.asarray(newgroup)
        newmoregroup = np.asarray(newmoregroup)

        # properties of the new group:
        newmoregroup_peaksum = np.sum(more_freqs[newmoregroup[mharms<=min_group_size], 1])
        # XXX better to compare fills over the same frequency range!
        if min(len(best_moregroup), len(newmoregroup)) > 0:
            ng = newmoregroup[:len(best_moregroup)]
            fills = np.round(more_freqs[ng[-1],0]/fzero) - len(ng)
            bg = best_moregroup[:len(newmoregroup)]
            best_fills = np.round(more_freqs[bg[-1],0]/fzero) - len(bg)
        else:
            fills = 0
            best_fills = 0
        takes = len(newmoregroup)
        best_takes = len(best_moregroup)
        if verbose > 1:
            print('newgroup:      divisor=%d, fzero=%7.2fHz, peaksum=%7g, fills=%d, takes=%d'
                  % (divisor, fzero, newmoregroup_peaksum, fills, takes), newgroup)
            if verbose > 2:
                print('bestgroup:     divisor=%d, fzero=%7.2fHz, peaksum=%7g, fills=%d, takes=%d'
                      % (best_divisor, best_fzero, best_group_peaksum, best_fills, best_takes), best_group)

        # select new group if better than best group:
        # sum of peak power must be larger and
        # less fills. But if the new group has more takes,
        # this might compensate for more fills.
        if newmoregroup_peaksum > best_group_peaksum \
                and fills - best_fills <= 0.5 * (takes - best_takes):
            best_group_peaksum = newmoregroup_peaksum
            best_group = newgroup
            best_moregroup = newmoregroup
            best_moreharms = mharms
            best_divisor = divisor
            best_fzero = fzero
            best_fzero_harmonics = fzero_harmonics
            if verbose > 2:
                print('new bestgroup: divisor=%d, fzero=%7.2fHz, peaksum=%7g, fills=%d, takes=%d'
                      % (best_divisor, best_fzero, best_group_peaksum, best_fills, best_takes), best_group)
            elif verbose > 1:
                print('took as new best group')

    # ##############################################################

    # no group found:
    if len(best_group) == 0:
        # erase fmax:
        freqs = np.delete(freqs, fmaxinx, axis=0)
        group = np.zeros((0, 5))
        return freqs, more_freqs, group, 1, fmax

    # group found:
    if verbose > 2:
        print('')
        print('## best groups found for fmax=%.2fHz: fzero=%.2fHz, divisor=%d:'
              % (fmax, best_fzero, best_divisor))
        print('## bestgroup:     ', best_group,
              '[', ', '.join(['%.2f' % f for f in freqs[best_group,0]]), ']')
        print('## bestmoregroup: ', best_moregroup,
              '[', ', '.join(['%.2f' % f for f in more_freqs[best_moregroup,0]]), ']')

    # increment double use count:
    more_freqs[best_moregroup, 4] += 1.0
    # fill up group:
    group = more_freqs[best_moregroup,:]
    group[:,0] = best_moreharms*best_fzero

    if verbose > 1:
        refi = np.argmax(group[:,1] > 0.0)
        print('')
        print('# resulting harmonic group for fmax=', fmax)
        for i in range(len(group)):
            print('f=%8.2fHz n=%5.2f: power=%10.3g power/p0=%10.3g'
                  % (group[i,0], group[i,0]/group[0,0], group[i,1], group[i,1]/group[refi,1]))

    # erase from freqs:
    freqs = np.delete(freqs, best_group, axis=0)

    # freqs: removed all frequencies of bestgroup
    # more_freqs: updated double use count
    return freqs, more_freqs, group, best_fzero_harmonics, fmax


def extract_fundamentals(good_freqs, all_freqs, deltaf, verbose=0,
                         freq_tol_fac=1.0,
                         mains_freq=60.0, min_freq=0.0, max_freq=2000.0,
                         max_divisor=4, max_double_use_count=1,
                         max_fill_ratio=0.25,
                         min_group_size=4, max_harmonics=0, max_groups=0, **kwargs):
    """Extract fundamental frequencies from power-spectrum peaks.
                         
    Parameters
    ----------
    good_freqs: 2-D array
        List of frequency, power, size, width and count of strong peaks in a power spectrum.
    all_freqs: 2-D array
        List of frequency, power, size, width and count of all peaks in a power spectrum.
    deltaf: float
        Frequency resolution of the power spectrum.
    verbose: int
        Verbosity level.
    freq_tol_fac: float
        Harmonics need to fall within `deltaf*freq_tol_fac`.
    mains_freq: float
        Frequency of the mains power supply.
    min_freq: float
        Minimum frequency accepted as a fundamental frequency.
    max_freq: float
        Maximum frequency accepted as a fundamental frequency.
    max_divisor: int
        Maximum divisor used for checking for sub-harmonics.
    max_double_use_count: int
        Maximum number of harmonic groups a single peak can be part of.
    max_fill_ratio: float
        Maximum allowed fraction of filled in frequencies.
    min_group_size: int
        Within min_group_size no harmonics are allowed to be filled in.
    max_harmonics: int
        Maximum number of harmonics to be returned for each group.
    max_groups: int
        If not zero the maximum number of most powerful harmonic groups.

    Returns
    -------
    group_list: list of 2-D arrays
        List of all harmonic groups found sorted by fundamental frequency.
        Each harmonic group is a 2-D array with the first dimension the harmonics
        and the second dimension containing frequency, power, and size of each harmonic.
        If the power is zero, there was no corresponding peak in the power spectrum.
    fzero_harmonics_list: list of int
        The harmonics from which the fundamental frequencies were computed.
    mains_list: 2-d array
        Array of mains peaks found in all_freqs (frequency, power, size).
    """
    if verbose > 0:
        print('')

    # set double use count to zero:
    all_freqs[:, 4] = 0.0

    # frequency tolerance:
    freqtol = freq_tol_fac * deltaf

    # remove power line harmonics from good_freqs:
    # XXX might be improved!!!
    if mains_freq > 0.0:
        pfreqtol = 1.0  # 1 Hz tolerance
        for inx in reversed(range(len(good_freqs))):
            n = int(np.round(good_freqs[inx, 0] / mains_freq))
            nd = np.abs(good_freqs[inx, 0] - n * mains_freq)
            if nd <= pfreqtol:
                if verbose > 1:
                    print('remove power line frequency', inx, good_freqs[inx, 0], np.abs(
                        good_freqs[inx, 0] - n * mains_freq))
                good_freqs = np.delete(good_freqs, inx, axis=0)

    group_list = list()
    fzero_harmonics_list = list()
    # as long as there are frequencies left in good_freqs:
    while good_freqs.shape[0] > 0:
        # we check for harmonic groups:
        good_freqs, all_freqs, harm_group, fzero_harmonics, fmax = \
            build_harmonic_group(good_freqs, all_freqs, freqtol,
                                verbose, min_group_size, max_divisor,
                                max_double_use_count, max_fill_ratio)

        if verbose > 1:
            print('')

        # nothing found:
        if harm_group.shape[0] == 0:
            if verbose > 0:
                print('Nothing found for fmax=%.2fHz' % fmax)
            continue

        # within min_group_size we do not want fill ins:
        group_size_ok = np.sum(harm_group[:min_group_size, 1] > 0.0) == min_group_size

        # check frequency range of fundamental:
        fundamental_ok = (harm_group[0, 0] >= min_freq and
                          harm_group[0, 0] <= max_freq)

        # check power hum (does this really ever happen???):
        mains_ok = ((mains_freq == 0.0) |
                    (np.abs(harm_group[0, 0] - mains_freq) > freqtol))

        # check:
        if group_size_ok and fundamental_ok and mains_ok:
            if verbose > 0:
                print('Accepting harmonic group: {:.2f}Hz p={:10.8f}'.format(
                    harm_group[0, 0], np.sum(harm_group[:, 1])))

            group_list.append(harm_group[:, 0:2])
            fzero_harmonics_list.append(fzero_harmonics)
        else:
            if verbose > 0:
                print('Discarded harmonic group: {:.2f}Hz p={:10.8f} g={:d} f={:} m={:}'.format(
                    harm_group[0, 0], np.sum(harm_group[:, 1]),
                    group_size_ok, fundamental_ok, mains_ok))

    # do not save more than n harmonics:
    if max_harmonics > 0:
        for group in group_list:
            if group.shape[0] > max_harmonics:
                if verbose > 1:
                    print('Discarding harmonics higher than %d for f=%.2fHz' % (max_harmonics, group[0, 0]))
                group = group[:max_harmonics, :]

    # select most powerful harmonic groups:
    if max_groups > 0:
        powers = [np.sum(group[:, 1]) for group in group_list]
        powers_inx = np.argsort(powers)
        group_list = [group_list[pi] for pi in powers_inx[-max_groups:]]
        fzero_harmonics_list = [fzero_harmonics_list[pi] for pi in powers_inx[-max_groups:]]
        if verbose > 0:
            print('Selected the %d most powerful groups.' % max_groups)
        
    # sort groups by fundamental frequency:
    ffreqs = [f[0, 0] for f in group_list]
    finx = np.argsort(ffreqs)
    group_list = [group_list[fi] for fi in finx]
    fzero_harmonics_list = [fzero_harmonics_list[fi] for fi in finx]

    if verbose > 0:
        print('')
        if len(group_list) > 0:
            print('## FUNDAMENTALS FOUND: ##')
            for i in range(len(group_list)):
                powers = group_list[i][:, 1]
                print('{:8.2f}Hz: {:10.8f} {:3d} {:3d}'.format(group_list[i][0, 0], np.sum(powers),
                                                               np.sum(powers <= 0.0), fzero_harmonics_list[i]))
        else:
            print('## NO FUNDAMENTALS FOUND ##')

    # assemble mains frequencies from all_freqs:
    mains_list = []
    if mains_freq > 0.0:
        pfreqtol = 1.0
        for inx in range(len(all_freqs)):
            n = int(np.round(all_freqs[inx, 0] / mains_freq))
            nd = np.abs(all_freqs[inx, 0] - n * mains_freq)
            if nd <= pfreqtol:
                mains_list.append(all_freqs[inx, 0:2])
                
    return group_list, fzero_harmonics_list, np.array(mains_list)


def threshold_estimate(psd_data, low_thresh_factor=6.0, high_thresh_factor=10.0,
                       nbins=100, hist_height=1.0/ np.sqrt(np.e)):
    """Estimate thresholds for peak detection from histogram of power spectrum.

    The standard deviation of the noise floor without peaks is estimated from
    the width of the histogram of the power spectrum at `hist_height` relative height.
    The histogram is computed in the third quarter of the linearly detrended power spectrum.

    Parameters
    ----------
    psd_data: array
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


def harmonic_groups(psd_freqs, psd, verbose=0, low_threshold=0.0, high_threshold=0.0,
                    thresh_bins=100, low_thresh_factor=6.0, high_thresh_factor=10.0,
                    max_peak_width_fac=20.0, min_peak_width=1.0,
                    freq_tol_fac=1.0, mains_freq=60.0, min_freq=0.0, max_freq=2000.0,
                    max_divisor=4, max_double_use_count=1, max_fill_ratio=0.25,
                    min_group_size=4, max_harmonics=0, max_groups=0, **kwargs):
    """Detect peaks in power spectrum and group them according to their harmonic structure.

    Parameters
    ----------
    psd_freqs: array
        Frequencies of the power spectrum.
    psd: array
        Power spectrum (linear, not decible).
    verbose: int
        Verbosity level.
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
    max_peak_width_fac: float
        The maximum allowed width at 0.75 height of a good peak in the decibel power spectrum
        in multiples of the frequency resolution.
    min_peak_width: float
        The minimum absolute value for the maximum width of a peak in Hertz.
    freq_tol_fac: float
        Harmonics need to fall within `deltaf*freq_tol_fac`.
    mains_freq: float
        Frequency of the mains power supply.
    min_freq: float
        Minimum frequency accepted as a fundamental frequency.
    max_freq: float
        Maximum frequency accepted as a fundamental frequency.
    max_divisor: int
        Maximum divisor used for checking for sub-harmonics.
    max_double_use_count: int
        Maximum number of harmonic groups a single peak can be part of.
    max_fill_ratio: float
        Maximum allowed fraction of filled in frequencies.
    min_group_size: int
        Within min_group_size no harmonics are allowed to be filled in.
    max_harmonics: int
        Maximum number of harmonics to be returned for each group.
    max_groups: int
        If not zero the maximum number of most powerful harmonic groups.

    Returns
    -------
    group_list: list of 2-D arrays
        List of all extracted harmonic groups, sorted by fundamental frequency.
        Each harmonic group is a 2-D array with the first dimension the harmonics
        and the second dimension containing frequency, power, and size of each harmonic.
        If the power is zero, there was no corresponding peak in the power spectrum.
    fzero_harmonics: list of ints
        The harmonics from which the fundamental frequencies were computed.
    mains: 2-d array
        Frequencies and power of multiples of the mains frequency found in the power spectrum.
    all_freqs: 2-d array
        Peaks in the power spectrum detected with low threshold
        [frequency, power, size, width, double use count].
    good_freqs: 1-d array
        Frequencies of peaks detected with high threshold.
    low_threshold: float
        The relative threshold for detecting all peaks in the decibel spectrum.
    high_threshold: float
        The relative threshold for detecting good peaks in the decibel spectrum.
    center: float
        The baseline level of the power spectrum.
    """
    if verbose > 0:
        print('')
        print(70 * '#')
        print('##### harmonic_groups', 48 * '#')

    # decibel power spectrum:
    log_psd = decibel(psd)
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
            print('low_threshold =%g  center+low_threshold =%g' % (low_threshold, center + low_threshold))
            print('high_threshold=%g  center+high_threshold=%g' % (high_threshold, center + high_threshold))
            print('center=%g' % center)

    # detect peaks in decibel power spectrum:
    peaks, troughs = detect_peaks(log_psd, low_threshold)
    all_freqs = peak_size_width(psd_freqs, log_psd, peaks, troughs, 0.75)

    if len(all_freqs) == 0:
        return [], [], [], np.zeros((0, 5)), [], low_threshold, high_threshold, center

    # maximum width of a frequency peak:
    wthresh = max_peak_width_fac * delta_f
    if wthresh < min_peak_width:
        wthresh = min_peak_width
        
    # select good peaks:
    freqs = all_freqs[(all_freqs[:,2] > high_threshold) &
                      (all_freqs[:,0] >= min_freq) &
                      (all_freqs[:,0] <= 2*min_group_size*max_freq) &
                      (all_freqs[:,3] < wthresh), :]

    # convert peak sizes back to power:
    freqs[:, 1] = power(freqs[:, 1])
    all_freqs[:, 1] = power(all_freqs[:, 1])

    # detect harmonic groups:
    groups, fzero_harmonics, mains = \
      extract_fundamentals(freqs, all_freqs, delta_f, verbose, freq_tol_fac,
                           mains_freq, min_freq, max_freq,
                           max_divisor, max_double_use_count,
                           max_fill_ratio, min_group_size,
                           max_harmonics, max_groups)

    return (groups, fzero_harmonics, mains, all_freqs, freqs[:, 0],
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
    freqs: list of 2D ndarrays
        First column in the ndarrays is fundamental frequency and
        second column the corresponding power.
        Further columns are optional and kept in the returned list.
        fundamental_freqs_and_power() returns such a list.

    Returns
    -------
    power_freqs: list of 2D ndarrays
        Same as freqs, but with an added column containing the relative power.
    """
    return [np.column_stack((f, f[:,1] - np.max(f[:,1]))) for f in freqs]


def add_power_ranks(freqs):
    """ Add a column with power ranks.

    Parameters
    ----------
    freqs: list of 2D ndarrays
        First column in the ndarrays is fundamental frequency and
        second column the corresponding power.
        Further columns are optional and kept in the returned list.
        fundamental_freqs_and_power() returns such a list.

    Returns
    -------
    rank_freqs: list of 2D ndarrays
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
    freqs: (list of (list of ...)) list of 2D ndarrays
        First column in the ndarrays is fundamental frequency.
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
    freqs: list of 2D ndarrays
        First column in the ndarrays is fundamental frequency and
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
    freqs: (list of (list of ...)) list of 2D ndarrays
        First column in the ndarrays is fundamental frequency and
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
    uniqe_freqs: (list of (list of ...)) list of 2D ndarrays
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
    cc0 = cm.gist_rainbow(np.linspace(0.0, 1.0, 8.0))
    # shuffle it:
    for k in range((len(cc0) + 1) // 2):
        colors.extend(cc0[k::(len(cc0) + 1) // 2])
    markers.extend(len(cc0) * 'o')
    mr2.extend(len(cc0) * 'v')
    # second darker color range:
    cc1 = cm.gist_rainbow(np.linspace(0.33 / 7.0, 1.0, 7.0))
    cc1 = mc.hsv_to_rgb(mc.rgb_to_hsv(np.array([cc1[:, :3]])) * np.array([1.0, 0.9, 0.7]))[0]
    cc1 = np.hstack((cc1, np.ones((len(cc1),1))))
    # shuffle it:
    for k in range((len(cc1) + 1) // 2):
        colors.extend(cc1[k::(len(cc1) + 1) // 2])
    markers.extend(len(cc1) * '^')
    mr2.extend(len(cc1) * '*')
    # third lighter color range:
    cc2 = cm.gist_rainbow(np.linspace(0.67 / 6.0, 1.0, 6.0))
    cc2 = mc.hsv_to_rgb(mc.rgb_to_hsv(np.array([cc1[:, :3]])) * np.array([1.0, 0.5, 1.0]))[0]
    cc2 = np.hstack((cc2, np.ones((len(cc2),1))))
    # shuffle it:
    for k in range((len(cc2) + 1) // 2):
        colors.extend(cc2[k::(len(cc2) + 1) // 2])
    markers.extend(len(cc2) * 'D')
    mr2.extend(len(cc2) * 'x')
    markers.extend(mr2)
    return colors, markers


def plot_harmonic_groups(ax, group_list, max_freq=2000.0, max_groups=0, sort_by_freq=True,
                         label_power=False, colors=None, markers=None, legend_rows=8, **kwargs):
    """
    Mark decibel power of fundamentals and their harmonics in a plot.

    Parameters
    ----------
    ax: axis for plot
            Axis used for plotting.
    group_list: list of 2-D arrays
            Lists of harmonic groups as returned by extract_fundamentals() and
            harmonic_groups() with the element [0, 0] of the harmonic groups being the fundamental frequency,
            and element[0, 1] being the corresponding power.
    max_freq: float
        If greater than zero only mark peaks below this frequency.
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
    powers = np.array([np.sum(fish[:,1]) for fish in group_list])
    max_power = np.max(powers)
    idx_maxpower = np.argsort(powers)
    if max_groups > 0 and len(idx_maxpower > max_groups):
        idx_maxpower = idx_maxpower[-max_groups:]
    idx = np.array(list(reversed(idx_maxpower)))

    # sort by frequency:
    if sort_by_freq:
        freqs = [group_list[group][0, 0] for group in idx]
        idx = idx[np.argsort(freqs)]

    # plot:
    for k, i in enumerate(idx):
        group = group_list[i]
        x = np.array([harmonic[0] for harmonic in group])
        y = np.array([harmonic[1] for harmonic in group])
        if max_freq > 0.0:
            y = y[x<=max_freq]
            x = x[x<=max_freq]
        msize = 7.0 + 10.0 * (powers[i] / max_power) ** 0.25
        color_kwargs = {}
        if colors is not None:
            color_kwargs = {'color': colors[k%len(colors)]}
        label = '%6.1f Hz' % group[0, 0]
        if label_power:
            label += ' %6.1f dB' % decibel(np.array([np.sum(group[:,1])]))[0]
        if markers is None:
            ax.plot(x, decibel(y), 'o', ms=msize, label=label,
                    clip_on=False, **color_kwargs)
        else:
            if k >= len(markers):
                break
            ax.plot(x, decibel(y), linestyle='None', marker=markers[k], mec=None, mew=0.0,
                    ms=msize, label=label, clip_on=False, **color_kwargs)

    # legend:
    if legend_rows > 0:
        ncol = (len(idx)-1) // legend_rows + 1
        leg = ax.legend(numpoints=1, ncol=ncol, **kwargs)
    else:
        leg = ax.legend(numpoints=1, **kwargs)


def plot_psd_harmonic_groups(ax, psd_freqs, psd, group_list, mains=None, all_freqs=None, good_freqs=None,
                             max_freq=2000.0):
    """
    Plot decibel power-spectrum with detected peaks, harmonic groups, and mains frequencies.
    
    Parameters:
    -----------
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
    max_freq: float
        Limits of frequency axis are set to (0, max_freq) if max_freq is greater than zero.
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
    plot_harmonic_groups(ax, group_list, max_freq=max_freq, max_groups=0, sort_by_freq=True,
                         colors=colors, markers=markers, legend_rows=8,
                         loc='upper right')
    # plot power spectrum:
    plot_decibel_psd(ax, psd_freqs, psd, max_freq=max_freq, color='blue')

    
def add_psd_peak_detection_config(cfg, low_threshold=0.0, high_threshold=0.0,
                                  thresh_bins=100,
                                  low_thresh_factor=6.0, high_thresh_factor=10.0,
                                  max_peak_width_fac=20.0, min_peak_width=1.0):
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
    # cfg['lowThreshold'][0] = 12. # panama
    # cfg['highThreshold'][0] = 18. # panama
    
    cfg.add_section('Threshold estimation:\nIf no thresholds are specified they are estimated from the histogram of the decibel power spectrum.')
    cfg.add('thresholdBins', thresh_bins, '', 'Number of bins used to compute the histogram used for threshold estimation.')
    cfg.add('lowThresholdFactor', low_thresh_factor, '', 'Factor for multiplying standard deviation of noise floor for lower threshold.')
    cfg.add('highThresholdFactor', high_thresh_factor, '', 'Factor for multiplying standard deviation of noise floor for higher threshold.')

    cfg.add_section('Peak detection in decibel power spectrum:')
    cfg.add('maxPeakWidthFac', max_peak_width_fac, '',
            'Maximum width of peaks at 0.75 hight in multiples of frequency resolution.')
    cfg.add('minPeakWidth', min_peak_width, 'Hz', 'Peaks do never need to be narrower than this.')


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
                    'high_thresh_factor': 'highThresholdFactor',
                    'max_peak_width_fac': 'maxPeakWidthFac',
                    'min_peak_width': 'minPeakWidth'})


def add_harmonic_groups_config(cfg, mains_freq=60.0, max_divisor=4, freq_tol_fac=1.0,
                               max_fill_ratio=0.25, max_double_use_count=1, min_group_size=4,
                               min_freq=20.0, max_freq=2000.0,
                               max_harmonics=0, max_groups=0):
    """ Add parameter needed for detection of harmonic groups as
    a new section to a configuration.

    Parameters
    ----------
    cfg: ConfigFile
        The configuration.
    """
    
    cfg.add_section('Harmonic groups:')
    cfg.add('mainsFreq', mains_freq, 'Hz', 'Mains frequency to be excluded.')
    cfg.add('maxDivisor', max_divisor, '', 'Maximum ratio between the frequency of the largest peak and its fundamental')
    cfg.add('freqTolerance', freq_tol_fac, '',
            'Harmonics need be within this factor times the frequency resolution of the power spectrum. Needs to be higher than 0.5!')
    cfg.add('maxFillRatio', max_fill_ratio, '',
            'Maximum fraction of filled in harmonics allowed (usefull values are smaller than 0.5)')
    cfg.add('maxDoubleUseCount', max_double_use_count, '', 'Maximum overall double use count allowed.')
    
    cfg.add_section('Acceptance of best harmonic groups:')
    cfg.add('minimumGroupSize', min_group_size, '',
'The number of harmonics (inclusively fundamental) that are allowed do be filled in.')
    cfg.add('minimumFrequency', min_freq, 'Hz', 'Minimum frequency allowed for the fundamental.')
    cfg.add('maximumFrequency', max_freq, 'Hz', 'Maximum frequency allowed for the fundamental.')
    cfg.add('maxHarmonics', max_harmonics, '', '0: keep all, >0 only keep the first # harmonics.')
    cfg.add('maxGroups', max_groups, '', 'Maximum number of harmonic groups. If 0 process all.')


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
                    'max_divisor': 'maxDivisor',
                    'freq_tol_fac': 'freqTolerance',
                    'max_fill_ratio': 'maxFillRatio',
                    'max_double_use_count': 'maxDoubleUseCount',
                    'min_group_size': 'minimumGroupSize',
                    'min_freq': 'minimumFrequency',
                    'max_freq': 'maximumFrequency',
                    'max_harmonics': 'maxHarmonics',
                    'max_groups': 'maxGroups'})


if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt
    from .fakefish import generate_wavefish
    from .powerspectrum import psd

    print("Checking harmonicgroups module ...\n")

    if len(sys.argv) < 2:
        # generate data:
        title = 'simulation'
        samplerate = 44100.0
        eodfs = [123.0, 321.0, 666.0, 668.0]
        fish1 = generate_wavefish(eodfs[0], samplerate, duration=8.0, noise_std=0.01,
                                  amplitudes=[0.5, 0.7, 0.3, 0.1, 0.05], phases=[0.0, 0.0, 0.0, 0.0, 0.0])
        fish2 = generate_wavefish(eodfs[1], samplerate, duration=8.0, noise_std=0.01,
                                  amplitudes=[1.0, 0.7, 0.2, 0.1], phases=[0.0, 0.0, 0.0, 0.0])
        fish3 = generate_wavefish(eodfs[2], samplerate, duration=8.0, noise_std=0.01,
                                  amplitudes=[10.0, 5.0, 1.0, 0.2], phases=[0.0, 0.0, 0.0, 0.0])
        fish4 = generate_wavefish(eodfs[3], samplerate, duration=8.0, noise_std=0.01,
                                  amplitudes=[6.0, 3.0, 1.0, 0.3], phases=[0.0, 0.0, 0.0, 0.0])
        data = fish1 + fish2 + fish3 + fish4
    else:
        from .dataloader import load_data
        print("load %s ..." % sys.argv[1])
        data, samplerate, unit = load_data(sys.argv[1], 0)
        title = sys.argv[1]

    # retrieve fundamentals from power spectrum:
    psd_data = psd(data, samplerate, freq_resolution=0.5)
    def call_harm():
        harmonic_groups(psd_data[0], psd_data[1],max_divisor=4)
    import timeit
    n = 50
    print(timeit.timeit(call_harm, number=n)/n)
    #exit()
    groups, _, mains, all_freqs, good_freqs, _, _, _ = harmonic_groups(psd_data[0], psd_data[1], verbose=0)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plot_psd_harmonic_groups(ax, psd_data[0], psd_data[1], groups, mains, all_freqs, good_freqs,
                             max_freq=3000.0)
    ax.set_title(title)
    plt.show()
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
