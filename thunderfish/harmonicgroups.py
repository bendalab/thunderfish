"""Functions for extracting harmonic groups from a power spectrum.

harmonic_groups(): detect peaks in a power spectrum and groups them
                   according to their harmonic structure.

extract_fundamentals(): collect harmonic groups from lists of power spectrum peaks.
threshold_estimate(): estimates thresholds for peak detection in a power spectrum.

fundamental_freqs(): extract the fundamental frequencies from lists of harmonic groups
                     as returned by harmonic_groups().
colors_markers(): Generate a list of colors and markers for plotting.
plot_harmonic_groups(): Mark decibel power of fundamentals and their harmonics.
plot_psd_harmonic_groups(): Plot decibel power-spectrum with detected peaks, harmonic groups, and mains frequencies.
"""

from __future__ import print_function
import numpy as np
from .peakdetection import detect_peaks, peak_size_width, hist_threshold
from .powerspectrum import decibel, power, plot_decibel_psd
try:
    import matplotlib.cm as cm
    import matplotlib.colors as mc
except ImportError:
    pass


def build_harmonic_group(freqs, more_freqs, deltaf, verbose=0, min_freq=20.0, max_freq=2000.0,
                         freq_tol_fac=0.7, max_divisor=4, max_upper_fill=1,
                         max_double_use_harmonics=8, max_double_use_count=1,
                         max_fill_ratio=0.25, power_n_harmonics=10, **kwargs):
    """Find all the harmonics belonging to the largest peak in a list of frequency peaks.

    Parameters
    ----------
    freqs: 2-D array
        List of frequency, height, size, width and count of strong peaks in a power spectrum.
    more_freqs:
        List of frequency, height, size, width and count of all peaks in a power spectrum.
    deltaf: float
        Frequency resolution of the power spectrum.
    verbose: int
        Verbosity level.
    min_freq: float
        Minimum frequency accepted as a fundamental frequency.
    max_freq: float
        Maximum frequency accepted as a fundamental frequency.
    freq_tol_fac: float
        Harmonics need to fall within deltaf*freq_tol_fac.
    max_divisor: int
        Maximum divisor used for checking for sub-harmonics.
    max_upper_fill: int
        Maximum number of frequencies that are allowed to be filled in
        (i.e. they are not contained in more_freqs) above the frequency of the
        largest peak in freqs for constructing a harmonic group.
    max_double_use_harmonics: int
        Maximum harmonics for which double uses of peaks are counted.
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
    
    # start at the strongest frequency:
    fmaxinx = np.argmax(freqs[:, 1])
    fmax = freqs[fmaxinx, 0]
    if verbose > 1:
        print('')
        print(70 * '#')
        print('freqs:     ', '[', ', '.join(['{:.2f}'.format(f) for f in freqs[:, 0]]), ']')
        print('more_freqs:', '[', ', '.join(
            ['{:.2f}'.format(f) for f in more_freqs[:, 0] if f < max_freq]), ']')
        print('## fmax is: {0: .2f}Hz: {1:.5g} ##\n'.format(fmax, np.max(freqs[:, 1])))

    # container for harmonic groups
    best_group = list()
    best_moregroup = list()
    best_group_peaksum = 0.0
    best_group_fill_ins = 1000000000
    best_divisor = 0
    best_fzero = 0.0
    best_fzero_harmonics = 0

    freqtol = freq_tol_fac * deltaf

    # ###########################################
    # SEARCH FOR THE REST OF THE FREQUENCY GROUP
    # start with the strongest fundamental and try to gather the full group of available harmonics
    # In order to find the fundamental frequency of a harmonic group,
    # we divide fmax (the strongest frequency in the spectrum)
    # by a range of integer divisors.
    # We do this, because fmax could just be a strong harmonic of the harmonic group

    for divisor in range(1, max_divisor + 1):

        # define the hypothesized fundamental, which is compared to all higher frequencies:
        fzero = fmax / divisor
        # fzero is not allowed to be smaller than our chosen minimum frequency:
        # if divisor > 1 and fzero < min_freq:   # XXX why not also for divisor=1???
        #    break
        fzero_harmonics = 1

        if verbose > 1:
            print('# divisor:', divisor, 'fzero=', fzero)

        # ###########################################
        # SEARCH ALL DETECTED FREQUENCIES in freqs
        # this in the end only recomputes fzero!
        newgroup = list()
        npre = -1  # previous harmonics
        ndpre = 0.0  # difference of previous frequency
        connected = True
        for j in range(freqs.shape[0]):

            if verbose > 2:
                print('check freq {:3d} {:8.2f} '.format(j, freqs[j, 0]), end='')

            # IS THE CURRENT FREQUENCY AN INTEGRAL MULTIPLE OF FZERO?
            # divide the frequency-to-be-checked by fzero
            # to get the multiplication factor between freq and fzero
            n = np.round(freqs[j, 0] / fzero)
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
                nn = np.round((freqs[j, 0] - freqs[newgroup[-1], 0]) / fzero)
                if nn == 0:
                    # the current frequency is the same harmonic as the previous one
                    # print(divisor, j, freqs[j,0], freqs[newgroup[-1],0])
                    if len(newgroup) > 1:
                        # check whether the current frequency is fzero apart from the previous harmonics:
                        nn = np.round((freqs[j, 0] - freqs[newgroup[-2], 0]) / fzero)
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

        if verbose > 3:
            print('newgroup:', divisor, fzero, newgroup)

        newmoregroup = list()
        fill_ins = 0
        double_use = 0
        ndpre = 0.0  # difference of previous frequency

        # ###########################################
        # SEARCH ALL DETECTED FREQUENCIES in morefreqs
        for j in range(more_freqs.shape[0]):

            if verbose > 3:
                print('check more_freq %3d %8.2f ' % (j, more_freqs[j, 0]), end='')

            # IS FREQUENCY A AN INTEGRAL MULTIPLE OF FREQUENCY B?
            # divide the frequency-to-be-checked with fzero:
            # what is the multiplication factor between freq and fzero?
            n = np.round(more_freqs[j, 0] / fzero)
            if n == 0:
                if verbose > 3:
                    print('discarded: n == 0')
                continue

            # !! the difference between the detection, divided by the derived integer
            # , and fzero should be very very small: 1 resolution step of the fft
            # (more_freqs[j,0] / n) = should be fzero, plus minus a little tolerance,
            # which is the fft resolution
            nd = np.abs((more_freqs[j, 0] / n) - fzero)

            # ... compare it to our tolerance
            if nd > freqtol:
                if verbose > 3:
                    print('discarded: not a harmonic n=%2d d=%5.2fHz tol=%5.2fHz' % (n, nd, freqtol))
                continue

            # two succeeding frequencies should also differ by fzero plus/minus tolerance:
            if len(newmoregroup) > 0:
                nn = np.round((more_freqs[j, 0] - more_freqs[newmoregroup[-1], 0]) / fzero)
                if nn == 0:
                    # the current frequency is close to the same harmonic as the previous one
                    # print(n, newmoregroup[-1], ( more_freqs[j,0] - more_freqs[newmoregroup[-1],0] )/fzero)
                    # print(divisor, j, n, more_freqs[j,0], more_freqs[newmoregroup[-1],0], more_freqs[newmoregroup[-2],0], newmoregroup[-2])
                    if len(newmoregroup) > 1 and newmoregroup[-2] >= 0:
                        # check whether the current frequency is fzero apart from the previous harmonics:
                        nn = np.round((more_freqs[j, 0] - more_freqs[newmoregroup[-2], 0]) / fzero)
                        nnd = np.abs(((more_freqs[j, 0] - more_freqs[newmoregroup[-2], 0]) / nn) - fzero)
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
                    nnd = np.abs(((more_freqs[j, 0] - more_freqs[newmoregroup[-1], 0]) / nn) - fzero)
                    if nnd > 2.0 * freqtol:
                        if verbose > 3:
                            print('discarded: distance to previous harmonics %2d %5.2f %5.2f %8.2f' %
                                  (nn, nnd, freqtol, fzero))
                        continue
            ndpre = nd

            # too many fill-ins upstream of fmax ?
            if more_freqs[j, 0] > fmax and n - 1 - len(newmoregroup) > max_upper_fill:
                # finish this group immediately
                if verbose > 3:
                    print('stopping group: too many upper fill-ins:', n - 1 - len(newmoregroup), '>',
                          max_upper_fill)
                break

            # fill in missing harmonics:
            while len(newmoregroup) < n - 1:  # while some harmonics are missing ...
                newmoregroup.append(-1)  # ... add marker for non-existent harmonic
                fill_ins += 1

            # count double usage of frequency:
            if n <= max_double_use_harmonics:
                double_use += more_freqs[j, 4]
                if verbose > 3 and more_freqs[j, 4] > 0:
                    print('double use of %.2fHz ' % more_freqs[j, 0], end='')

            # take frequency:
            newmoregroup.append(j)
            if verbose > 3:
                print('append')

        # double use of points:
        if double_use > max_double_use_count:
            if verbose > 1:
                print('discarded group because of double use:', double_use)
            continue

        # ratio of total fill-ins too large:
        if len(newmoregroup) == 0 or float(fill_ins) / float(len(newmoregroup)) > max_fill_ratio:
            if verbose > 1:
                if len(newmoregroup) == 0:
                    print('discarded group because newmoregroup is empty! %d from %d' %
                          (fill_ins, len(newmoregroup)))
                else:
                    print('discarded group because of too many fill ins! %d from %d (%g)' %
                          (fill_ins, len(newmoregroup), float(fill_ins) / float(len(newmoregroup))), newmoregroup)
            continue

        # REASSEMBLE NEW GROUP BECAUSE FZERO MIGHT HAVE CHANGED AND
        # CALCULATE THE PEAKSUM, GIVEN THE UPPER LIMIT
        # DERIVED FROM morefreqs which can be low because of too many fill ins.
        # newgroup is needed to delete the right frequencies from freqs later on.
        newgroup = []
        fk = 0
        for j in range(len(newmoregroup)):
            if newmoregroup[j] >= 0:
                # existing frequency peak:
                f = more_freqs[newmoregroup[j], 0]
                # find this frequency in freqs:
                for k in range(fk, freqs.shape[0]):
                    if np.abs(freqs[k, 0] - f) < 1.0e-8:
                        newgroup.append(k)
                        fk = k + 1
                        break
                if fk >= freqs.shape[0]:
                    break

        # fmax might not be in our group, because we adjust fzero:
        if not fmaxinx in newgroup:
            if verbose > 1:
                print("discarded: lost fmax")
            continue

        n = power_n_harmonics
        newmoregroup_peaksum = np.sum(more_freqs[newmoregroup[:n], 1])
        fills = np.sum(np.asarray(newmoregroup[:len(best_moregroup)]) < 0)
        best_fills = np.sum(np.asarray(best_moregroup[:len(newmoregroup)]) < 0)
        takes = np.sum(np.asarray(newmoregroup) >= 0)
        best_takes = np.sum(np.asarray(best_moregroup) >= 0)

        if verbose > 1:
            print('newgroup:      divisor={d}, fzero={fz:.2f}Hz, peaksum={ps}, fills={f}, takes={t}'.format(d=divisor,
                                                                                                            fz=fzero,
                                                                                                            ps=newmoregroup_peaksum,
                                                                                                            f=fills,
                                                                                                            t=takes),
                  newgroup)
            print('newmoregroup:  divisor={d}, fzero={fz:.2f}Hz, peaksum={ps}, fills={f}, takes={t}'.format(d=divisor,
                                                                                                            fz=fzero,
                                                                                                            ps=newmoregroup_peaksum,
                                                                                                            f=fills,
                                                                                                            t=takes),
                  newmoregroup)
            if verbose > 2:
                print('bestgroup:     divisor={d}, fzero={fz:.2f}Hz, peaksum={ps}, fills={f}, takes={t}'.format(
                    d=best_divisor, fz=best_fzero, ps=best_group_peaksum, f=best_fills, t=best_takes), best_group)

        # TAKE THE NEW GROUP IF BETTER:
        # sum of peak power must be larger and
        # less fills. But if the new group has more takes,
        # this might compensate for more fills.
        if newmoregroup_peaksum > best_group_peaksum \
                and fills - best_fills <= 0.5 * (takes - best_takes):

            best_group_peaksum = newmoregroup_peaksum
            if len(newgroup) == 1:
                best_group_fill_ins = np.max((2, fill_ins))  # give larger groups a chance XXX we might reduce this!
            else:
                best_group_fill_ins = fill_ins
            best_group = newgroup
            best_moregroup = newmoregroup
            best_divisor = divisor
            best_fzero = fzero
            best_fzero_harmonics = fzero_harmonics

            if verbose > 2:
                print('new bestgroup:     divisor={d}, fzero={fz:.2f}Hz, peaksum={ps}, fills={f}, takes={t}'.format(
                    d=best_divisor, fz=best_fzero, ps=best_group_peaksum, f=best_fills, t=best_takes), best_group)
                print('new bestmoregroup: divisor={d}, fzero={fz:.2f}Hz, peaksum={ps}, fills={f}, takes={t}'.format(
                    d=best_divisor, fz=best_fzero, ps=best_group_peaksum, f=best_fills, t=best_takes), best_moregroup)
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
        print('## best groups found for fmax={fm:.2f}Hz: fzero={fz:.2f}Hz, d={d:d}:'.format(fm=fmax, fz=best_fzero,
                                                                                            d=best_divisor))
        print('## bestgroup:     ', best_group, '[', ', '.join(['{:.2f}'.format(f) for f in freqs[best_group, 0]]), ']')
        print('## bestmoregroup: ', best_moregroup, '[', ', '.join(
            ['{:.2f}'.format(f) for f in more_freqs[best_moregroup, 0]]), ']')

    # fill up group:
    group = np.zeros((len(best_moregroup), 5))
    for i, inx in enumerate(best_moregroup):
        # increment double use counter:
        more_freqs[inx, 4] += 1.0
        if inx >= 0:
            group[i, :] = more_freqs[inx, :]
        # take adjusted peak frequencies:
        group[i, 0] = (i + 1) * best_fzero

    if verbose > 1:
        refi = np.nonzero(group[:, 1] > 0.0)[0][0]
        print('')
        print('# resulting harmonic group for fmax=', fmax)
        for i in range(group.shape[0]):
            print('{0:8.2f}Hz n={1:5.2f}: p={2:10.3g} p/p0={3:10.3g}'.format(group[i, 0], group[i, 0] / group[0, 0],
                                                                             group[i, 1], group[i, 1] / group[refi, 1]))

    # erase from freqs:
    for inx in reversed(best_group):
        freqs = np.delete(freqs, inx, axis=0)

    # freqs: removed all frequencies of bestgroup
    # more_freqs: updated double use count
    # group: the group
    # fmax: fmax
    return freqs, more_freqs, group, best_fzero_harmonics, fmax


def extract_fundamentals(good_freqs, all_freqs, deltaf, verbose=0, freq_tol_fac=0.7,
                         mains_freq=60.0, min_freq=0.0, max_freq=2000.0,
                         max_divisor=4, max_upper_fill=1,
                         max_double_use_harmonics=8, max_double_use_count=1,
                         max_fill_ratio=0.25, power_n_harmonics=10,
                         min_group_size=3, max_harmonics=0, max_groups=0, **kwargs):
    """Extract fundamental frequencies from power-spectrum peaks.
                         
    Parameters
    ----------
    good_freqs: 2-D array
        List of frequency, height, size, width and count of strong peaks in a power spectrum.
    all_freqs: 2-D array
        List of frequency, height, size, width and count of all peaks in a power spectrum.
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
    max_upper_fill: int
        Maximum number of frequencies that are allowed to be filled in
        (i.e. they are not contained in more_freqs) above the frequency of the
        largest peak in freqs for constructing a harmonic group.
    max_double_use_harmonics: int
        Maximum harmonics for which double uses of peaks are counted.
    max_double_use_count: int
        Maximum number of harmonic groups a single peak can be part of.
    max_fill_ratio: float
        Maximum allowed fraction of filled in frequencies..
    power_n_harmonics: int
        Maximum number of harmonics over which the total power of the signal is computed.
    min_group_size: int
        Minimum required number of harmonics that are not filled in and
        are not part of other, so far detected,  harmonics groups.
    max_harmonics: int
        Maximum number of harmonics to be returned for each group.
    max_groups: int
        If not zero the maximum number of most powerful harmonic groups.

    Returns
    -------
    group_list: list of 2-D arrays
        List of all harmonic groups found sorted by fundamental frequency.
        Each harmonic group is a 2-D array with the first dimension the harmonics
        and the second dimension containing frequency, height, and size of each harmonic.
        If the power is zero, there was no corresponding peak in the power spectrum.
    fzero_harmonics_list: list of int
        The harmonics from which the fundamental frequencies were computed.
    mains_list: 2-d array
        Array of mains peaks found in all_freqs (frequency, height, size).
    """
    if verbose > 0:
        print('')

    # set double use count to zero:
    all_freqs[:, 4] = 0.0

    freqtol = freq_tol_fac * deltaf

    # remove power line harmonics from good_freqs:
    # XXX might be improved!!!
    if mains_freq > 0.0:
        pfreqtol = 1.0  # 1 Hz tolerance
        for inx in reversed(range(len(good_freqs))):
            n = np.round(good_freqs[inx, 0] / mains_freq)
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
            build_harmonic_group(good_freqs, all_freqs, deltaf,
                                verbose, min_freq, max_freq, freq_tol_fac,
                                max_divisor, max_upper_fill,
                                max_double_use_harmonics, max_double_use_count,
                                max_fill_ratio, power_n_harmonics)

        if verbose > 1:
            print('')

        # nothing found:
        if harm_group.shape[0] == 0:
            if verbose > 0:
                print('Nothing found for fmax=%.2fHz' % fmax)
            continue

        # count number of harmonics which have been detected, are not fill-ins,
        # and are not doubly used:
        group_size = np.sum((harm_group[:, 1] > 0.0) & (harm_group[:, 4] < 2.0))
        group_size_ok = (group_size >= min_group_size)

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
                    group_size, fundamental_ok, mains_ok))

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
            n = np.round(all_freqs[inx, 0] / mains_freq)
            nd = np.abs(all_freqs[inx, 0] - n * mains_freq)
            if nd <= pfreqtol:
                mains_list.append(all_freqs[inx, 0:2])
                
    return group_list, fzero_harmonics_list, np.array(mains_list)


def threshold_estimate(psd_data, low_thresh_factor=6.0, high_thresh_factor=10.0,
                       nbins=100, hist_height=1.0/ np.sqrt(np.e)):
    """Estimate thresholds for peak detection from histogram of power spectrum.

    The standard deviation of the noise floor without peaks is estimated from
    the width of the histogram of the power spectrum at `hist_height` relative height.
    The histogtram is computed in the third quarter of the power spectrum.

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
    noise_std, center = hist_threshold(psd_data_seg, th_factor=1.0, nbins=nbins)
    low_threshold = noise_std * low_thresh_factor
    high_threshold = noise_std * high_thresh_factor
    return low_threshold, high_threshold, center


def harmonic_groups(psd_freqs, psd, verbose=0, low_threshold=0.0, high_threshold=0.0,
                    thresh_bins=100, low_thresh_factor=6.0, high_thresh_factor=10.0,
                    max_peak_width_fac=10.0, min_peak_width=1.0,
                    freq_tol_fac=0.7, mains_freq=60.0, min_freq=0.0, max_freq=2000.0,
                    max_work_freq=4000.0, max_divisor=4, max_upper_fill=1,
                    max_double_use_harmonics=8, max_double_use_count=1,
                    max_fill_ratio=0.25, power_n_harmonics=10,
                    min_group_size=3, max_harmonics=0, max_groups=0, **kwargs):
    """Detect peaks in power spectrum and extract fundamentals of harmonic groups.

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
    max_work_freq: float
        Maximum frequency to be used for strong ("good") peaks.
    max_divisor: int
        Maximum divisor used for checking for sub-harmonics.
    max_upper_fill: int
        Maximum number of frequencies that are allowed to be filled in
        (i.e. they are not contained in more_freqs) above the frequency of the
        largest peak in freqs for constructing a harmonic group.
    max_double_use_harmonics: int
        Maximum harmonics for which double uses of peaks are counted.
    max_double_use_count: int
        Maximum number of harmonic groups a single peak can be part of.
    max_fill_ratio: float
        Maximum allowed fraction of filled in frequencies.
    power_n_harmonics: int
        Maximum number of harmonics over which the total power of the signal is computed.
    min_group_size: int
        Minimum required number of harmonics that are not filled in and
        are not part of other, so far detected,  harmonics groups.
    max_harmonics: int
        Maximum number of harmonics to be returned for each group.
    max_groups: int
        If not zero the maximum number of most powerful harmonic groups.

    Returns
    -------
    group_list: list of 2-D arrays
        List of all extracted harmonic groups, sorted by fundamental frequency.
        Each harmonic group is a 2-D array with the first dimension the harmonics
        and the second dimension containing frequency, height, and size of each harmonic.
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
        low_threshold, high_threshold, center = threshold_estimate(log_psd, low_thresh_factor,
                                                                   high_thresh_factor,
                                                                   thresh_bins)
        
        if verbose > 1:
            print('')
            print('low_threshold=', low_threshold, center + low_threshold)
            print('high_threshold=', high_threshold, center + high_threshold)
            print('center=', center)

    # detect peaks in decibel power spectrum:
    peaks, troughs = detect_peaks(log_psd, low_threshold)
    all_freqs = peak_size_width(psd_freqs, log_psd, peaks, troughs)

    if len(all_freqs) == 0:
        return [], [], [], np.zeros((0, 5)), [], low_threshold, high_threshold, center

    # maximum width of a frequency peak:
    wthresh = max_peak_width_fac * delta_f
    if wthresh < min_peak_width:
        wthresh = min_peak_width
        
    # select good peaks:
    freqs = all_freqs[(all_freqs[:, 2] > high_threshold) &
                      (all_freqs[:, 0] >= min_freq) &
                      (all_freqs[:, 0] <= max_work_freq) &
                      (all_freqs[:, 3] < wthresh), :]

    # convert peak sizes back to power:
    freqs[:, 1] = power(freqs[:, 1])
    all_freqs[:, 1] = power(all_freqs[:, 1])

    # detect harmonic groups:
    groups, fzero_harmonics, mains = extract_fundamentals(freqs, all_freqs, delta_f,
                                                          verbose, freq_tol_fac,
                                                          mains_freq, min_freq, max_freq,
                                                          max_divisor, max_upper_fill,
                                                          max_double_use_harmonics,
                                                          max_double_use_count, max_fill_ratio,
                                                          power_n_harmonics, min_group_size,
                                                          max_harmonics, max_groups)

    return groups, fzero_harmonics, mains, all_freqs, freqs[:, 0], low_threshold, high_threshold, center


def fundamental_freqs(group_list, return_power=False):
    """
    Extract the fundamental frequencies from lists of harmonic groups.

    Parameters
    ----------
    group_list: list of 2-D arrays or list of list of 2-D arrays
        Lists of harmonic groups as returned by extract_fundamentals() and
        harmonic_groups() with the element [0][0] of the
        harmonic groups being the fundamental frequency.

    Returns
    -------
    fundamentals: 1-D array or list of 1-D array
        Single array or list of arrays (corresponding to the input group_list)
        of the fundamental frequencies.
    """
    if len(group_list) == 0:
        fundamentals = np.array([])
        fund_power = np.array([])
    elif hasattr(group_list[0][0][0], '__len__'):
        fundamentals = []
        fund_power = []
        for groups in group_list:
            fundamentals.append(np.array([harmonic_group[0][0] for harmonic_group in groups]))
            fund_power.append(np.array([harmonic_group[0][1] for harmonic_group in groups]))
    else:
        fundamentals = np.array([harmonic_group[0][0] for harmonic_group in group_list])
        fund_power = np.array([harmonic_group[0][1] for harmonic_group in group_list])

    if return_power:
        return fundamentals, fund_power
    else:
        return fundamentals


def fundamental_freqs_and_db(group_list):

    """
    Extract the fundamental frequencies and their power in dB from lists of harmonic groups.

    Parameters
    ----------
    group_list: list of 2-D arrays or list of list of 2-D arrays
            Lists of harmonic groups as returned by extract_fundamentals() and
            harmonic_groups() with the element [0][0] of the harmonic groups
            being the fundamental frequency,
            and element[0][1] being the corresponding power.

    Returns
    -------
    eodf_db_matrix: 2-D array or list of 2-D arrays
        Matrix with fundamental frequencies in first column and
        corresponding power in dB in second column.
    """

    if len(group_list) == 0:
        eodf_db_matrix = np.array([])
    elif hasattr(group_list[0][0][0], '__len__'):
        eodf_db_matrix = []
        for groups in group_list:
            f = [np.array([harmonic_group[0][0], harmonic_group[0][1]]) for harmonic_group in group_list]
            f[:, 1] = decibel(f[:, 1])  # calculate decibel using 1 as reference power
            eodf_db_matrix.append(f)
    else:
        eodf_db_matrix = np.array([np.array([harmonic_group[0][0], harmonic_group[0][1]])
                                   for harmonic_group in group_list])
        eodf_db_matrix[:, 1] = decibel(eodf_db_matrix[:, 1])  # calculate decibel using 1 as reference power

    return eodf_db_matrix


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


def plot_harmonic_groups(ax, group_list, max_groups=0, sort_by_freq=True,
                         colors=None, markers=None, legend_rows=8, **kwargs):
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
    max_groups: int
            If not zero plot only the max_groups most powerful groups.
    sort_by_freq: boolean
            If True sort legend by frequency, otherwise by power.
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
    powers = np.array([np.sum(fish[:10, 1]) for fish in group_list])
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
        msize = 7.0 + 10.0 * (powers[i] / max_power) ** 0.25
        color_kwargs = {}
        if colors is not None:
            color_kwargs = {'color': colors[k%len(colors)]}
        if markers is None:
            ax.plot(x, decibel(y), 'o', ms=msize, label='%.1f Hz' % group[0, 0], **color_kwargs)
        else:
            if k >= len(markers):
                break
            ax.plot(x, decibel(y), linestyle='None', marker=markers[k], mec=None, mew=0.0,
                    ms=msize, label='%.1f Hz' % group[0, 0], **color_kwargs)

    # legend:
    if legend_rows > 0:
        ncol = (len(idx)-1) // legend_rows + 1
        ax.legend(numpoints=1, ncol=ncol, **kwargs)
    else:
        ax.legend(numpoints=1, **kwargs)


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
        fpeakinx = [np.round(fp/(psd_freqs[1]-psd_freqs[0])) for fp in fpeaks if fp < psd_freqs[-1]]
        ax.plot(fpeaks[:len(fpeakinx)], decibel(psd[fpeakinx]), linestyle='None',
                marker='.', color='k', ms=10, mec=None, mew=0.0,
                label='%3.0f Hz mains' % mains[0, 0])
    # mark harmonic groups:
    colors, markers = colors_markers()
    plot_harmonic_groups(ax, group_list, max_groups=0, sort_by_freq=True,
                         colors=colors, markers=markers, legend_rows=8,
                         loc='upper right')
    # plot power spectrum:
    plot_decibel_psd(ax, psd_freqs, psd, max_freq=max_freq, color='blue')

    
def add_psd_peak_detection_config(cfg, low_threshold=0.0, high_threshold=0.0,
                                  thresh_bins=100,
                                  low_thresh_factor=6.0, high_thresh_factor=10.0,
                                  max_peak_width_fac=10.0, min_peak_width=1.0):
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


def add_harmonic_groups_config(cfg, mains_freq=60.0, max_divisor=4, freq_tol_fac=0.7,
                               max_upper_fill=1, max_fill_ratio=0.25,
                               max_double_use_harmonics=8, max_double_use_count=1,
                               power_n_harmonics=10, min_group_size=3,
                               min_freq=20.0, max_freq=2000.0, max_work_freq=4000.0,
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
    cfg.add('maxUpperFill', max_upper_fill, '',
            'As soon as more than this number of harmonics need to be filled in conescutively stop searching for higher harmonics.')
    cfg.add('maxFillRatio', max_fill_ratio, '',
            'Maximum fraction of filled in harmonics allowed (usefull values are smaller than 0.5)')
    cfg.add('maxDoubleUseHarmonics', max_double_use_harmonics, '', 'Maximum harmonics up to which double uses are penalized.')
    cfg.add('maxDoubleUseCount', max_double_use_count, '', 'Maximum overall double use count allowed.')
    cfg.add('powerNHarmonics', power_n_harmonics, '', 'Compute total power over the first # harmonics.')
    
    cfg.add_section('Acceptance of best harmonic groups:')
    cfg.add('minimumGroupSize', min_group_size, '',
'Minimum required number of harmonics (inclusively fundamental) that are not filled in and are not used by other groups.')
    # cfg['minimumGroupSize'][0] = 2 # panama
    cfg.add('minimumFrequency', min_freq, 'Hz', 'Minimum frequency allowed for the fundamental.')
    cfg.add('maximumFrequency', max_freq, 'Hz', 'Maximum frequency allowed for the fundamental.')
    cfg.add('maximumWorkingFrequency', max_work_freq, 'Hz',
            'Maximum frequency to be used to search for harmonic groups and to adjust fundamental frequency.')
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
                    'max_upper_fill': 'maxUpperFill',
                    'max_fill_ratio': 'maxFillRatio',
                    'max_double_use_harmonics': 'maxDoubleUseHarmonics',
                    'max_double_use_count': 'maxDoubleUseCount',
                    'power_n_harmonics': 'powerNHarmonics',
                    'min_group_size': 'minimumGroupSize',
                    'min_freq': 'minimumFrequency',
                    'max_freq': 'maximumFrequency',
                    'max_work_freq': 'maximumWorkingFrequency',
                    'max_harmonics': 'maxHarmonics',
                    'max_groups': 'maxGroups'})


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from .fakefish import generate_wavefish
    from .powerspectrum import psd

    print("Checking harmonicgroups module ...")
    
    # generate data:
    samplerate = 44100.0
    eodfs = [123.0, 321.0, 666.0, 668.0]
    fish1 = generate_wavefish(eodfs[0], samplerate, duration=8.0, noise_std=0.01,
                              amplitudes=[1.0, 0.5, 0.2, 0.1, 0.05], phases=[0.0, 0.0, 0.0, 0.0, 0.0])
    fish2 = generate_wavefish(eodfs[1], samplerate, duration=8.0, noise_std=0.01,
                              amplitudes=[1.0, 0.7, 0.2, 0.1], phases=[0.0, 0.0, 0.0, 0.0])
    fish3 = generate_wavefish(eodfs[2], samplerate, duration=8.0, noise_std=0.01,
                              amplitudes=[10.0, 5.0, 1.0], phases=[0.0, 0.0, 0.0])
    fish4 = generate_wavefish(eodfs[3], samplerate, duration=8.0, noise_std=0.01,
                              amplitudes=[6.0, 3.0, 1.0], phases=[0.0, 0.0, 0.0])
    data = fish1 + fish2 + fish3 + fish4

    # analyse:
    psd_data = psd(data, samplerate, fresolution=0.5)
    groups, _, mains, all_freqs, good_freqs, _, _, _ = harmonic_groups(psd_data[1], psd_data[0])
    fundamentals = fundamental_freqs(groups)
    print(fundamentals)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plot_psd_harmonic_groups(ax, psd_data[1], psd_data[0], groups, mains, all_freqs, good_freqs,
                             max_freq=3000.0)
    plt.show()
    
