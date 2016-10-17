"""Functions for extracting harmonic groups from a power spectrum.

harmonic_groups(): detects peaks in a power spectrum and groups them
                   according to their harmonic structure.

extract_fundamental_freqs(): extracts a simple list of fundamental frequencies from
                             the detailed list returned by harmonic_groups().

extract_fundamentals():
threshold_estimate(): estimates thresholds for peak detection in a power spectrum.
"""

from __future__ import print_function
import numpy as np
from .peakdetection import detect_peaks, accept_peaks_size_width


def build_harmonic_group(freqs, more_freqs, deltaf, verbose=0, min_freq=20.0, max_freq=2000.0,
                         freq_tol_fac=0.7, max_divisor=4, max_upper_fill=1,
                         max_double_use_harmonics=8, max_double_use_count=1,
                         max_fill_ratio=0.25, power_n_harmonics=10, **kwargs):
    """Find all the harmonics belonging to the largest peak in a list of frequency peaks.

    Args:
        freqs (2-D numpy array): list of frequency, height, size, width and count of
                                 strong peaks in a power spectrum.
        more_freqs (): list of frequency, height, size, width and count of
                       all peaks in a power spectrum.
        deltaf (float): frequency resolution of the power spectrum
        verbose (int): verbosity level
        min_freq (float): minimum frequency accepted as a fundamental frequency
        max_freq (float): maximum frequency accepted as a fundamental frequency
        freq_tol_fac (float): harmonics need to fall within deltaf*freq_tol_fac
        max_divisor (int): maximum divisor used for checking for sub-harmonics
        max_upper_fill (int): maximum number of frequencies that are allowed to be filled in
            (i.e. they are not contained in more_freqs) above the frequency of the
            largest peak in freqs for constructing a harmonic group.
        max_double_use_harmonics (int): maximum harmonics for which double uses of peaks
                                        are counted.
        max_double_use_count (int): maximum number of harmonic groups a single peak can be part of.
        max_fill_ratio (float): maximum allowed fraction of filled in frequencies.
        power_n_harmonics (int): maximum number of harmonics over which the total power
                                 of the signal is computed.
    
    Returns:
        freqs (2-D numpy array): list of strong frequencies with the frequencies of group removed
        more_freqs (2-D numpy array): list of all frequencies with updated double-use counts
        group (2-D numpy array): the detected harmonic group. Might be empty.
        best_fzero_harmonics (int): the highest harmonics that was used to recompute
                                    the fundamental frequency
        fmax (float): the frequency of the largest peak in freqs for which the harmonic group was detected.
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
        if float(fill_ins) / float(len(newmoregroup)) > max_fill_ratio:
            if verbose > 1:
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
                         min_group_size=3, max_harmonics=0, **kwargs):
    """Extract fundamental frequencies from power-spectrum peaks.
                         
    Args:
        good_freqs (2-D numpy array): list of frequency, height, size, width and count of
                                 strong peaks in a power spectrum.
        all_freqs (): list of frequency, height, size, width and count of
                       all peaks in a power spectrum.
        deltaf (float): frequency resolution of the power spectrum
        verbose (int): verbosity level
        freq_tol_fac (float): harmonics need to fall within deltaf*freq_tol_fac
        mains_freq (float): frequency of the mains power supply.
        min_freq (float): minimum frequency accepted as a fundamental frequency
        max_freq (float): maximum frequency accepted as a fundamental frequency
        max_divisor (int): maximum divisor used for checking for sub-harmonics
        max_upper_fill (int): maximum number of frequencies that are allowed to be filled in
            (i.e. they are not contained in more_freqs) above the frequency of the
            largest peak in freqs for constructing a harmonic group.
        max_double_use_harmonics (int): maximum harmonics for which double uses of peaks
                                        are counted.
        max_double_use_count (int): maximum number of harmonic groups a single peak can be part of.
        max_fill_ratio (float): maximum allowed fraction of filled in frequencies.
        power_n_harmonics (int): maximum number of harmonics over which the total power
                                 of the signal is computed.
        min_group_size (int): minimum required number of harmonics that are not filled in and
                              are not part of other, so far detected,  harmonics groups.
        max_harmonics (int): maximum number of harmonics to be returned for each group.

    Returns:
        group_list (list of 2-D numpy arrays): list of all harmonic groups found sorted
            by fundamental frequency.
            Each harmonic group is a 2-D numpy array with the first dimension the harmonics
            and the second dimension containing frequency, height, and size of each harmonic.
            If the power is zero, there was no corresponding peak in the power spectrum.
        fzero_harmonics_list (list of int): the harmonics from which the fundamental frequencies were computed.
        mains_list (2-d array): list of mains peaks found in all_freqs (frequency, height, size)
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

    # sort groups by fundamental frequency:
    ffreqs = [f[0, 0] for f in group_list]
    finx = np.argsort(ffreqs)
    group_list = [group_list[fi] for fi in finx]
    fzero_harmonics_list = [fzero_harmonics_list[fi] for fi in finx]

    # do not save more than n harmonics:
    if max_harmonics > 0:
        for group in group_list:
            if group.shape[0] > max_harmonics:
                if verbose > 1:
                    print('Discarding some tailing harmonics for f=%.2fHz' % group[0, 0])
                group = group[:max_harmonics, :]

    if verbose > 0:
        print('')
        if len(group_list) > 0:
            print('## FUNDAMENTALS FOUND: ##')
            for i in range(len(group_list)):
                power = group_list[i][:, 1]
                print('{:8.2f}Hz: {:10.8f} {:3d} {:3d}'.format(group_list[i][0, 0], np.sum(power),
                                                               np.sum(power <= 0.0), fzero_harmonics_list[i]))
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


def threshold_estimate(data, noise_factor, peak_factor):
    """Estimate noise standard deviation from histogram
    for usefull peak-detection thresholds.

    The standard deviation of the noise floor without peaks is estimated from
    the width of the histogram of the data at 1/sqrt(e) relative height.

    Args:
        data: the data from which to estimate the thresholds
        noise_factor (float): multiplies the estimate of the standard deviation
                              of the noise to result in the low_threshold
        peak_factor (float): the high_threshold is the low_threshold plus
                             this fraction times the distance between largest pe aks
                             and low_threshold plus half the low_threshold

    Returns:
        low_threshold (float): the threshold just above the noise floor
        high_threshold (float): the threshold for clear peaks
        center: (float): estimate of the median of the data without peaks
    """

    # estimate noise standard deviation:
    # XXX what about the number of bins for small data sets?
    hist, bins = np.histogram(data, 100, density=True)
    inx = hist > np.max(hist) / np.sqrt(np.e)
    lower = bins[0:-1][inx][0]
    upper = bins[1:][inx][-1]  # needs to return the next bin
    center = 0.5 * (lower + upper)
    noisestd = 0.5 * (upper - lower)

    # low threshold:
    lowthreshold = noise_factor * noisestd

    # high threshold:
    lowerth = center + 0.5 * lowthreshold
    cumhist = np.cumsum(hist) / np.sum(hist)
    upperpthresh = 0.95
    if bins[-2] >= lowerth:
        pthresh = cumhist[bins[:-1] >= lowerth][0]
        upperpthresh = pthresh + 0.95 * (1.0 - pthresh)
    upperbins = bins[:-1][cumhist > upperpthresh]
    if len(upperbins) > 0:
        upperth = upperbins[0]
    else:
        upperth = bins[-1]
    highthreshold = lowthreshold + peak_factor * noisestd
    if upperth > lowerth + 0.1 * noisestd:
        highthreshold = lowerth + peak_factor * (upperth - lowerth) + 0.5 * lowthreshold - center

    return lowthreshold, highthreshold, center


def harmonic_groups(psd_freqs, psd, verbose=0, low_threshold=0.0, high_threshold=0.0,
                    noise_fac=6.0, peak_fac=0.5, max_peak_width_fac=3.5, min_peak_width=1.0,
                    freq_tol_fac=0.7, mains_freq=60.0, min_freq=0.0, max_freq=2000.0,
                    max_work_freq=4000.0, max_divisor=4, max_upper_fill=1,
                    max_double_use_harmonics=8, max_double_use_count=1,
                    max_fill_ratio=0.25, power_n_harmonics=10,
                    min_group_size=3, max_harmonics=0, **kwargs):
    """Detect peaks in power spectrum and extract fundamentals of harmonic groups.

    Args:
        psd_freqs (array): frequencies of the power spectrum
        psd (array): power spectrum (linear, not decible)
        verbose (int): verbosity level
        low_threshold (float): the relative threshold for detecting all peaks
                               in the decibel spectrum.
        high_threshold (float): the relative threshold for detecting good peaks
                                in the decibel spectrum
                                
        noise_factor (float): multiplies the estimate of the standard deviation
                              of the noise to result in the low_threshold
        peak_factor (float): the high_threshold is the low_threshold plus
                             this fraction times the distance between largest peaks
                             and low_threshold plus half the low_threshold
        max_peak_width_fac (float): the maximum allowed width of a good peak
                                    in the decibel power spectrum in multiples of
                                    the frequency resolution.
        min_peak_width (float): the minimum absolute value for the maximum width
                                of a peak in Hertz.
        freq_tol_fac (float): harmonics need to fall within deltaf*freq_tol_fac
        mains_freq (float): frequency of the mains power supply.
        min_freq (float): minimum frequency accepted as a fundamental frequency
        max_freq (float): maximum frequency accepted as a fundamental frequency
        max_work_freq (float): maximum frequency to be used for strong ("good") peaks
        max_divisor (int): maximum divisor used for checking for sub-harmonics
        max_upper_fill (int): maximum number of frequencies that are allowed to be filled in
            (i.e. they are not contained in more_freqs) above the frequency of the
            largest peak in freqs for constructing a harmonic group.
        max_double_use_harmonics (int): maximum harmonics for which double uses of peaks
                                        are counted.
        max_double_use_count (int): maximum number of harmonic groups a single peak can be part of.
        max_fill_ratio (float): maximum allowed fraction of filled in frequencies.
        power_n_harmonics (int): maximum number of harmonics over which the total power
                                 of the signal is computed.
        min_group_size (int): minimum required number of harmonics that are not filled in and
                              are not part of other, so far detected,  harmonics groups.
        max_harmonics (int): maximum number of harmonics to be returned for each group.

    Returns:
        group_list (list of 2-D numpy arrays): list of all extracted harmonic groups, sorted
            by fundamental frequency.
            Each harmonic group is a 2-D numpy array with the first dimension the harmonics
            and the second dimension containing frequency, height, and size of each harmonic.
            If the power is zero, there was no corresponding peak in the power spectrum.
        fzero_harmonics (list of ints) : The harmonics from
            which the fundamental frequencies were computed.
        mains (2-d array): frequencies and power of multiples of the mains frequency found in the power spectrum.
        all_freqs (2-d array): peaks in the power spectrum detected with low threshold
                  [frequency, power, size, width, double use count].
        good_freqs (1-d array): frequencies of peaks detected with high threshold.
        low_threshold (float): the relative threshold for detecting all peaks in the decibel spectrum.
        high_threshold (float): the relative threshold for detecting good peaks in the decibel spectrum.
        center (float): the baseline level of the power spectrum.
    """
    if verbose > 0:
        print('')
        print(70 * '#')
        print('##### harmonic_groups', 48 * '#')

    # decibel power spectrum:
    log_psd = 10.0 * np.log10(psd)

    # thresholds:
    center = np.NaN
    if low_threshold <= 0.0 or high_threshold <= 0.0:
        n = len(log_psd)
        low_threshold, high_threshold, center = threshold_estimate(log_psd[2 * n // 3:n * 9 // 10],
                                                                   noise_fac, peak_fac)
        if verbose > 1:
            print('')
            print('low_threshold=', low_threshold, center + low_threshold)
            print('high_threshold=', high_threshold, center + high_threshold)
            print('center=', center)

    # detect peaks in decibel power spectrum:
    all_freqs, _ = detect_peaks(log_psd, low_threshold, psd_freqs,
                                accept_peaks_size_width)

    if len(all_freqs) == 0:
        # TODO: Why has not been a peak detected?
        return [], [], [], np.zeros((0, 5)), [], low_threshold, high_threshold, center

    # select good peaks:
    wthresh = max_peak_width_fac * (psd_freqs[1] - psd_freqs[0])
    if wthresh < min_peak_width:
        wthresh = min_peak_width
    freqs = all_freqs[(all_freqs[:, 2] > high_threshold) &
                      (all_freqs[:, 0] >= min_freq) &
                      (all_freqs[:, 0] <= max_work_freq) &
                      (all_freqs[:, 3] < wthresh), :]

    # convert peak sizes back to power:
    freqs[:, 1] = 10.0 ** (0.1 * freqs[:, 1])
    all_freqs[:, 1] = 10.0 ** (0.1 * all_freqs[:, 1])

    # detect harmonic groups:
    groups, fzero_harmonics, mains = extract_fundamentals(freqs, all_freqs,
                                                          psd_freqs[1] - psd_freqs[0],
                                                          verbose, freq_tol_fac,
                                                          mains_freq, min_freq, max_freq,
                                                          max_divisor, max_upper_fill,
                                                          max_double_use_harmonics,
                                                          max_double_use_count,max_fill_ratio,
                                                          power_n_harmonics, min_group_size,
                                                          max_harmonics)

    return groups, fzero_harmonics, mains, all_freqs, freqs[:, 0], low_threshold, high_threshold, center


def extract_fundamental_freqs(fishlists):
    """
    Extracts the fundamental frequencies of multiple or single fishlists created by the harmonicgroups modul.

    This function gets a 4-D array as input. This input consists of multiple fishlists from the harmonicgroups modul
    lists up (fishlists[list][fish][harmonic][frequency, power]). The amount of lists doesn't matter. With a for-loop
    this function collects all fundamental frequencies of every fishlist. In the end the output is a 2-D array
    containing the fundamentals of each list (fundamentals[list][fundamental_frequencies]).

    :param fishlists:       (4-D array or 3-D array) List of or single fishlists with harmonics and each frequency and power.
                            fishlists[fishlist][fish][harmonic][frequency, power]
                            fishlists[fish][harmonic][frequency, power]
    :return fundamentals:   (1-D array or 2-D array) list of or single arrays containing the fundamentals of a fishlist.
                            fundamentals = [ [f1, f1, ..., f1, f1], [f2, f2, ..., f2, f2], ..., [fn, fn, ..., fn, fn] ]
                            fundamentals = [f1, f1, ..., f1, f1]
    """
    if hasattr(fishlists[0][0][0], '__len__'):
        fundamentals = []
        for fishlist in range(len(fishlists)):
            fundamentals.append(np.array([fish[0][0] for fish in fishlists[fishlist]]))
    else:
        fundamentals = np.array([fish[0][0] for fish in fishlists])
    return fundamentals


def add_psd_peak_detection_config(cfg, low_threshold=0.0, high_threshold=0.0,
                                  noise_fac=6.0, peak_fac=0.5,
                                  max_peak_width_fac=3.5, min_peak_width=1.0):
    """ Add parameter needed for detection of peaks in power spectrum used by
    harmonic_groups() as a new section to a configuration.

    Args:
      cfg (ConfigFile): the configuration
    """

    cfg.add_section('Thresholds for peak detection in power spectra:')
    cfg.add('lowThreshold', low_threshold, 'dB', 'Threshold for all peaks.\n If 0.0 estimate threshold from histogram.')
    cfg.add('highThreshold', high_threshold, 'dB', 'Threshold for good peaks. If 0.0 estimate threshold from histogram.')
    # cfg['lowThreshold'][0] = 12. # panama
    # cfg['highThreshold'][0] = 18. # panama
    
    cfg.add_section('Threshold estimation:\nIf no thresholds are specified they are estimated from the histogram of the decibel power spectrum.')
    cfg.add('noiseFactor', noise_fac, '', 'Factor for multiplying std of noise floor for lower threshold.')
    cfg.add('peakFactor', peak_fac, '', 'Fractional position of upper threshold above lower threshold.')

    cfg.add_section('Peak detection in decibel power spectrum:')
    cfg.add('maxPeakWidthFac', max_peak_width_fac, '',
            'Maximum width of peaks at 0.75 hight in multiples of frequency resolution (might be increased)')
    cfg.add('minPeakWidth', min_peak_width, 'Hz', 'Peaks do not need to be narrower than this.')


def psd_peak_detection_args(cfg):
    """ Translates a configuration to the respective parameter names for the
    detection of peaks in power spectrum used by harmonic_groups().
    The return value can then be passed as key-word arguments to this function.

    Args:
      cfg (ConfigFile): the configuration

    Returns:
      a (dict): dictionary with names of arguments of the harmonic-group()
      function and their values as supplied by cfg.
    """

    return cfg.map({'low_threshold': 'lowThreshold',
                    'high_threshold': 'highThreshold',
                    'noise_fac': 'noiseFactor',
                    'peak_fac': 'peakFactor',
                    'max_peak_width_fac': 'maxPeakWidthFac',
                    'min_peak_width': 'minPeakWidth'})


def add_harmonic_groups_config(cfg, mains_freq=60.0, max_divisor=4, freq_tol_fac=0.7,
                               max_upper_fill=1, max_fill_ratio=0.25,
                               max_double_use_harmonics=8, max_double_use_count=1,
                               power_n_harmonics=10, min_group_size=3,
                               min_freq=20.0, max_freq=2000.0, max_work_freq=4000.0,
                               max_harmonics=0):
    """ Add parameter needed for detection of harmonic groups as
    a new section to a configuration.

    Args:
      cfg (ConfigFile): the configuration
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


def harmonic_groups_args(cfg):
    """ Translates a configuration to the
    respective parameter names of the harmonic-group detection functions.
    The return value can then be passed as key-word arguments to this function.

    Args:
      cfg (ConfigFile): the configuration

    Returns:
      a (dict): dictionary with names of arguments of the harmonic-group detection
      functions and their values as supplied by cfg.
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
                    'max_harmonics': 'maxHarmonics'})

