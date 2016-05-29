"""Functions for extracting harmonic groups from a power spectrum.
"""

from __future__ import print_function
import numpy as np
import peakdetection as pd


def build_harmonic_groups(freqs, more_freqs, deltaf, cfg):
    verbose = cfg['verboseLevel'][0]

    # start at the strongest frequency:
    fmaxinx = np.argmax(freqs[:, 1])
    fmax = freqs[fmaxinx, 0]
    if verbose > 1:
        print('')
        print(70 * '#')
        print('freqs:     ', '[', ', '.join(['{:.2f}'.format(f) for f in freqs[:, 0]]), ']')
        print('more_freqs:', '[', ', '.join(
            ['{:.2f}'.format(f) for f in more_freqs[:, 0] if f < cfg['maximumFrequency'][0]]), ']')
        print('## fmax is: {0: .2f}Hz: {1:.5g} ##\n'.format(fmax, np.max(freqs[:, 1])))

    # container for harmonic groups
    best_group = list()
    best_moregroup = list()
    best_group_peaksum = 0.0
    best_group_fill_ins = 1000000000
    best_divisor = 0
    best_fzero = 0.0
    best_fzero_harmonics = 0

    freqtol = cfg['freqTolerance'][0] * deltaf

    # ###########################################
    # SEARCH FOR THE REST OF THE FREQUENCY GROUP
    # start with the strongest fundamental and try to gather the full group of available harmonics
    # In order to find the fundamental frequency of a fish harmonic group,
    # we divide fmax (the strongest frequency in the spectrum)
    # by a range of integer divisors.
    # We do this, because fmax could just be a strong harmonic of the harmonic group

    for divisor in xrange(1, cfg['maxDivisor'][0] + 1):

        # define the hypothesized fundamental, which is compared to all higher frequencies:
        fzero = fmax / divisor
        # fzero is not allowed to be smaller than our chosen minimum frequency:
        # if divisor > 1 and fzero < cfg['minimumFrequency'][0]:   # XXX why not also for divisor=1???
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
        for j in xrange(freqs.shape[0]):

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
        for j in xrange(more_freqs.shape[0]):

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
            if more_freqs[j, 0] > fmax and n - 1 - len(newmoregroup) > cfg['maxUpperFill'][0]:
                # finish this group immediately
                if verbose > 3:
                    print('stopping group: too many upper fill-ins:', n - 1 - len(newmoregroup), '>', cfg['maxUpperFill'][0])
                break

            # fill in missing harmonics:
            while len(newmoregroup) < n - 1:  # while some harmonics are missing ...
                newmoregroup.append(-1)  # ... add marker for non-existent harmonic
                fill_ins += 1

            # count double usage of frequency:
            if n <= cfg['maxDoubleUseHarmonics'][0]:
                double_use += more_freqs[j, 4]
                if verbose > 3 and more_freqs[j, 4] > 0:
                    print('double use of %.2fHz ' % more_freqs[j, 0], end='')

            # take frequency:
            newmoregroup.append(j)
            if verbose > 3:
                print('append')

        # double use of points:
        if double_use > cfg['maxDoubleUseCount'][0]:
            if verbose > 1:
                print('discarded group because of double use:', double_use)
            continue

        # ratio of total fill-ins too large:
        if float(fill_ins) / float(len(newmoregroup)) > cfg['maxFillRatio'][0]:
            if verbose > 1:
                print('dicarded group because of too many fill ins! %d from %d (%g)' %
                      (fill_ins, len(newmoregroup), float(fill_ins) / float(len(newmoregroup))), newmoregroup)
            continue

        # REASSEMBLE NEW GROUP BECAUSE FZERO MIGHT HAVE CHANGED AND
        # CALCULATE THE PEAKSUM, GIVEN THE UPPER LIMIT
        # DERIVED FROM morefreqs which can be low because of too many fill ins.
        # newgroup is needed to delete the right frequencies from freqs later on.
        newgroup = []
        fk = 0
        for j in xrange(len(newmoregroup)):
            if newmoregroup[j] >= 0:
                # existing frequency peak:
                f = more_freqs[newmoregroup[j], 0]
                # find this frequency in freqs:
                for k in xrange(fk, freqs.shape[0]):
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

        n = cfg['powerNHarmonics'][0]
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
                                                                                                            t=takes), newgroup)
            print('newmoregroup:  divisor={d}, fzero={fz:.2f}Hz, peaksum={ps}, fills={f}, takes={t}'.format(d=divisor,
                                                                                                            fz=fzero,
                                                                                                            ps=newmoregroup_peaksum,
                                                                                                            f=fills,
                                                                                                            t=takes), newmoregroup)
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
        for i in xrange(group.shape[0]):
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


def extract_fundamentals(good_freqs, all_freqs, deltaf, cfg):
    """
    Extract fundamental frequencies from power-spectrum peaks.

    Returns:
        group_list (list): list of all harmonic groups found
        fzero_harmonics_list (list): the harmonics from which the fundamental frequencies were computed
        mains_list (2-d array): list of mains peaks found
    """

    verbose = cfg['verboseLevel'][0]
    if verbose > 0:
        print('')

    # set double use count to zero:
    all_freqs[:, 4] = 0.0

    freqtol = cfg['freqTolerance'][0] * deltaf
    mainsfreq = cfg['mainsFreq'][0]

    # remove power line harmonics from good_freqs:
    # XXX might be improved!!!
    if mainsfreq > 0.0:
        pfreqtol = 1.0  # 1 Hz tolerance
        for inx in reversed(xrange(len(good_freqs))):
            n = np.round(good_freqs[inx, 0] / mainsfreq)
            nd = np.abs(good_freqs[inx, 0] - n * mainsfreq)
            if nd <= pfreqtol:
                if verbose > 1:
                    print('remove power line frequency', inx, good_freqs[inx, 0], np.abs(
                        good_freqs[inx, 0] - n * mainsfreq))
                good_freqs = np.delete(good_freqs, inx, axis=0)

    group_list = list()
    fzero_harmonics_list = list()
    # as long as there are frequencies left in good_freqs:
    while good_freqs.shape[0] > 0:
        # we check for harmonic groups:
        good_freqs, all_freqs, harm_group, fzero_harmonics, fmax = \
            build_harmonic_groups(good_freqs, all_freqs, deltaf, cfg)

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
        group_size_ok = (group_size >= cfg['minimumGroupSize'][0])

        # check frequency range of fundamental:
        fundamental_ok = (harm_group[0, 0] >= cfg['minimumFrequency'][0] and
                          harm_group[0, 0] <= cfg['maximumFrequency'][0])

        # check power hum (does this really ever happen???):
        mains_ok = ((mainsfreq == 0.0) |
                    (np.abs(harm_group[0, 0] - mainsfreq) > freqtol))

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
    maxharmonics = cfg['maxHarmonics'][0]
    if maxharmonics > 0:
        for group in group_list:
            if group.shape[0] > maxharmonics:
                if verbose > 0:
                    print('Discarding some tailing harmonics')
                group = group[:maxharmonics, :]

    if verbose > 0:
        print('')
        if len(group_list) > 0:
            print('## FUNDAMENTALS FOUND: ##')
            for i in xrange(len(group_list)):
                power = group_list[i][:, 1]
                print('{:8.2f}Hz: {:10.8f} {:3d} {:3d}'.format(group_list[i][0, 0], np.sum(power),
                                                               np.sum(power <= 0.0), fzero_harmonics_list[i]))
        else:
            print('## NO FUNDAMENTALS FOUND ##')

    # assemble mains frequencies from all_freqs:
    mains_list = []
    if mainsfreq > 0.0:
        pfreqtol = 1.0
        for inx in xrange(len(all_freqs)):
            n = np.round(all_freqs[inx, 0] / mainsfreq)
            nd = np.abs(all_freqs[inx, 0] - n * mainsfreq)
            if nd <= pfreqtol:
                mains_list.append(all_freqs[inx])
    return group_list, fzero_harmonics_list, np.array(mains_list)


def threshold_estimate(data, noise_factor, peak_factor):
    """
    Estimate noise standard deviation from histogram
    for usefull peak-detection thresholds.

    The standard deviation of the noise floor without peaks is estimated from
    the histogram of the data at 1/sqrt(e) relative height.

    Args:
        data: the data from which to estimate the thresholds
        noise_factor (float): multiplies the estimate of the standard deviation
                              of the noise to result in the low_threshold
        peak_factor (float): the high_threshold is the low_threshold plus
                             this fraction times the distance between largest peaks
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
    lower = bins[inx][0]
    upper = bins[inx][-1]  # needs to return the next bin
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
    upperbins = bins[cumhist > upperpthresh]
    if len(upperbins) > 0:
        upperth = upperbins[0]
    else:
        upperth = bins[-1]
    highthreshold = lowthreshold + peak_factor * noisestd
    if upperth > lowerth + 0.1 * noisestd:
        highthreshold = lowerth + peak_factor * (upperth - lowerth) + 0.5 * lowthreshold - center

    return lowthreshold, highthreshold, center


def harmonic_groups(psd_freqs, psd, cfg):
    """
    Detect peaks in power spectrum and extract fundamentals of harmonic groups.

    Args:
        psd_freqs (array): frequencies of the power spectrum
        psd (array): power spectrum
        cfg (dict): configuration parameter

    Returns:
        groups (list): all harmonic groups, sorted by fundamental frequency.
                       Each harmonic group contains a 2-d array with frequencies
                       and power of the fundamental and all harmonics.
                       If the power is zero, there was no corresponding peak
                       in the power spectrum.
        fzero_harmonics (list) : The harmonics from
                       which the fundamental frequencies were computed.
        mains (2-d array): frequencies and power of multiples of mains frequency.
        all_freqs (2-d array): peaks in the power spectrum
                  detected with low threshold
                  [frequency, power, size, width, double use count]
        good_freqs (array): frequencies of peaks detected with high threshold
        low_threshold (float): the relative threshold for detecting all peaks in the decibel spectrum
        high_threshold (float): the relative threshold for detecting good peaks in the decibel spectrum
        center (float): the baseline level of the power spectrum
    """

    verbose = cfg['verboseLevel'][0]

    if verbose > 0:
        print('')
        print(70 * '#')
        print('##### harmonic_groups', 48 * '#')

    # decibel power spectrum:
    log_psd = 10.0 * np.log10(psd)

    # thresholds:
    low_threshold = cfg['lowThreshold'][0]
    high_threshold = cfg['highThreshold'][0]
    center = np.NaN
    if cfg['lowThreshold'][0] <= 0.0 or cfg['highThreshold'][0] <= 0.0:
        n = len(log_psd)
        low_threshold, high_threshold, center = threshold_estimate(log_psd[2 * n / 3:n * 9 / 10],
                                                                   cfg['noiseFactor'][0],
                                                                   cfg['peakFactor'][0])
        if verbose > 1:
            print('')
            print('low_threshold=', low_threshold, center + low_threshold)
            print('high_threshold=', high_threshold, center + high_threshold)
            print('center=', center)

    # detect peaks in decibel power spectrum:
    all_freqs, _ = pd.detect_peaks(log_psd, low_threshold, psd_freqs,
                                   pd.accept_peaks_size_width)

    if len(all_freqs) == 0:
        # TODO: Why has not been a peak detected?
        return [], [], [], np.zeros((0,5)), [], low_threshold, high_threshold, center

    # select good peaks:
    wthresh = cfg['maxPeakWidthFac'][0] * (psd_freqs[1] - psd_freqs[0])
    if wthresh < cfg['minPeakWidth'][0]:
        wthresh = cfg['minPeakWidth'][0]
    freqs = all_freqs[(all_freqs[:, 2] > high_threshold) &
                      (all_freqs[:, 0] >= cfg['minimumFrequency'][0]) &
                      (all_freqs[:, 0] <= cfg['maximumWorkingFrequency'][0]) &
                      (all_freqs[:, 3] < wthresh), :]

    # convert peak sizes back to power:
    freqs[:, 1] = 10.0 ** (0.1 * freqs[:, 1])
    all_freqs[:, 1] = 10.0 ** (0.1 * all_freqs[:, 1])

    # detect harmonic groups:
    groups, fzero_harmonics, mains = extract_fundamentals(freqs, all_freqs, psd_freqs[1] - psd_freqs[0], cfg)

    return groups, fzero_harmonics, mains, all_freqs, freqs[:, 0], low_threshold, high_threshold, center
