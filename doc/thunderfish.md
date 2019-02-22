# thunderfish

Automatically detect and analyze all fish present in an EOD recording
and generate a summary plot and data tables.


## Command line arguments

```
thunderfish --help
```
returns
```
usage: thunderfish [-h] [--version] [-v] [-c [CFGFILE]] [-p] [-s] [-f FORMAT]
                   [-o OUTPATH] [-b]
                   [file [file ...]] [channel]

Analyze EOD waveforms of weakly electric fish.

positional arguments:
  file                  name of the file with the time series data
  channel               channel to be analyzed

optional arguments:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
  -v                    verbosity level
  -c [CFGFILE], --save-config [CFGFILE]
                        save configuration to file CFGFILE after reading all
                        configuration files (defaults to thunderfish.cfg)
  -p                    save output plot as pdf file
  -s                    save analysis results to files
  -f FORMAT             file format used for saving analysis results, one of
                        dat, ascii, csv, md, tex, html (defaults to the format
                        specified in the configuration file or "dat")
  -o OUTPATH            path where to store results and figures
  -b                    show the cost function of the best window algorithm

by Benda-Lab (2015-2019)
```

## Summary plot

In the plot you can press
- `q`: Close the plot and show the next one or quit.
- `p`: Play the analyzed section of the reording on the default audio device.


## Output files

With the `-s` switch analyzis results are saved to files.

Output files are placed in the current working directory if no path is
specified via the `-o` switch. If the path specified via `-o` does not
exist it is created.

The following sections list and describe the files that are generated.
The filenames are composed of the basename of the input file (RECORDING).
The fish detected in the recordings are numbered, starting with 0 (N).
The file extension depends on the chosen file format (EXT).


### RECORDING-eodwaveform-N.EXT

For each fish the average waveform with standard deviation.

| time/ms | mean/a.u. | std/a.u. | fit/a.u. |
|--------:|----------:|---------:|---------:|
|  -1.701 |  -0.27500 |  0.01269 | -0.27253 |
|  -1.678 |  -0.23612 |  0.01258 | -0.23345 |
|  -1.655 |  -0.19632 |  0.01294 | -0.19375 |
|  -1.633 |  -0.15507 |  0.01369 | -0.15251 |
|  -1.610 |  -0.10935 |  0.01536 | -0.10720 |
|  -1.587 |  -0.05924 |  0.01713 | -0.05684 |
|  -1.565 |  -0.00366 |  0.01885 | -0.00124 |

The columns contain:
1. The time in milliseconds.
2. The averaged waveform in the unit of the input data.
3. The corresponding standard deviation.
4. A fit to the averaged waveform. In case of a wave fish this is
   a Fourier series, for pulse fish it is an exponential fit to the tail of the last peak.


### RECORDING-wavespectrum-N.EXT

The parameter of the Fourier series fitted to the waveform of a wave-type fish.

| harmonics | frequency/Hz | amplitude/a.u. | relampl/% | phase/rad | power/a.u.^2/Hz | relpower/% |
|----------:|-------------:|---------------:|----------:|----------:|----------------:|-----------:|
|         0 |       728.09 |       0.326238 |   100.000 |    0.0000 |      1.0302e-01 |   100.0000 |
|         1 |      1456.18 |       0.221786 |    67.983 |   -0.7173 |      4.4380e-02 |    43.0785 |
|         2 |      2184.27 |       0.032462 |     9.950 |   -1.9980 |      8.5350e-04 |     0.8285 |
|         3 |      2912.37 |       0.037329 |    11.442 |    2.3485 |      9.9695e-04 |     0.9677 |
|         4 |      3640.46 |       0.020644 |     6.328 |    2.9600 |      2.6696e-04 |     0.2591 |

The columns contain:
1. The index of the harmonics. The first one with index 0 is the fundamental frequency.
2. The frequency of the harmonics in Hertz.
3. The amplitude of each harmonics obtained by fitting a Fourier series to the data in the unit of the input data.
4. The amplitude of each harmonics relative to the amplitude of the fundamental in percent.
5. The phase of each harmonics obtained by fitting a Fourier series to the data in radians ranging from 0 to 2 pi.
6. The power spectral desnity of the harmonics from the original power spectrum of the data.
7. The power spectral desnity of the harmonics relative to the power of the fundamental in percent.


### RECORDING-wavefish.EXT

Fundamental EOD frequency and and other properties of each
wave-type fish detected in the recording.

| EODf/Hz | power/dB | p-p-amplitude/a.u. | noise/%  | rmserror/% | n    |
|--------:|---------:|-------------------:|---------:|-----------:|-----:|
|   91.03 |  -22.615 |              0.305 |    28.37 |       0.18 | 4445 |
|  142.33 |  -60.274 |              0.213 |    46.44 |       0.07 |  729 |
|  172.69 |  -48.334 |              0.105 |   110.09 |       0.13 | 1582 |
|  197.66 |  -26.918 |              0.038 |   315.74 |       0.08 | 5334 |
|  261.47 |  -62.351 |              0.052 |   231.60 |       0.07 | 4667 |
|  348.63 |  -63.843 |              0.040 |   299.59 |       0.14 | 4348 |
|  543.46 |  -37.559 |              0.020 |   612.68 |       0.39 | 4749 |
|  555.59 |  -20.378 |              0.021 |   587.69 |       0.75 | 5210 |
|  567.63 |  -49.081 |              0.021 |   582.43 |       0.15 | 5901 |

The columns contain:
1. The EOD frequency in Hertz.
2. The power of this EOD in decibel.
3. The peak-to-peak amplitude in the units of the input data.
4. The root-mean-variance of the averaged EOD waveform relative to the
   peak-to_peak amplitude in percent.
5. The root-mean-squared difference betwenn the averaged EOD waveform and 
   the fit of the Fourier series relative to the peak-to_peak amplitude in percent.
6. The number of EODs used for computing the averaged EOD waveform.


### RECORDING-pulsespectrum-N.EXT

The power spectrum of a single EOD pulse of a pulse-type fish:

| frequency/Hz | power/a.u.^2/Hz |
|-------------:|----------------:|
|         0.00 |      4.5771e-10 |
|         0.34 |      9.1552e-10 |
|         0.67 |      9.1582e-10 |
|         1.01 |      9.1632e-10 |
|         1.35 |      9.1703e-10 |
|         1.68 |      9.1793e-10 |

The columns contain:
1. The frequency in Hertz.
2. The power spectral density.


### RECORDING-pulsepeaks-N.EXT

Properties of peaks and troughs of a pulse-type fish's EOD.

| P | time/ms | amplitude/a.u. | relampl/% | width/ms |
|--:|--------:|---------------:|----------:|---------:|
| 1 |   0.000 |        0.78342 |     100.0 |    0.334 |
| 2 |   0.385 |       -0.85865 |    -109.6 |    0.248 |

The columns contain:
1. The name of the peak/trough. Peaks and troughs are numbered sequentially. P1 is the 
   largest peak with positive amplitude.
2. The time of the peak/trough relative to P1 in milliseconds.
3. The amplitude of the peak/trough in the unit of the input data.
4. The amplitude of the peak/trough relative to the amplitude of P1.
5. The width of the peak/trough at half height in milliseconds. 


### RECORDING-pulsefish.EXT

Properties of each pulse-type fish detected in the recording.

| EODf/Hz | period/ms | max-ampl/a.u. | min-ampl/a.u. | p-p-amplitude/a.u. | tstart/ms | tend/ms | width/ms | tau1/ms | n  | peakpower/dB | peakfreq/Hz | poweratt5/dB | poweratt50/dB | lowcutoff/Hz | median/ms | mean/ms | std/ms |
|--------:|----------:|--------------:|--------------:|-------------------:|----------:|--------:|---------:|--------:|---:|-------------:|------------:|-------------:|--------------:|-------------:|----------:|--------:|-------:|
|   30.71 |     32.57 |         0.783 |         0.859 |              1.642 |    -0.499 |   1.134 |    1.633 |   0.121 | 10 |       -66.30 |      899.35 |       -24.20 |        -21.54 |       125.50 |     32.56 |   32.57 |   0.08 |

The columns contain:
1. The EOD frequency in Hertz.
2. The period between two pulses (1/EODf) in milliseconds.
3. The amplitude of the largest peak (P1 peak) in the units of the input data.
4. The amplitude of the larges trough in the units of the input data.
5. The peak-to-peak amplitude in the units of the input data.
6. The time where the pulse starts relative to P1 in milliseconds.
7. The time where the pulse ends relative to P1 in milliseconds.
8. The total width of the pulse on milliseconds.
9. The time constant of the exponential decay of the tail of the pulse in milliseconds.
10. The number of EODs used for computing the averaged EOD waveform.
11. The peak power of the single pulse spectrum in decibel.
12. The frequency at the peak power of the single pulse spectrum in Hertz.
13. How much the average power below 5 Hz is attenuated relative to the peak power in decibel.
14. How much the average power below 50 Hz is attenuated relative to the peak power in decibel.
15. Frequency at which the power reached half of the peak power relative to the initial power in Hertz.
16. The median inter-pulse interval after removal of outlier intervals in milliseconds.
17. The mean inter-pulse interval after removal of outlier intervals in milliseconds.
18. The standard deviation of inter-pulse intervals after removal of outlier intervals in milliseconds.
