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


### RECORDING-waveform-N.EXT

For each fish the average waveform with standard deviation.

| time    | mean     | std     | fit      |
|--------:|---------:|--------:|---------:|
| -18.390 | -0.31223 | 0.00106 | -0.31173 |
| -18.367 | -0.31129 | 0.00094 | -0.31082 |
| -18.345 | -0.31036 | 0.00106 | -0.30994 |
| -18.322 | -0.30924 | 0.00111 | -0.30908 |
| -18.299 | -0.30814 | 0.00103 | -0.30822 |

The columns contain:
1. The time in milliseconds.
2. The averaged waveform in the unit of the input data.
3. The corresponding standard deviation.
4. A fit to the averaged waveform. In case of a wave fish this is
   a Fourier series, for pulse fish it is an exponential fit to the tail of the last peak.


### RECORDING-spectrum-N.EXT

The parameter of the Fourier series fitted to the waveform of a wave-type fish.

| harmonics | frequency | amplitude | relampl  | phase    | power       | relpower  |
|----------:|----------:|----------:|---------:|---------:|------------:|----------:|
|         0 |     76.61 |  0.556175 |  100.000 |   0.0000 |  2.7460e-01 |  100.0000 |
|         1 |    153.23 |  0.239355 |   43.036 |  -1.9342 |  4.5127e-02 |   16.4339 |
|         2 |    229.84 |  0.028035 |    5.041 |  -1.8851 |  7.6312e-04 |    0.2779 |
|         3 |    306.45 |  0.064939 |   11.676 |  -2.8247 |  4.0351e-03 |    1.4694 |
|         4 |    383.07 |  0.013518 |    2.430 |   2.1898 |  1.3847e-04 |    0.0504 |

The columns contain:
1. The index of the harmonics. The first one with index 0 is the fundamental frequency.
2. The frequency of the harmonics in Hertz.
3. The amplitude of each harmonics obtained by fitting a Fourier series to the data in the unit of the input data.
4. The amplitude of each harmonics relative to the amplitude of the fundamental in percent.
5. The phase of each harmonics obtained by fitting a Fourier series to the data in radians ranging from 0 to 2 pi.
6. The power spectral desnity of the harmonics from the original power spectrum of the data.
7. The power spectral desnity of the harmonics relative to the power of the fundamental in percent.


### RECORDING-powerspectrum-N.EXT

The power spectrum of a single EOD pulse of a pulse-type fish:

| frequency | power      |
|----------:|-----------:|
|      0.00 | 4.7648e-10 |
|      0.34 | 9.5307e-10 |
|      0.67 | 9.5336e-10 |
|      1.01 | 9.5386e-10 |
|      1.35 | 9.5455e-10 |
|      1.68 | 9.5544e-10 |

The columns contain:
1. The frequency in Hertz.
2. The power spectral density.


### RECORDING-peaks-N.EXT

Properties of peaks and troughs of a pulse-type fish's EOD.

| P | time  | amplitude | relampl | width |
|--:|------:|----------:|--------:|------:|
| 1 | 0.000 |   0.78412 |   100.0 | 0.333 |
| 2 | 0.385 |  -0.85923 |  -109.6 | 0.248 |

The columns contain:
1. The name of the peak/trough. Peaks and troughs are numbered sequentially. P1 is the 
   largest peak with positive amplitude.
2. The time of the peak/trough relative to P1 in milliseconds.
3. The amplitude of the peak/trough in the unit of the input data.
4. The amplitude of the peak/trough relative to the amplitude of P1.
5. The width of the peak/trough at half height in milliseconds. 


### RECORDING-wavefish-eodfs.EXT

The fundamental EOD frequency and its power for each wavetype fish detected in the recording.

| EODf   | power    |
|-------:|---------:|
|  91.02 |  -22.584 |
| 172.73 |  -46.826 |
| 197.66 |  -23.576 |
| 543.46 |  -37.562 |
| 555.59 |  -20.043 |
| 567.63 |  -47.607 |

The columns contain:
1. The EOD frequency in Hertz.
2. The power of this EOD in decibel.