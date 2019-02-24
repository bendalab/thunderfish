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

With the `-s` switch analysis results are saved to files.

Output files are placed in the current working directory if no path is
specified via the `-o` switch. If the path specified via `-o` does not
exist it is created.

The following sections list and describe the files that are generated.
The filenames are composed of the basename of the input file (RECORDING).
The fish detected in the recordings are numbered, starting with 0 (N).
The file extension depends on the chosen file format (EXT).


### RECORDING-eodwaveform-N.EXT

For each fish the average waveform with standard deviation.

<table>
<thead>
  <tr>
    <th align="left">time</th>
    <th align="left">mean</th>
    <th align="left">std</th>
    <th align="left">fit</th>
  </tr>
  <tr>
    <th align="left">ms</th>
    <th align="left">a.u.</th>
    <th align="left">a.u.</th>
    <th align="left">a.u.</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="right">-1.746</td>
    <td align="right">-0.34837</td>
    <td align="right">0.01194</td>
    <td align="right">-0.34562</td>
  </tr>
  <tr>
    <td align="right">-1.723</td>
    <td align="right">-0.30700</td>
    <td align="right">0.01199</td>
    <td align="right">-0.30411</td>
  </tr>
  <tr>
    <td align="right">-1.701</td>
    <td align="right">-0.26664</td>
    <td align="right">0.01146</td>
    <td align="right">-0.26383</td>
  </tr>
  <tr>
    <td align="right">-1.678</td>
    <td align="right">-0.22713</td>
    <td align="right">0.01153</td>
    <td align="right">-0.22426</td>
  </tr>
  <tr>
    <td align="right">-1.655</td>
    <td align="right">-0.18706</td>
    <td align="right">0.01187</td>
    <td align="right">-0.18428</td>
  </tr>
</tbody>
</table>

The columns contain:
1. Time in milliseconds.
2. Averaged waveform in the unit of the input data.
3. Corresponding standard deviation.
4. A fit to the averaged waveform. In case of a wave fish this is
   a Fourier series, for pulse fish it is an exponential fit to the tail of the last peak.


### RECORDING-wavespectrum-N.EXT

The parameter of the Fourier series fitted to the waveform of a wave-type fish.

<table>
<thead>
  <tr>
    <th align="left">harmonics</th>
    <th align="left">frequency</th>
    <th align="left">amplitude</th>
    <th align="left">relampl</th>
    <th align="left">phase</th>
    <th align="left">power</th>
    <th align="left">relpower</th>
  </tr>
  <tr>
    <th align="left">-</th>
    <th align="left">Hz</th>
    <th align="left">a.u.</th>
    <th align="left">%</th>
    <th align="left">rad</th>
    <th align="left">a.u.^2/Hz</th>
    <th align="left">%</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="right">0</td>
    <td align="right">728.09</td>
    <td align="right">0.32634</td>
    <td align="right">100.00</td>
    <td align="right">0.0000</td>
    <td align="right">1.0530e-01</td>
    <td align="right">100.00</td>
  </tr>
  <tr>
    <td align="right">1</td>
    <td align="right">1456.18</td>
    <td align="right">0.22179</td>
    <td align="right">67.96</td>
    <td align="right">-0.7157</td>
    <td align="right">4.8310e-02</td>
    <td align="right">45.88</td>
  </tr>
  <tr>
    <td align="right">2</td>
    <td align="right">2184.27</td>
    <td align="right">0.03244</td>
    <td align="right">9.94</td>
    <td align="right">-1.9988</td>
    <td align="right">1.0239e-03</td>
    <td align="right">0.97</td>
  </tr>
  <tr>
    <td align="right">3</td>
    <td align="right">2912.37</td>
    <td align="right">0.03715</td>
    <td align="right">11.38</td>
    <td align="right">2.3472</td>
    <td align="right">1.3243e-03</td>
    <td align="right">1.26</td>
  </tr>
  <tr>
    <td align="right">4</td>
    <td align="right">3640.46</td>
    <td align="right">0.02056</td>
    <td align="right">6.30</td>
    <td align="right">2.9606</td>
    <td align="right">3.9391e-04</td>
    <td align="right">0.37</td>
  </tr>
</tbody>
</table>

The columns contain:
1. Index of the harmonics. The first one with index 0 is the fundamental frequency.
2. Frequency of the harmonics in Hertz.
3. Amplitude of each harmonics obtained by fitting a Fourier series to the data in the unit of the input data.
4. Amplitude of each harmonics relative to the amplitude of the fundamental in percent.
5. Phase of each harmonics obtained by fitting a Fourier series to the data in radians ranging from 0 to 2 pi.
6. Power spectral density of the harmonics from the original power spectrum of the data.
7. Power spectral density of the harmonics relative to the power of the fundamental in percent.


### RECORDING-wavefish.EXT

Fundamental EOD frequency and and other properties of each
wave-type fish detected in the recording.

<table>
<thead>
  <tr>
    <th align="left">index</th>
    <th align="left">EODf</th>
    <th align="left">power</th>
    <th align="left">p-p-amplitude</th>
    <th align="left">noise</th>
    <th align="left">rmserror</th>
    <th align="left">n</th>
  </tr>
  <tr>
    <th align="left">-</th>
    <th align="left">Hz</th>
    <th align="left">dB</th>
    <th align="left">a.u.</th>
    <th align="left">%</th>
    <th align="left">%</th>
    <th align="left">-</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="right">0</td>
    <td align="right">555.59</td>
    <td align="right">-20.04</td>
    <td align="right">0.242</td>
    <td align="right">42.9</td>
    <td align="right">0.07</td>
    <td align="right">4445</td>
  </tr>
  <tr>
    <td align="right">1</td>
    <td align="right">91.02</td>
    <td align="right">-22.58</td>
    <td align="right">0.213</td>
    <td align="right">50.3</td>
    <td align="right">0.10</td>
    <td align="right">729</td>
  </tr>
  <tr>
    <td align="right">2</td>
    <td align="right">197.66</td>
    <td align="right">-23.58</td>
    <td align="right">0.131</td>
    <td align="right">91.0</td>
    <td align="right">0.14</td>
    <td align="right">1582</td>
  </tr>
  <tr>
    <td align="right">3</td>
    <td align="right">666.75</td>
    <td align="right">-31.42</td>
    <td align="right">0.039</td>
    <td align="right">322.7</td>
    <td align="right">0.05</td>
    <td align="right">5334</td>
  </tr>
  <tr>
    <td align="right">4</td>
    <td align="right">583.25</td>
    <td align="right">-32.32</td>
    <td align="right">0.049</td>
    <td align="right">258.9</td>
    <td align="right">0.19</td>
    <td align="right">4667</td>
  </tr>
</tbody>
</table>

The columns contain:
1. Index of the fish (the number that is also used to number the files).
2. EOD frequency in Hertz.
3. Power of this EOD in decibel.
4. Peak-to-peak amplitude in the units of the input data.
5. Root-mean-variance of the averaged EOD waveform relative to the
   peak-to_peak amplitude in percent.
6. Root-mean-squared difference betwenn the averaged EOD waveform and 
   the fit of the Fourier series relative to the peak-to_peak amplitude in percent.
7. Number of EODs used for computing the averaged EOD waveform.


### RECORDING-pulsespectrum-N.EXT

The power spectrum of a single EOD pulse of a pulse-type fish:

<table>
<thead>
  <tr>
    <th align="left">frequency</th>
    <th align="left">power</th>
  </tr>
  <tr>
    <th align="left">Hz</th>
    <th align="left">a.u.^2/Hz</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="right">0.00</td>
    <td align="right">4.7637e-10</td>
  </tr>
  <tr>
    <td align="right">0.34</td>
    <td align="right">9.5284e-10</td>
  </tr>
  <tr>
    <td align="right">0.67</td>
    <td align="right">9.5314e-10</td>
  </tr>
  <tr>
    <td align="right">1.01</td>
    <td align="right">9.5363e-10</td>
  </tr>
  <tr>
    <td align="right">1.35</td>
    <td align="right">9.5432e-10</td>
  </tr>
  <tr>
    <td align="right">1.68</td>
    <td align="right">9.5522e-10</td>
  </tr>
</tbody>
</table>

The columns contain:
1. Frequency in Hertz.
2. Power spectral density.


### RECORDING-pulsepeaks-N.EXT

Properties of peaks and troughs of a pulse-type fish's EOD.

<table>
<thead>
  <tr>
    <th align="left">P</th>
    <th align="left">time</th>
    <th align="left">amplitude</th>
    <th align="left">relampl</th>
    <th align="left">width</th>
  </tr>
  <tr>
    <th align="left">-</th>
    <th align="left">ms</th>
    <th align="left">a.u.</th>
    <th align="left">%</th>
    <th align="left">ms</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="right">1</td>
    <td align="right">0.000</td>
    <td align="right">0.78409</td>
    <td align="right">100.00</td>
    <td align="right">0.333</td>
  </tr>
  <tr>
    <td align="right">2</td>
    <td align="right">0.385</td>
    <td align="right">-0.85939</td>
    <td align="right">-109.60</td>
    <td align="right">0.248</td>
  </tr>
</tbody>
</table>

The columns contain:
1. Name of the peak/trough. Peaks and troughs are numbered sequentially. P1 is the 
   largest peak with positive amplitude.
2. Time of the peak/trough relative to P1 in milliseconds.
3. Amplitude of the peak/trough in the unit of the input data.
4. Amplitude of the peak/trough relative to the amplitude of P1.
5. Width of the peak/trough at half height in milliseconds. 


### RECORDING-pulsefish.EXT

Properties of each pulse-type fish detected in the recording.

<table>
<thead>
  <tr>
    <th align="left" colspan="11">waveform</th>
    <th align="left" colspan="5">power spectrum</th>
    <th align="left" colspan="4">inter-pulse interval statistics</th>
  </tr>
  <tr>
    <th align="left">index</th>
    <th align="left">EODf</th>
    <th align="left">period</th>
    <th align="left">max-ampl</th>
    <th align="left">min-ampl</th>
    <th align="left">p-p-amplitude</th>
    <th align="left">tstart</th>
    <th align="left">tend</th>
    <th align="left">width</th>
    <th align="left">tau1</th>
    <th align="left">n</th>
    <th align="left">peakpower</th>
    <th align="left">peakfreq</th>
    <th align="left">poweratt5</th>
    <th align="left">poweratt50</th>
    <th align="left">lowcutoff</th>
    <th align="left">medianipi</th>
    <th align="left">meanipi</th>
    <th align="left">stdipi</th>
    <th align="left">nipi</th>
  </tr>
  <tr>
    <th align="left">-</th>
    <th align="left">Hz</th>
    <th align="left">ms</th>
    <th align="left">a.u.</th>
    <th align="left">a.u.</th>
    <th align="left">a.u.</th>
    <th align="left">ms</th>
    <th align="left">ms</th>
    <th align="left">ms</th>
    <th align="left">ms</th>
    <th align="left">-</th>
    <th align="left">dB</th>
    <th align="left">Hz</th>
    <th align="left">dB</th>
    <th align="left">dB</th>
    <th align="left">Hz</th>
    <th align="left">ms</th>
    <th align="left">ms</th>
    <th align="left">ms</th>
    <th align="left">-</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="right">0</td>
    <td align="right">30.68</td>
    <td align="right">32.60</td>
    <td align="right">0.784</td>
    <td align="right">0.859</td>
    <td align="right">1.643</td>
    <td align="right">-0.272</td>
    <td align="right">0.839</td>
    <td align="right">1.111</td>
    <td align="right">0.123</td>
    <td align="right">100</td>
    <td align="right">-66.30</td>
    <td align="right">896.32</td>
    <td align="right">-24.02</td>
    <td align="right">-21.46</td>
    <td align="right">127.18</td>
    <td align="right">32.59</td>
    <td align="right">32.60</td>
    <td align="right">0.07</td>
    <td align="right">117</td>
  </tr>
</tbody>
</table>

The columns contain:
1. Index of the fish (the number that is also used to number the files).
2. EOD frequency in Hertz.
3. Period between two pulses (1/EODf) in milliseconds.
4. Amplitude of the largest peak (P1 peak) in the units of the input data.
5. Amplitude of the larges trough in the units of the input data.
6. Peak-to-peak amplitude in the units of the input data.
7. Time where the pulse starts relative to P1 in milliseconds.
8. Time where the pulse ends relative to P1 in milliseconds.
9. Total width of the pulse on milliseconds.
10. Time constant of the exponential decay of the tail of the pulse in milliseconds.
11. Number of EODs used for computing the averaged EOD waveform.
12. Peak power of the single pulse spectrum in decibel.
13. Frequency at the peak power of the single pulse spectrum in Hertz.
14. How much the average power below 5 Hz is attenuated relative to the peak power in decibel.
15. How much the average power below 50 Hz is attenuated relative to the peak power in decibel.
16. Frequency at which the power reached half of the peak power relative to the initial power in Hertz.
17. Median inter-pulse interval after removal of outlier intervals in milliseconds.
18. Mean inter-pulse interval after removal of outlier intervals in milliseconds.
19. Standard deviation of inter-pulse intervals after removal of outlier intervals in milliseconds.
20. Number of inter-pulse intervals after removal of outlier intervals.
