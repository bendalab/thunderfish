# thunderfish

Automatically detect and analyze all fish present in an EOD recording.

## Authors

The [Neuroethology-lab](https://uni-tuebingen.de/en/faculties/faculty-of-science/departments/biology/institutes/neurobiology/lehrbereiche/neuroethology/) at the Institute of Neuroscience at the University of T&uuml;bingen:
1. Jan Benda
2. J&ouml;rg Henninger
3. Juan Sehuanes
4. Till Raab


## Command line arguments

```
thunderfish --help
```
returns
```
usage: thunderfish [-h] [--version] [-v] [-c] [--channel CHANNEL] [-j [JOBS]]
                   [-s] [-f {dat,ascii,csv,rtai,md,tex,html}] [-p] [-k]
                   [-o OUTPATH] [-b]
                   [file [file ...]]

Analyze EOD waveforms of weakly electric fish.

positional arguments:
  file                  name of the file with the time series data

optional arguments:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
  -v                    verbosity level. Increase by specifying -v multiple
                        times, or like -vvv.
  -c                    save configuration to file thunderfish.cfg after
                        reading all configuration files
  --channel CHANNEL     channel to be analyzed. Default is to use first
                        channel.
  -j [JOBS]             number of jobs run in parallel. Without argument use
                        all CPU cores.
  -s                    save analysis results to files
  -f {dat,ascii,csv,rtai,md,tex,html}
                        file format used for saving analysis results, defaults
                        to the format specified in the configuration file or
                        "dat")
  -p                    save output plot as pdf file
  -k                    keep path of input file when saving analysis files
  -o OUTPATH            path where to store results and figures
  -b                    show the cost function of the best window algorithm

version 1.7 by Benda-Lab (2015-2019)

examples:
- analyze a single file interactively:
  > thunderfish data.wav
- analyze many files automatically and save analysis results and plot to files:
  > thunderfish -s -p *.wav
- analyze all wav files in the river1/ directory, use all CPUs, and write files directly to "results/":
  > thunderfish -j -s -p -o results/ *.wav
- analyze all wav files in the river1/ directory and write files to "results/river1/":
  > thunderfish -s -p -o results/ -k river1/*.wav
- write configuration file:
  > thunderfish -c
benda@knifefish ~/data/fishgrid/Luis2018 $ thunderfish -h
usage: thunderfish [-h] [--version] [-v] [-c] [--channel CHANNEL] [-j [JOBS]]
                   [-s] [-f {dat,ascii,csv,rtai,md,tex,html}] [-p] [-k]
                   [-o OUTPATH] [-b]
                   [file [file ...]]

Analyze EOD waveforms of weakly electric fish.

positional arguments:
  file                  name of the file with the time series data

optional arguments:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
  -v                    verbosity level. Increase by specifying -v multiple
                        times, or like -vvv.
  -c                    save configuration to file thunderfish.cfg after
                        reading all configuration files
  --channel CHANNEL     channel to be analyzed. Default is to use first
                        channel.
  -j [JOBS]             number of jobs run in parallel. Without argument use
                        all CPU cores.
  -s                    save analysis results to files
  -f {dat,ascii,csv,rtai,md,tex,html}
                        file format used for saving analysis results, defaults
                        to the format specified in the configuration file or
                        "dat")
  -p                    save output plot as pdf file
  -k                    keep path of input file when saving analysis files
  -o OUTPATH            path where to store results and figures
  -b                    show the cost function of the best window algorithm

version 1.7 by Benda-Lab (2015-2019)

examples:
- analyze the single file data.wav interactively:
  > thunderfish data.wav
- automatically analyze all wav files in the current working directory and save analysis results and plot to files:
  > thunderfish -s -p *.wav
- analyze all wav files in the river1/ directory, use all CPUs, and write files directly to "results/":
  > thunderfish -j -s -p -o results/ *.wav
- analyze all wav files in the river1/ directory and write files to "results/river1/":
  > thunderfish -s -p -o results/ -k river1/*.wav
- write configuration file:
  > thunderfish -c
```


## Configuration file

Many parameters of the algorithms used by thunderfish can be set via a
configuration file.

Generate the configuration file by executing
```
thunderfish -c
```
This first reads in all configuration files found (see below) and then writes
the file `thunderfish.cfg` into the current working directory.

Whenever you run thunderfish it searches for configuration files in 
1. the current working directory 
2. the directory of each input file
3. the parent directories of each input file, up to three levels up.

Best practice is to move the configuration file at the root of the file tree
where data files of a recording session are stored.

Use the `-vv` switch to see which configuration files are loaded:
```
thunderfish -vv data.wav
```

Open the configuration file in your favourite editor and edit the settings.
Each parameter is briefly explained in the comment preceding the parameter.


### Important configuration parameter

The list of configuration parameter is overwhelming and most of them you 
do not need to touch at all. Here is a list of the few that matter:

- `frequencyResolution`: this sets the nnft parameter for computing
  the power spectrum such to achieve the requested resolution in
  frequency. The longer your analysis window the smaller you can set the
  resultion (not smaller then the inverse analysis window).

- `numberPSDWindows`: If larger than one then only fish that are
  present in all windows are reported. If you have very stationary data 
  (from a restrained fish, not from a fishfinder) set this to one.

- `lowThresholdFactor`, `highThresholdFactor`: play around with these
  numbers if not all wavefish are detected.

- `maxPeakWidthFac`: increase this number (to 50, 100, ...) if not all
  wavefish are detected, and before you start fiddling around with the
  threshold factors.

- `mainsFreq`: Set it to the frequency of your mains power supply
  or to zero if you have hum-free recordings.

- `maxGroups`: Set to 1 if you know that only a single fish is in your
  recording.

- `bestWindowSize`: How much of the data should be used for analysis.
  If you have stationary data (from a restrained fish, not from a
  fishfinder) you may want to use the full recording by setting this
  to zero. Otherwise thunderfish searches for the most stationary data
  segment of the requested length.

- `pulseWidthPercentile`: If low frequency pulse fish are missed then
  reduce this number.

- `eodMaxEODs`: the average waveform is estimated by averaging over at
  maximum this number of EODs. If wavefish change their frequency then
  you do not want to set this number too high (10 to 100 is enough for
  reducing noise). If you have several fish on your recording then
  this number needs to be high (1000) to average away the other fish.
  Set it to zero in order to use all EODs in the data segement
  selected for analysis.

- `eodExponentialFitFraction`: An exponential is fitted to the tail of
  the last peak/trough of EODs of pulse-type fish. This number
  (between 0 and 1) controls how much of the tail is used for the fit.

- `fileFormat`: sets the default file format to be used for storing
  the analysis results.


## Summary plot

In the plot you can press
- `q`: Close the plot and show the next one or quit.
- `p`: Play the analyzed section of the reording on the default audio device.
- `o`: Switch on zoom mode. You can draw a rectangle with the mouse to zoom in.
- `Backspace`: Zoom back. 
- `f`: Toggle full screen mode.


## Output files

With the `-s` switch analysis results are saved to files.

Output files are placed in the current working directory if no path is
specified via the `-o` switch. If the path specified via `-o` does not
exist it is created.

The following files are generated:
- `RECORDING-eodwaveform-N.EXT`: averaged EOD waveform
- `RECORDING-wavefish.EXT`: list of all detected wave-type fish
- `RECORDING-wavespectrum-N.EXT`: for each wave-type fish the Fourier spectrum
- `RECORDING-pulsefish.EXT`: list of all detected pulse-type fish
- `RECORDING-pulsepeaks-N.EXT`: for each pulse-type fish properties of peaks and troughs
- `RECORDING-pulsespectrum-N.EXT`: for each pulse-type fish the power spectrum of a singel pulse

The filenames are composed of the basename of the input file (RECORDING).
The fish detected in the recordings are numbered, starting with 0 (N).
The file extension depends on the chosen file format (EXT).
The following sections describe the content of the generated files.


### RECORDING-eodwaveform-N.EXT

For each fish the average waveform with standard deviation and fit.

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
6. Root-mean-squared difference between the averaged EOD waveform and 
   the fit of the Fourier series relative to the peak-to_peak amplitude in percent.
7. Number of EODs used for computing the averaged EOD waveform.


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


### RECORDING-pulsefish.EXT

Properties of each pulse-type fish detected in the recording.

<table>
<thead>
  <tr>
    <th align="left" colspan="13">waveform</th>
    <th align="left" colspan="5">power spectrum</th>
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
    <th align="left">tau</th>
    <th align="left">firstpeak</th>
    <th align="left">lastpeak</th>
    <th align="left">n</th>
    <th align="left">peakfreq</th>
    <th align="left">peakpower</th>
    <th align="left">poweratt5</th>
    <th align="left">poweratt50</th>
    <th align="left">lowcutoff</th>
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
    <th align="left">-</th>
    <th align="left">-</th>
    <th align="left">Hz</th>
    <th align="left">dB</th>
    <th align="left">dB</th>
    <th align="left">dB</th>
    <th align="left">Hz</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="right">0</td>
    <td align="right">30.68</td>
    <td align="right">32.60</td>
    <td align="right">0.797</td>
    <td align="right">0.838</td>
    <td align="right">1.635</td>
    <td align="right">-0.295</td>
    <td align="right">0.884</td>
    <td align="right">1.179</td>
    <td align="right">0.125</td>
    <td align="right">1</td>
    <td align="right">2</td>
    <td align="right">100</td>
    <td align="right">895.65</td>
    <td align="right">-66.31</td>
    <td align="right">-19.88</td>
    <td align="right">-18.68</td>
    <td align="right">162.17</td>
  </tr>
</tbody>
</table>

The columns contain:
1. Index of the fish (the number that is also used to number the files).
2. EOD frequency in Hertz.
3. Period between two pulses (1/EODf) in milliseconds.
4. Amplitude of the largest peak (P1 peak) in the units of the input data.
5. Amplitude of the largest trough in the units of the input data.
6. Peak-to-peak amplitude in the units of the input data.
7. Time where the pulse starts relative to P1 in milliseconds.
8. Time where the pulse ends relative to P1 in milliseconds.
9. Total width of the pulse in milliseconds.
10. Time constant of the exponential decay of the tail of the pulse in milliseconds.
11. Index of the first peak in the pulse (i.e. -1 for P-1)
12. Index of the last peak in the pulse (i.e. 3 for P3)
13. Number of EODs used for computing the averaged EOD waveform.
14. Frequency at the peak power of the single pulse spectrum in Hertz.
15. Peak power of the single pulse spectrum in decibel.
16. How much the average power below 5 Hz is attenuated relative to the peak power in decibel.
17. How much the average power below 50 Hz is attenuated relative to the peak power in decibel.
18. Frequency at which the power reached half of the peak power relative to the initial power in Hertz.


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

