# eodexplorer

View and explore properties of EOD waveforms.

## Authors

The [Neuroethology-lab](https://uni-tuebingen.de/en/faculties/faculty-of-science/departments/biology/institutes/neurobiology/lehrbereiche/neuroethology/) at the Institute of Neuroscience at the University of T&uuml;bingen:
- Jan Benda


## Command line arguments

```
eodexplorer --help
```
returns
```
usage: eodexplorer [-h] [--version] [-l] [-j [JOBS]]
                   [-D {all,allpower,noise,timing,ampl,relampl,power,relpower,phase,time,width,none}]
                   [-d COLUMN] [-n MAX] [-w {first,second,ampl,power,phase}]
                   [-s] [-c COLUMN] [-m CMAP] [-p PATH] [-P PATH]
                   [-f {dat,ascii,csv,rtai,md,tex,html,same}]
                   file

View and explore properties of EOD waveforms.

positional arguments:
  file                  a wavefish.* or pulsefish.* summary file as generated
                        by collectfish

optional arguments:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
  -l                    list all available data columns and exit
  -j [JOBS]             number of jobs run in parallel for loading waveform
                        data. Without argument use all CPU cores.
  -D {all,allpower,noise,timing,ampl,relampl,power,relpower,phase,time,width,none}
                        default selection of data columns, check them with the
                        -l option
  -d COLUMN             data columns to be appended or removed (if already
                        listed) for analysis
  -n MAX                maximum number of harmonics or peaks to be used
  -w {first,second,ampl,power,phase}
                        add first or second derivative of EOD waveform, or
                        relative amplitude, power, or phase to the plot of
                        selected EODs.
  -s                    save PCA components and exit
  -c COLUMN             data column to be used for color code or "index"
  -m CMAP               name of color map
  -p PATH               path to the analyzed EOD waveform data
  -P PATH               path to the raw EOD recordings
  -f {dat,ascii,csv,rtai,md,tex,html,same}
                        file format used for saving PCA data ("same" uses same
                        format as input file)

version 1.8 by Benda-Lab (2019-2019)

mouse:
left click              select data points
left and drag           rectangular selection and zoom of data points
shift + left click/drag add data points to selection
ctrl + left click/drag  add/remove data points to/from selection

key shortcuts:
l                       list selected EOD waveforms on console
p,P                     toggle between data columns, PC, and scaled PC axis
<, pageup               decrease number of displayed data columns/PC axis
>, pagedown             increase number of displayed data columns/PC axis
w                       toggle maximized waveform plot
o, z                    toggle zoom mode on or off
backspace               zoom back
ctrl + a                select all
+, -                    increase, decrease pick radius
0                       reset pick radius
n, N                    decrease, increase number of bins of histograms
h                       toggle between scatter plot and 2D histogram
c, C                    cycle color map trough data columns
left, right, up, down   show and move magnified scatter plot
escape                  close magnified scatter plot
```