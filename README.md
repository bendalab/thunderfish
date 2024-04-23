[![license](https://img.shields.io/pypi/l/thunderfish.svg)](https://github.com/bendalab/thunderfish/blob/master/LICENSE)
[![tests](https://github.com/bendalab/thunderfish/workflows/tests/badge.svg?dummy=42)](https://github.com/bendalab/thunderfish/actions)
[![codecov](https://bendalab.github.io/thunderfish/coverage.svg?dummy=42)](https://bendalab.github.io/thunderfish/cover)
[![PyPI version](https://img.shields.io/pypi/v/thunderfish.svg)](https://pypi.python.org/pypi/thunderfish/)
![downloads](https://img.shields.io/pypi/dm/thunderfish.svg)
[![contributors](https://img.shields.io/github/contributors/bendalab/thunderfish)](https://github.com/bendalab/thunderfish/graphs/contributors)
[![commits](https://img.shields.io/github/commit-activity/m/bendalab/thunderfish)](https://github.com/bendalab/thunderfish/pulse)
<!--
![python](https://img.shields.io/pypi/pyversions/thunderfish.svg)
![issues open](https://img.shields.io/github/issues/bendalab/thunderfish.svg)
![issues closed](https://img.shields.io/github/issues-closed/bendalab/thunderfish.svg)
![pullrequests open](https://img.shields.io/github/issues-pr/bendalab/thunderfish.svg)
![pullrequests closed](https://img.shields.io/github/issues-pr-closed/bendalab/thunderfish.svg)
-->

# ThunderFish

Algorithms and programs for analysing electric field recordings of
weakly electric fish.

[Documentation](https://bendalab.github.io/thunderfish/) |
[API Reference](https://bendalab.github.io/thunderfish/api/)

Weakly electric fish generate an electric organ discharge (EOD).  In
wave-type fish the EOD resembles a sinewave of a specific frequency
and with higher harmonics. In pulse-type fish EODs have a distinct
waveform and are separated in time. The thunderfish package provides
algorithms and tools for analysing both wavefish and pulsefish EODs.

![thunderfish](docs/img/Apteronotus-Fishfinder-Panama-RioCanita-2014-05-17-L13-c0.png)


## Installation

ThunderFish is available from
[PyPi](https://pypi.org/project/thunderfish/). Simply run:
```
pip install thunderfish
```


## Software

The thunderfish package provides the following software:

- [`fishfinder`](https://bendalab.github.io/thunderfish/fishfinder/): Browse EOD recordings and detect EOD frequencyies on the fly.
- [`thunderfish`](https://bendalab.github.io/thunderfish/thunderfish/): Detect, analyze, and plot all EOD waveforms in short recordings.
- [`collectfish`](https://bendalab.github.io/thunderfish/collectfish/): Collect data generated by [`thunderfish`](https://bendalab.github.io/thunderfish/thunderfish/).
- [`eodexplorer`](https://bendalab.github.io/thunderfish/eodexplorer/): View and explore properties of EOD waveforms.
- [`thunderbrowse`](https://bendalab.github.io/thunderfish/api/thunderbrowse/): Browse multi-channel EOD recordings.
- [`thunderlogger`](https://bendalab.github.io/thunderfish/api/thunderlogger/): Extract EOD waveforms from logger recordings.


## Algorithms

The following modules provide the algorithms for analyzing EOD recordings.
Look into the modules for more information.

### Input/output

- [`hopkinsloader`](https://bendalab.github.io/thunderfish/api/hopkinsloader.html): Load EODs from Hopkins files.

### EOD analysis

- [`bestwindow`](https://bendalab.github.io/thunderfish/api/bestwindow.html): Select the region within a recording with the most stable signal of largest amplitude that is not clipped.
- [`checkpulse`](https://bendalab.github.io/thunderfish/api/checkpulse.html): Check whether a pulse-type or a wave-type weakly electric fish is present in a recording.
- [`consistentfishes`](https://bendalab.github.io/thunderfish/api/consistentfishes.html): Create a list of EOD frequencies with fishes present in all provided fish lists.
- [`eodanalysis`](https://bendalab.github.io/thunderfish/api/eodanalysis.html): Analyse EOD waveforms.
- [`harmonics`](https://bendalab.github.io/thunderfish/api/harmonics.html): Extract and analyze harmonic frequencies from power spectra.
- [`pulses`](https://bendalab.github.io/thunderfish/api/pulses.html): Extract and cluster EOD waverforms of pulse-type electric fish.

### EOD simulations

- [`fakefish`](https://bendalab.github.io/thunderfish/api/fakefish.html): Simulate EOD waveforms.
- [`efield`](https://bendalab.github.io/thunderfish/api/efield.html): Simulations of spatial electric fields.
- [`fishshapes`](https://bendalab.github.io/thunderfish/api/fishshapes.html): Manipulate and plot fish outlines.



