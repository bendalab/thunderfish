"""Command line script for converting, downsampling, renaming and merging data files.

```sh
convertdata -o test.wav test.raw
```
converts 'test.raw' to 'test.wav'.

The script reads all input files with `dataloader.load_data()`,
combines the audio and marker data, and writes them along with the
metadata to an output file using `datawriter.write_data()`. Thus, all
formats supported by these functions and the installed python audio
modules are supported.

Run
```sh
convertdata -l
```
for a list of supported output file formats and
```sh
convertdata -f wav -l
```
for a list of supported encodings for a given output format.

Running
```sh
convertdata --help
```
prints
```text
usage: convertdata [-h] [--version] [-v] [-l] [-f FORMAT] [-e ENCODING] [-s SCALE] [-u [THRESH]] [-U [THRESH]]
                   [-d FAC] [-c CHANNELS] [-a KEY=VALUE] [-r KEY] [-n NUM] [-o OUTPATH]
                   [file ...]

Convert, downsample, rename, and merge data files.

positional arguments:
  file          one or more input files to be combined into a single output file

options:
  -h, --help    show this help message and exit
  --version     show program's version number and exit
  -v            print debug output
  -l            list supported file formats and encodings
  -f FORMAT     audio format of output file
  -e ENCODING   audio encoding of output file
  -s SCALE      scale the data by factor SCALE
  -u [THRESH]   unwrap clipped data with threshold (default is 0.5) and divide by two
  -U [THRESH]   unwrap clipped data with threshold (default is 0.5) and clip
  -d FAC        downsample by integer factor
  -c CHANNELS   comma and dash separated list of channels to be saved (first channel is 0)
  -a KEY=VALUE  add key-value pairs to metadata. Keys can have section names separated by "."
  -r KEY        remove keys from metadata. Keys can have section names separated by "."
  -n NUM        merge NUM input files into one output file
  -o OUTPATH    path or filename of output file. Metadata keys enclosed in curly braces will be replaced by their
                values from the input file

version 1.12.0 by Benda-Lab (2020-2024)
```

"""

import os
import sys
import argparse
import numpy as np
from .version import __version__, __year__
from audioio import add_metadata, remove_metadata, cleanup_metadata
from audioio import bext_history_str, add_history
from audioio.audioconverter import add_arguments, parse_channels
from audioio.audioconverter import make_outfile, format_outfile
from audioio.audioconverter import modify_data
from .dataloader import load_data, DataLoader, markers
from .datawriter import available_formats, available_encodings
from .datawriter import format_from_extension, write_data


def check_format(format):
    """
    Check whether requested audio format is valid and supported.

    If the format is not available print an error message on console.

    Parameters
    ----------
    format: string
        Audio format to be checked.

    Returns
    -------
    valid: bool
        True if the requested audio format is valid.
    """
    if not format or format.upper() not in available_formats():
        print(f'! invalid data file format "{format}"!')
        print('run')
        print(f'> {__file__} -l')
        print('for a list of available formats')
        return False
    else:
        return True


def list_formats_encodings(data_format):
    """ List available formats or encodings.

    Parameters
    ----------
    data_format: None or str
        If provided, list encodings for this data format.
    """
    if not data_format:
        print('available file formats:')
        for f in available_formats():
            print(f'  {f}')
    else:
        if not check_format(data_format):
            sys.exit(-1)
        print(f'available encodings for {data_format} file format:')
        for e in available_encodings(data_format):
            print(f'  {e}')


def main(*cargs):
    """Command line script for converting, downsampling, renaming and
    merging data files.

    Parameters
    ----------
    cargs: list of strings
        Command line arguments as returned by sys.argv[1:].

    """
    # command line arguments:
    parser = argparse.ArgumentParser(add_help=True,
        description='Convert, downsample, rename, and merge data files.',
        epilog=f'version {__version__} by Benda-Lab (2020-{__year__})')
    add_arguments(parser)
    if len(cargs) == 0:
        cargs = None
    args = parser.parse_args(cargs)
    
    channels = parse_channels(args.channels)
    
    if args.list_formats:
        if args.data_format is None and len(args.file) > 0:
            args.data_format = args.file[0]
        list_formats_encodings(args.data_format)
        return

    if len(args.file) == 0:
        print('! need to specify at least one input file !')
        sys.exit(-1)
        
    nmerge = args.nmerge
    if nmerge == 0:
        nmerge = len(args.file)

    for i0 in range(0, len(args.file), nmerge):
        infile = args.file[i0]
        outfile, data_format = make_outfile(args.outpath, infile,
                                            args.data_format,
                                            nmerge < len(args.file),
                                            format_from_extension)
        if not check_format(data_format):
            sys.exit(-1)
        if os.path.realpath(infile) == os.path.realpath(outfile):
            print(f'! cannot convert "{infile}" to itself !')
            sys.exit(-1)
        # read in data:
        pre_history = None 
        try:
            with DataLoader(infile) as sf:
                data = sf[:,:]
                samplingrate = sf.samplerate
                unit = sf.unit
                amax = sf.ampl_max
                md = sf.metadata()
                locs, labels = sf.markers()
                pre_history = bext_history_str(sf.encoding,
                                               sf.samplerate,
                                               sf.channels,
                                               sf.filepath)
                if sf.encoding is not None and args.encoding is None:
                    args.encoding = sf.encoding
        except FileNotFoundError:
            print(f'file "{infile}" not found!')
            sys.exit(-1)
        if args.verbose > 1:
            print(f'loaded data file "{infile}"')
        for infile in args.file[i0+1:i0+nmerge]:
            try:
                xdata, xrate, xunit, xamax = load_data(infile)
            except FileNotFoundError:
                print(f'file "{infile}" not found!')
                sys.exit(-1)
            if abs(samplingrate - xrate) > 1:
                print('! cannot merge files with different sampling rates !')
                print(f'    file "{args.file[i0]}" has {samplingrate:.0f}Hz')
                print(f'    file "{infile}" has {xrate:.0f}Hz')
                sys.exit(-1)
            if xdata.shape[1] != data.shape[1]:
                print('! cannot merge files with different numbers of channels !')
                print(f'    file "{args.file[i0]}" has {data.shape[1]} channels')
                print(f'    file "{infile}" has {xdata.shape[1]} channels')
                sys.exit(-1)
            if xamax > amax:
                amax = xamax
            data = np.vstack((data, xdata))
            xlocs, xlabels = markers(infile)
            locs = np.vstack((locs, xlocs))
            labels = np.vstack((labels, xlabels))
            if args.verbose > 1:
                print(f'loaded data file "{infile}"')
        data, samplingrate = modify_data(data, samplingrate, md,
                                         channels, args.scale,
                                         args.unwrap_clip,
                                         args.unwrap, amax, unit,
                                         args.decimate)
        add_metadata(md, args.md_list, '.')
        if len(args.remove_keys) > 0:
            remove_metadata(md, args.remove_keys, '.')
            cleanup_metadata(md)
        outfile = format_outfile(outfile, md)
        # history:
        hkey = 'CodingHistory'
        if 'BEXT' in md:
            hkey = 'BEXT.' + hkey
        history = bext_history_str(args.encoding, samplingrate,
                                   data.shape[1], outfile)
        add_history(md, history, hkey, pre_history)
        # write out data:
        try:
            write_data(outfile, data, samplingrate, amax, unit,
                       md, locs, labels,
                       format=data_format, encoding=args.encoding)
        except PermissionError:
            print(f'failed to write "{outfile}": permission denied!')
            sys.exit(-1)
        # message:
        if args.verbose > 1:
            print(f'wrote "{outfile}"')
        elif args.verbose:
            print(f'converted data file "{infile}" to "{outfile}"')


if __name__ == '__main__':
    main(*sys.argv[1:])
