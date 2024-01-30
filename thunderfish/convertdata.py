"""Command line script for converting data files.

```sh
convertdata -o test.wav test.raw
```
converts 'test.raw' to 'test.wav'.

The script basically reads all input files with
`dataloader.load_data()`, combines the audio data, and writes them
with `datawriter.write_data()`. Thus, all formats supported by these
functions and the installed python audio modules are supported.

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
usage: convertdata [-h] [--version] [-v] [-l] [-f FORMAT] [-e ENCODING] [-c CHANNELS] [-o OUTPATH] [file ...]

Convert data file formats.

positional arguments:
  file         one or more input data files to be combined into a single output file

options:
  -h, --help   show this help message and exit
  --version    show program's version number and exit
  -v           print debug output
  -l           list supported file formats and encodings
  -f FORMAT    data format of output file
  -e ENCODING  data encoding of output file
  -c CHANNELS  comma and dash separated list of channels to be saved (first channel is 0)
  -o OUTPATH   path or filename of output file

version 1.10.0 by Benda-Lab (2020-2024)
```

"""

import os
import sys
import argparse
import numpy as np
from .version import __version__, __year__
from .dataloader import load_data, metadata
from .datawriter import available_formats, available_encodings
from .datawriter import format_from_extension, write_data


def check_format(format):
    """
    Check whether requested data format is valid and supported.

    If the format is not available print an error message on console.

    Parameters
    ----------
    format: string
        Data format to be checked.

    Returns
    -------
    valid: bool
        True if the requested data format is valid.
    """
    if format and format.upper() not in available_formats():
        print(f'! invalid data format "{format}"!')
        print('run')
        print(f'> {__file__} -l')
        print('for a list of available formats')
        return False
    else:
        return True


def main(*cargs):
    """
    Command line script for converting data files.

    Parameters
    ----------
    cargs: list of strings
        Command line arguments as returned by sys.argv.
    """
    # command line arguments:
    parser = argparse.ArgumentParser(add_help=True,
        description='Convert data file formats.',
        epilog=f'version {__version__} by Benda-Lab (2020-{__year__})')
    parser.add_argument('--version', action='version', version=__version__)
    parser.add_argument('-v', action='store_true', dest='verbose',
                        help='print debug output')
    parser.add_argument('-l', dest='list_formats', action='store_true',
                        help='list supported file formats and encodings')
    parser.add_argument('-f', dest='data_format', default=None, type=str, metavar='FORMAT',
                        help='data format of output file')
    parser.add_argument('-e', dest='data_encoding', default=None, type=str, metavar='ENCODING',
                        help='data encoding of output file')
    parser.add_argument('-c', dest='channels', default='',
                        type=str, metavar='CHANNELS',
                        help='comma and dash separated list of channels to be saved (first channel is 0)')
    parser.add_argument('-o', dest='outpath', default=None, type=str,
                        help='path or filename of output file')
    parser.add_argument('file', nargs='*', type=str,
                        help='one or more input data files to be combined into a single output file')
    if len(cargs) == 0:
        cargs = None
    args = parser.parse_args(cargs)

    cs = [s.strip() for s in args.channels.split(',')]
    channels = []
    for c in cs:
        if len(c) == 0:
            continue
        css = [s.strip() for s in c.split('-')]
        if len(css) == 2:
            channels.extend(list(range(int(css[0]), int(css[1])+1)))
        else:
            channels.append(int(c))

    if not check_format(args.data_format):
        sys.exit(-1)

    if args.list_formats:
        if not args.data_format:
            print('available data formats:')
            for f in available_formats():
                print(f'  {f}')
        else:
            print(f'available encodings for data format {args.data_format}:')
            for e in available_encodings(args.data_format):
                print(f'  {e}')
        return

    if len(args.file) == 0:
        print('! need to specify at least one input file !')
        sys.exit(-1)
    infile = args.file[0]
    # output file:
    if not args.outpath or os.path.isdir(args.outpath):
        outfile = infile
        if args.outpath:
            outfile = os.path.join(args.outpath, outfile)
        if not args.data_format:
            args.data_format = 'wav'
        outfile = os.path.splitext(outfile)[0] + os.extsep + args.data_format
    else:
        outfile = args.outpath
        if args.data_format:
            outfile = os.path.splitext(outfile)[0] + os.extsep + args.data_format
        else:
            args.data_format = format_from_extension(outfile)
            if not args.data_format:
                args.data_format = 'wav'
                outfile = outfile + os.extsep + args.data_format
    check_format(args.data_format)
    if os.path.realpath(infile) == os.path.realpath(outfile):
        print(f'! cannot convert "{infile}" to itself !')
        sys.exit(-1)
    # read in data:
    data, samplingrate, unit = load_data(infile)
    md = metadata(infile)
    for infile in args.file[1:]:
        xdata, xrate, xunit = load_data(infile)
        if abs(samplingrate - xrate) > 1:
            print('! cannot merge files with different sampling rates !')
            print(f'    file "{args.file[0]}" has {samplingrate:.0f}Hz')
            print(f'    file "{infile}" has {xrate:.0f}Hz')
            sys.exit(-1)
        if xdata.shape[1] != data.shape[1]:
            print('! cannot merge files with different numbers of channels !')
            print(f'    file "{args.file[0]}" has {data.shape[1]} channels')
            print(f'    file "{infile}" has {xdata.shape[1]} channels')
            sys.exit(-1)
        data = np.vstack((data, xdata))
    # write out data:
    if len(channels) > 0:
        data = data[:,channels]
    write_data(outfile, data, samplingrate, md,
               format=args.data_format, encoding=args.data_encoding)
    # message:
    if args.verbose:
        print(f'converted data file "{infile}" to "{outfile}"')


if __name__ == '__main__':
    main(*sys.argv)
