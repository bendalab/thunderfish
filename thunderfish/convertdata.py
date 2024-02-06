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
usage: convertdata [-h] [--version] [-v] [-l] [-f FORMAT] [-e ENCODING] [-u [THRESH]] [-U [THRESH]] [-d FAC]
                   [-c CHANNELS] [-n NUM] [-o OUTPATH]
                   [file ...]

Convert, downsample, rename, and merge data files.

positional arguments:
  file         one or more input data files to be combined into a single output file

options:
  -h, --help   show this help message and exit
  --version    show program's version number and exit
  -v           print debug output
  -l           list supported file formats and encodings
  -f FORMAT    data format of output file
  -e ENCODING  data encoding of output file
  -u [THRESH]  unwrap clipped data with threshold (default is 0.5) and divide by two
  -U [THRESH]  unwrap clipped data with threshold (default is 0.5) and clip
  -d FAC       downsample by integer factor
  -c CHANNELS  comma and dash separated list of channels to be saved (first channel is 0)
  -n NUM       merge NUM input files into one output file (default is all files)
  -o OUTPATH   path or filename of output file. Metadata keys enclosed in curly braces will be replaced by their
               values from the input file

version 1.12.0 by Benda-Lab (2020-2024)
```

"""

import os
import sys
import argparse
import numpy as np
from scipy.signal import decimate
from .version import __version__, __year__
from audioio import unwrap, flatten_metadata
from .dataloader import load_data, metadata
from .datawriter import available_formats, available_encodings
from .datawriter import format_from_extension, write_data


def check_format(format):
    """
    Check whether requested data format is valid and supported.

    If the format is not available print an error message on console.

    Parameters
    ----------
    format: str
        Data format to be checked.

    Returns
    -------
    valid: bool
        True if the requested data format is valid.
    """
    if not format or format.upper() not in available_formats():
        print(f'! invalid data format "{format}"!')
        print('run')
        print(f'> {__file__} -l')
        print('for a list of available formats')
        return False
    else:
        return True


def update_gain(md, fac):
    """ Update gain setting in metadata.

    Parameters
    ----------
    md: nested dict
        Metadata to be updated.
    fac: float
        Factor that was used to scale the data.

    Returns
    -------
    done: bool
        True if gain has been found and set.
    """
    for k in md:
        if k.strip().upper() == 'GAIN':
            vs = md[k]
            if isinstance(vs, (int, float)):
                md[k] /= fac
            else:
                # extract initial number:
                n = len(vs)
                ip = n
                for i in range(len(vs)):
                    if vs[i] == '.':
                        ip = i + 1
                    if not vs[i] in '0123456789.+-':
                        n = i
                        break
                v = float(vs[:n])
                u = vs[n:].removesuffix('/V')  # fix some TeeGrid gains
                nd = n - ip
                md[k] = f'{v/fac:.{nd}f}{u}'
            return True
        elif isinstance(md[k], dict):
            if update_gain(md[k], fac):
                return True
    return False
    

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
    parser.add_argument('--version', action='version', version=__version__)
    parser.add_argument('-v', action='count', dest='verbose', default=0,
                        help='print debug output')
    parser.add_argument('-l', dest='list_formats', action='store_true',
                        help='list supported file formats and encodings')
    parser.add_argument('-f', dest='data_format', default=None, type=str, metavar='FORMAT',
                        help='data format of output file')
    parser.add_argument('-e', dest='data_encoding', default=None, type=str, metavar='ENCODING',
                        help='data encoding of output file')
    parser.add_argument('-u', dest='unwrap', default=0, type=float,
                        metavar='THRESH', const=0.5, nargs='?',
                        help='unwrap clipped data with threshold (default is 0.5) and divide by two')
    parser.add_argument('-U', dest='unwrap_clip', default=0, type=float,
                        metavar='THRESH', const=0.5, nargs='?',
                        help='unwrap clipped data with threshold (default is 0.5) and clip')
    parser.add_argument('-d', dest='decimate', default=1, type=int,
                        metavar='FAC',
                        help='downsample by integer factor')
    parser.add_argument('-c', dest='channels', default='',
                        type=str, metavar='CHANNELS',
                        help='comma and dash separated list of channels to be saved (first channel is 0)')
    parser.add_argument('-n', dest='nmerge', default=0, type=int, metavar='NUM',
                        help='merge NUM input files into one output file (default is all files)')
    parser.add_argument('-o', dest='outpath', default=None, type=str,
                        help='path or filename of output file. Metadata keys enclosed in curly braces will be replaced by their values from the input file')
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

    if args.list_formats:
        if not args.data_format:
            print('available data formats:')
            for f in available_formats():
                print(f'  {f}')
        else:
            if not check_format(args.data_format):
                sys.exit(-1)
            print(f'available encodings for data format {args.data_format}:')
            for e in available_encodings(args.data_format):
                print(f'  {e}')
        return

    if len(args.file) == 0:
        print('! need to specify at least one input file !')
        sys.exit(-1)
        
    nmerge = args.nmerge
    if nmerge == 0:
        nmerge = len(args.file)

    for i0 in range(0, len(args.file), nmerge):
        infile = args.file[i0]
        # output file and format:
        if nmerge < len(args.file) and args.outpath and \
           format_from_extension(args.outpath) is None and \
           not os.path.exists(args.outpath):
            os.mkdir(args.outpath)
        data_format = args.data_format
        if not args.outpath or os.path.isdir(args.outpath):
            outfile = infile
            if args.outpath:
                outfile = os.path.join(args.outpath, outfile)
            if not data_format:
                print('! need to specify a data format via -f or a file extension !')
                sys.exit(-1)
            outfile = os.path.splitext(outfile)[0] + os.extsep + data_format.lower()
        else:
            outfile = args.outpath
            if data_format:
                outfile = os.path.splitext(outfile)[0] + os.extsep + data_format.lower()
            else:
                data_format = format_from_extension(outfile)
        if not check_format(data_format):
            sys.exit(-1)
        if os.path.realpath(infile) == os.path.realpath(outfile):
            print(f'! cannot convert "{infile}" to itself !')
            sys.exit(-1)
        # read in data:
        data, samplingrate, unit = load_data(infile)
        md = metadata(infile)
        if args.verbose > 1:
            print(f'loaded data file "{infile}"')
        for infile in args.file[i0+1:i0+nmerge]:
            xdata, xrate, xunit = load_data(infile)
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
            data = np.vstack((data, xdata))
            if args.verbose > 1:
                print(f'loaded data file "{infile}"')
        # select channels:
        if len(channels) > 0:
            data = data[:,channels]
        # fix data:
        if args.unwrap_clip > 1e-3:
            unwrap(data, args.unwrap_clip)
            data[data > 1] = 1
            data[data < -1] = -1
        elif args.unwrap > 1e-3:
            unwrap(data, args.unwrap)
            data *= 0.5
            update_gain(md, 0.5)
        # decimate:
        if args.decimate > 1:
            data = decimate(data, args.decimate, axis=0)
            samplingrate /= args.decimate
        # metadata into file name:
        if len(md) > 0 and '{' in outfile and '}' in outfile:
            fmd = flatten_metadata(md)
            fmd = {k:(fmd[k] if isinstance(fmd[k], (int, float)) else fmd[k].replace(' ', '_')) for k in fmd}
            outfile = outfile.format(**fmd)
        # write out data:
        write_data(outfile, data, samplingrate, unit, md,
                   format=data_format, encoding=args.data_encoding)
        # message:
        if args.verbose > 1:
            print(f'wrote "{outfile}"')
        elif args.verbose:
            print(f'converted data file "{infile}" to "{outfile}"')


if __name__ == '__main__':
    main(*sys.argv[1:])
