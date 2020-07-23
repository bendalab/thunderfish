"""
# Collect data generated by thunderfish in a wavefish and a pulsefish table.
"""

import os
import sys
import argparse
import numpy as np
from .version import __version__, __year__
from .configfile import ConfigFile
from .tabledata import TableData, add_write_table_config, write_table_args
from .harmonics import add_harmonic_groups_config
from .eodanalysis import wave_quality, wave_quality_args, add_eod_quality_config
from .eodanalysis import pulse_quality, pulse_quality_args
from .eodanalysis import adjust_eodf


def collect_fish(files, insert_file=True, append_file=False, simplify_file=False,
                 meta_data=None, meta_recordings=None, skip_recordings=False,
                 temp_col=None, q10=1.62, add_species=False, max_fish=0, harmonics=None,
                 peaks0=None, peaks1=None, cfg=None, verbose=0):
    """
    Combine all *-wavefish.* and/or *-pulsefish.* files into respective summary tables.

    Data from the *-wavespectrum-*.* and the *-pulsepeaks-*.* files can be added
    as specified by `harmonics`, `peaks0`, and `peaks1`.

    Meta data of the recordings can also be added via `meta_data` and
    `meta_recordings`.  If `meta_data` contains a column with
    temperature, this column can be specified by the `temp_col`
    parameter. In this case, an 'T_adjust' and an 'EODf_adjust' column
    are inserted into the resulting tables containing the mean
    temperature and EOD frequencies adjusted to this temperature,
    respectively. For the temperature adjustment of EOD frequency via
    the Q10 value can be supplied by the `q10` parameter.

    Parameters
    ----------
    files: list of strings
        Files to be combined.
    insert_file: boolean
        Insert the basename of the recording file as the first column.
    append_file: boolean
        Add the basename of the recording file as the last column.
        Overwrites `insert_file`.
    simplify_file: boolean
        Remove initial common directories from input files.
    meta_data: TableData or None
        Table with additional data for each of the recordings.
        The meta data are inserted into the summary table according to
        the name of the recording as specified in `meta_recordings`.
    meta_recordings: array of strings
        For each row in `meta_data` the name of the recording.
        This name is matched agains the basename of input `files`.
    skip_recordings: bool
        If True skip recordings that are not found in `meta_recordings`.
    temp_col: string or None
        A column in `meta_data` with temperatures to which EOD frequences should be adjusted.
    q10: float
        Q10 value describing temperature dependence of EOD frequencies.
        The default of 1.62 is from Dunlap, Smith, Yetka (2000) Brain Behav Evol,
        measured for Apteronotus lepthorhynchus in the lab.
    add_species: bool
        If True add column with identified wavefish species (experimental).
    max_fish: int
        Maximum number of fish to be taken, if 0 take all.
    harmonics: int
        Number of harmonic to be added to the wave-type fish table (amplitude, relampl, phase).
        This data is read in from the corresponding *-wavespectrum-*.* files.
    peaks0: int
        Index of the first peak of a EOD pulse to be added to the pulse-type fish table.
        This data is read in from the corresponding *-pulsepeaks-*.* files.
    peaks1: int
        Index of the last peak of a EOD pulse to be added to the pulse-type fish table.
        This data is read in from the corresponding *-pulsepeaks-*.* files.
    cfg: ConfigFile
        Configuration parameter for EOD quality assessment.
    verbose: int
        Verbose output. 1: print infos on meta data coverage, 2: print additional infos on discarded recordings.

    Returns
    -------
    wave_table: TableData
        Summary table for all wave-type fish.
    pulse_table: TableData
        Summary table for all pulse-type fish.
    """
    def find_recording(recording, meta_recordings):
        """ Find row of a recording in meta data.

        Parameters
        ----------
        recording: string
            Base name of a recording.
        meta_recordings: list of string
            List of meta data recordings where to find `recording`.
        """
        if meta_data is not None:
            rec = os.path.splitext(os.path.basename(recording))[0]
            for i in range(len(meta_recordings)):
                if rec == meta_recordings[i]:
                    return i
        return -1
        
    if append_file and insert_file:
        insert_file = False
    # prepare meta recodings names:
    meta_recordings_used = None
    if meta_recordings is not None:
        meta_recordings_used = np.zeros(len(meta_recordings), dtype=np.bool)
        for r in range(len(meta_recordings)):
            meta_recordings[r] = os.path.splitext(os.path.basename(meta_recordings[r]))[0]
    # prepare adjusted temperatures:
    if meta_data is not None and temp_col is not None:
        temp_idx = meta_data.index(temp_col)
        temp = meta_data[:,temp_idx]
        mean_tmp = np.round(np.nanmean(temp)/0.1)*0.1
        meta_data.insert(temp_idx+1, 'T_adjust', 'C', '%.1f')
        meta_data.append_data_column([mean_tmp]*meta_data.rows(), temp_idx+1)
    # load data:    
    wave_table = None
    pulse_table = None
    all_table = None
    file_pathes = []
    for file_name in files:
        # file name:
        table = None
        base_path, file_ext = os.path.splitext(file_name)[0:2]
        if base_path.endswith('-pulsefish'):
            base_path = base_path[:-10]
            fish_type = 'pulse'
        elif base_path.endswith('-wavefish'):
            base_path = base_path[:-9]
            fish_type = 'wave'
        else:
            continue
        if base_path.startswith('./'):
            base_path = base_path[2:]
        recording = base_path
        file_pathes.append(os.path.normpath(recording).split(os.path.sep))
        # find row in meta_data:
        mr = -1
        if meta_data is not None:
            mr = find_recording(recording, meta_recordings)
            if mr < 0:
                if skip_recordings:
                    if verbose > 0:
                        print('skip recording %s: no metadata found' % recording)
                    continue
                elif verbose > 0:
                    print('no metadata found for recording %s' % recording)
            else:
                meta_recordings_used[mr] = True
        # data:
        data = TableData(file_name)
        table = wave_table if fish_type == 'wave' else pulse_table
        # prepare tables:
        if not table:
            df = TableData(data)
            df.clear_data()
            if meta_data is not None:
                for s in range(data.nsecs):
                    df.insert_section(0, 'metadata')
                for c in range(meta_data.columns()):
                    df.insert(c, *meta_data.column_head(c))
            if insert_file:
                df.insert(0, ['recording']*data.nsecs + ['file'], '', '%-s')
            if fish_type == 'wave':
                if harmonics is not None:
                    wave_spec = TableData(base_path + '-wavespectrum-0' + file_ext)
                    if data.nsecs > 0:
                        df.append_section('harmonics')
                    for h in range(min(harmonics, wave_spec.rows())+1):
                        df.append('ampl%d' % h, wave_spec.unit('amplitude'),
                                      wave_spec.format('amplitude'))
                        if h > 0:
                            df.append('relampl%d' % h, '%', '%.2f')
                            df.append('relpower%d' % h, '%', '%.2f')
                        df.append('phase%d' % h, 'rad', '%.3f')
            else:
                if peaks0 is not None:
                    pulse_peaks = TableData(base_path + '-pulsepeaks-0' + file_ext)
                    if data.nsecs > 0:
                        df.append_section('peaks')
                    for p in range(peaks0, peaks1+1):
                        if p != 1:
                            df.append('P%dtime' % p, 'ms', '%.3f')
                        df.append('P%dampl' % p, pulse_peaks.unit('amplitude'),
                                  pulse_peaks.format('amplitude'))
                        if p != 1:
                            df.append('P%drelampl' % p, '%', '%.2f')
                        df.append('P%dwidth' % p, 'ms', '%.3f')
            if append_file:
                df.append(['recording']*data.nsecs + ['file'], '', '%-s')
            if fish_type == 'wave':
                wave_table = df
                table = wave_table
            else:
                pulse_table = df
                table = pulse_table
            if not all_table:
                df = TableData()
                if insert_file:
                    df.append('file', '', '%-s')
                if meta_data is not None:
                    for c in range(meta_data.columns()):
                        df.append(*meta_data.column_head(c))
                df.append('index', '', '%d')
                df.append('EODf', 'Hz', '%.1f')
                df.append('type', '', '%-5s')
                if append_file:
                    df.append('file', '', '%-s')
                all_table = df
        # fill tables:
        n = data.rows() if not max_fish or max_fish > data.rows() else max_fish
        for r in range(n):
            # fish index:
            idx = r
            if 'index' in data:
                idx = data[r,'index']
            # check quality:
            skips = ''
            if fish_type == 'wave':
                wave_spec = TableData(base_path + '-wavespectrum-%d'%idx + file_ext)
                if cfg is not None:
                    props = data.row_dict(r)
                    props['clipped'] *= 0.01 
                    props['noise'] *= 0.01 
                    props['rmserror'] *= 0.01 
                    _, skips, msg = wave_quality(props, **wave_quality_args(cfg))
            else:
                if cfg is not None:
                    props = data.row_dict(r)
                    props['clipped'] *= 0.01 
                    props['noise'] *= 0.01
                    skips, msg, _ = pulse_quality(props, **pulse_quality_args(cfg))
            if len(skips) > 0:
                if verbose > 1:
                    print('skip fish %2d from %s: %s' % (idx, recording, skips))
                continue
            # fill in data:
            data_col = 0
            if insert_file:
                table.append_data(recording, data_col)
                all_table.append_data(recording, data_col)
                data_col += 1
            if mr >= 0:
                for c in range(meta_data.columns()):
                    table.append_data(meta_data[mr,c], data_col)
                    all_table.append_data(meta_data[mr,c], data_col)
                    data_col += 1
            elif meta_data is not None:
                data_col += meta_data.columns()
            table.append_data(data[r,:].array(), data_col)
            all_table.append_data(data[r,'index'], data_col)
            all_table.append_data(data[r,'EODf'])
            all_table.append_data(fish_type)
            if peaks0 is not None and fish_type == 'pulse':
                pulse_peaks = TableData(base_path + '-pulsepeaks-%d'%idx + file_ext)
                for p in range(peaks0, peaks1+1):
                    for pr in range(pulse_peaks.rows()):
                        if pulse_peaks[pr,'P'] == p:
                            break
                    else:
                        continue
                    if p != 1:
                        table.append_data(pulse_peaks[pr,'time'], 'P%dtime' % p)
                    table.append_data(pulse_peaks[pr,'amplitude'], 'P%dampl' % p)
                    if p != 1:
                        table.append_data(pulse_peaks[pr,'relampl'], 'P%drelampl' % p)
                    table.append_data(pulse_peaks[pr,'width'], 'P%dwidth' % p)
            elif harmonics is not None and fish_type == 'wave':
                for h in range(min(harmonics, wave_spec.rows())+1):
                    table.append_data(wave_spec[h,'amplitude'])
                    if h > 0:
                        table.append_data(wave_spec[h,'relampl'])
                        table.append_data(wave_spec[h,'relpower'])
                    table.append_data(wave_spec[h,'phase'])
            if append_file:
                table.append_data(recording)
                all_table.append_data(recording)
            table.fill_data()
            all_table.fill_data()
    # check coverage of meta data:
    if meta_recordings_used is not None:
        if np.all(meta_recordings_used):
            if verbose > 0:
                print('found recordings for all meta data')
        else:
            if verbose > 0:
                print('no recordings found for:')
            for mr in range(len(meta_recordings)):
                recording = meta_recordings[mr]
                if not meta_recordings_used[mr]:
                    if verbose > 0:
                        print(recording)
                    all_table.set_column(0)
                    if insert_file:
                        all_table.append_data(recording)
                    for c in range(meta_data.columns()):
                        all_table.append_data(meta_data[mr,c])
                    all_table.append_data(np.nan) # index
                    all_table.append_data(np.nan) # EODf
                    all_table.append_data('none') # type
                    if append_file:
                        all_table.append_data(recording)
    # adjust EODf to mean temperature:
    for table in [wave_table, pulse_table, all_table]:
        if table is not None and temp_col is not None:
            eodf_idx = table.index('EODf')
            table.insert(eodf_idx+1, 'EODf_adjust', 'Hz', '%.1f')
            table.fill_data()
            temp_idx = table.index(temp_col)
            tadjust_idx = table.index('T_adjust')
            for r in range(table.rows()):
                eodf = table[r,eodf_idx]
                if np.isfinite(table[r,temp_col]) and np.isfinite(table[r,tadjust_idx]):
                    eodf = adjust_eodf(eodf, table[r,temp_col], table[r,tadjust_idx], q10)
                table[r,eodf_idx+1] = eodf
    # add wavefish species (experimental):
    if add_species and wave_table:
        eodfs = 'EODf_adjust'
        if eodfs not in wave_table:
            eodfs = 'EODf'
        species = np.zeros(wave_table.rows(), object)
        species[wave_table[:,eodfs] < 250.0] = 'Sterno'
        species[(wave_table[:,'reltroughampl'] < 72.0) & (wave_table[:,eodfs] > 250.0) & (wave_table[:,eodfs] < 600.0)] = 'Eigen'
        species[(wave_table[:,eodfs] > 600.0) | ((wave_table[:,'reltroughampl'] > 72.0) & (wave_table[:,eodfs] > 500.0))] = 'Aptero'
        species[(wave_table[:,'reltroughampl'] > 72.0) & (wave_table[:,eodfs] > 250.0) & (wave_table[:,eodfs] < 500.0)] = 'unknown'
        if append_file:
            wave_table.insert(wave_table.columns()-1, 'species', '', '%-s', species)
        else:
            wave_table.append('species', '', '%-s', species)
        if all_table:
            if append_file:
                sc = all_table.insert(all_table.columns()-1, 'species', '', '%-s')
            else:
                sc = all_table.append('species', '', '%-s')
            tc = all_table.index('type')
            wi = 0
            for r in range(all_table.rows()):
                if all_table[r,tc] == 'wave':
                    all_table.append_data(species[wi], sc)
                    wi += 1
                else:
                    all_table.append_data(all_table[r,tc], sc)
    # simplify pathes:
    if simplify_file and len(file_pathes) > 1:
        fp0 = file_pathes[0]
        for fi in range(len(fp0)):
            is_same = True
            for fp in file_pathes[1:]:
                if fi >= len(fp) or fp[fi] != fp0[fi]:
                    is_same = False
                    break
            if not is_same:
                break
        for table in [wave_table, pulse_table, all_table]:
            if table is not None:
                for k in range(table.rows()):
                    idx = table.index('file')
                    fps = os.path.normpath(table[k,idx]).split(os.path.sep)
                    table[k,idx] = os.path.sep.join(fps[fi:])
    return wave_table, pulse_table, all_table

    
def rangestr(string):
    """
    Parse string of the form N:M .
    """
    if string[0] == '=':
        string = '-' + string[1:]
    ss = string.split(':')
    v0 = v1 = None
    if len(ss) == 1:
        v0 = int(string)
        v1 = v0
    else:
        v0 = int(ss[0])
        v1 = int(ss[1])
    return (v0, v1)


def main():
    # command line arguments:
    parser = argparse.ArgumentParser(add_help=True,
        description='Collect data generated by thunderfish in a wavefish and a pulsefish table.',
        epilog='version %s by Benda-Lab (2019-%s)' % (__version__, __year__))
    parser.add_argument('--version', action='version', version=__version__)
    parser.add_argument('-v', action='count', dest='verbose', default=0,
                        help='verbosity level: -v for meta data coverage, -vv for additional info on discarded recordings.')
    parser.add_argument('-t', dest='table_type', default=None, choices=['wave', 'pulse'],
                        help='wave-type or pulse-type fish')
    parser.add_argument('-a', dest='append_file', action='store_true',
                        help='append the file name as the last column')
    parser.add_argument('-c', dest='simplify_file', action='store_true',
                        help='remove initial common directories from input files')
    parser.add_argument('-m', dest='max_fish', type=int, metavar='N',
                        help='maximum number of fish to be taken from each recording')
    parser.add_argument('-p', dest='pulse_peaks', type=rangestr,
                        default=(None, None), metavar='N:M',
                        help='add properties of peak N to M of pulse-type EODs to the table')
    parser.add_argument('-w', dest='harmonics', type=int, metavar='N',
                        help='add properties of first N harmonics of wave-type EODs to the table')
    parser.add_argument('-r', dest='remove_cols', action='append', default=[], metavar='COLUMN',
                        help='columns to be removed from output table')
    parser.add_argument('-s', dest='statistics', action='store_true',
                        help='also write table with statistics')
    parser.add_argument('-i', dest='meta_file', metavar='FILE:REC:TEMP', default='', type=str,
                        help='insert rows from metadata table in FILE matching recording in colum REC. The optional TEMP specifies a column with temperatures to which EOD frequencies should be adjusted')
    parser.add_argument('-q', dest='q10', metavar='Q10', default=1.62, type=float,
                        help='Q10 value for adjusting EOD frequencies to a common temperature')
    parser.add_argument('-g', dest='add_species', action='store_true', default=False,
                        help='append column with genus/species name (for wavefish only, experimental)')
    parser.add_argument('-S', dest='skip', action='store_true',
                        help='skip recordings that are not contained in metadata table')
    parser.add_argument('-n', dest='file_suffix', metavar='NAME', default='', type=str,
                        help='name for summary files that is appended to "wavefish" or "pulsefish"')
    parser.add_argument('-o', dest='out_path', metavar='PATH', default='.', type=str,
                        help='path where to store summary tables')
    parser.add_argument('-f', dest='format', default='auto', type=str,
                        choices=TableData.formats + ['same'],
                        help='file format used for saving summary tables ("same" uses same format as input files)')
    parser.add_argument('file', nargs='+', default='', type=str,
                        help='a *-wavefish.* or *-pulsefish.* file as generated by thunderfish')
    # fix minus sign issue:
    ca = []
    pa = False
    for a in sys.argv[1:]:
        if pa and a[0] == '-':
            a = '=' + a[1:]
        pa = False
        if a == '-p':
            pa = True
        ca.append(a)
    # read in command line arguments:    
    args = parser.parse_args(ca)
    verbose = args.verbose
    table_type = args.table_type
    remove_cols = args.remove_cols
    statistics = args.statistics
    meta_file = args.meta_file
    file_suffix = args.file_suffix
    out_path = args.out_path
    data_format = args.format
    # read configuration:
    cfgfile = __package__ + '.cfg'
    cfg = ConfigFile()
    add_harmonic_groups_config(cfg)
    add_eod_quality_config(cfg)
    add_write_table_config(cfg, table_format='csv', unit_style='row',
                           align_columns=True, shrink_width=False)
    cfg.load_files(cfgfile, args.file[0], 3)
    # output format:
    if data_format == 'same':
        ext = os.path.splitext(args.file[0])[1][1:]
        if ext in TableData.ext_formats:
            data_format = TableData.ext_formats[ext]
        else:
            data_format = 'dat'
    if data_format != 'auto':
        cfg.set('fileFormat', data_format)
    # create output folder:
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    # read in meta file:
    md = None
    rec_data = None
    temp_col = None
    if len(meta_file) > 0:
        mds = meta_file.split(':')
        meta_data = mds[0]
        md = TableData(meta_data)
        if len(mds) < 2:
            print('no recording column specified for the table in %s. Choose one of' % meta_data)
            for k in md.keys():
                print(' ', k)
            exit()
        rec_col = mds[1]
        if rec_col not in md:
            print('%s is not a valid key for the table in %s. Choose one of' % (rec_col, meta_data))
            for k in md.keys():
                print(' ', k)
            exit()
        else:
            rec_data = md[:,rec_col]
            del md[:,rec_col]
        if len(mds) > 2:
            temp_col = mds[2]
            if temp_col not in md:
                print('%s is not a valid key for the table in %s. Choose one of' % (temp_col, meta_data))
                for k in md.keys():
                    print(' ', k)
                exit()
    # collect files:
    wave_table, pulse_table, all_table = collect_fish(args.file, True, args.append_file,
                                           args.simplify_file, md, rec_data, args.skip,
                                           temp_col, args.q10, args.add_species,
                                           args.max_fish, args.harmonics,
                                           args.pulse_peaks[0],  args.pulse_peaks[1],
                                           cfg, verbose)
    # write tables:
    if len(file_suffix) > 0 and file_suffix[0] != '-':
        file_suffix = '-' + file_suffix
    tables = []
    table_names = []
    if pulse_table and (not table_type or table_type == 'pulse'):
        tables.append(pulse_table)
        table_names.append('pulse')
    if wave_table and (not table_type or table_type == 'wave'):
        tables.append(wave_table)
        table_names.append('wave')
    if all_table and not table_type:
        tables.append(all_table)
        table_names.append('all')
    for table, name in zip(tables, table_names):
        for rc in remove_cols:
            if rc in table:
                table.remove(rc)
        table.write(os.path.join(out_path, '%sfish%s' % (name, file_suffix)),
                    **write_table_args(cfg))
        if statistics:
            s = table.statistics()
            s.write(os.path.join(out_path, '%sfish%s-statistics' % (name, file_suffix)),
                    **write_table_args(cfg))


if __name__ == '__main__':
    main()
