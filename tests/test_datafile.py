from nose.tools import assert_true, assert_equal, assert_almost_equal, assert_raises
import os
import numpy as np
import thunderfish.datafile as dfl

def setup_table():
    df = dfl.DataFile()
    df.add_section("data")
    df.add_section("info")
    df.add_column("size", "m", "%.2f", 2.34)
    df.add_column("weight", "kg", "%.0f", 122.8)
    df.add_section("reaction")
    df.add_column("speed", "m/s", "%.3g", 98.7)
    df.add_column("jitter", "mm", "%.0f", 23)
    df.add_column("size", "g", "%.2e", 1.234)
    df.add_data((56.7, float('NaN'), 0.543, 45, 1.235e2), 0)
    df.add_data((8.9, 43.21, 6789.1, 3405, 1.235e-4), 0)
    for k in range(5):
        df.add_data(0.5*(1.0+k)*np.random.randn(5)+10.+k, 0)
    df.adjust_columns()
    return df

def test_write():
    df = setup_table()
    for number_cols in [None, 'index', 'num', 'aa', 'AA']:
        for tf in dfl.DataFile.formats:
            df.write(table_format=tf, number_cols=number_cols)

def test_properties():
    df = setup_table()
    assert_equal(len(df), 5, 'len() failed %d' % len(df))
    assert_equal(df.columns(), 5, 'columns() failed %d' % df.columns())
    assert_equal(df.rows(), 8, 'rows() failed %d' % df.rows())
    assert_equal(df.shape, (5, 8), 'shape failed %d %d' % df.shape)

def test_write_load():
    df = setup_table()
    
    for number_cols in [None, 'index', 'num', 'aa', 'AA']:
        for tf in dfl.DataFile.formats[:-1]:
            orgfilename = 'test.' + dfl.DataFile.extensions[tf]
            with open(orgfilename, 'w') as ff:
                df.write(ff, table_format=tf, number_cols=number_cols)
            sf = dfl.DataFile(orgfilename)
            sf.adjust_columns()
            filename = 'test-read.' + dfl.DataFile.extensions[tf]
            with open(filename, 'w') as ff:
                sf.write(ff, table_format=tf, number_cols=number_cols)
            with open(orgfilename, 'r') as f1, open(filename, 'r') as f2:
                for k, (line1, line2) in enumerate(zip(f1, f2)):
                    if line1 != line2:
                        print('%s: %s' % (tf, dfl.DataFile.descriptions[tf]))
                        print('files differ!')
                        print('original table:')
                        df.write(sys.stdout, table_format=tf, number_cols=number_cols)
                        print('')
                        print('read in table:')
                        sf.write(sys.stdout, table_format=tf, number_cols=number_cols)
                        print('')
                        print('line %2d "%s" from original table does not match\n        "%s" from read in table.' % (k+1, line1.strip(), line2.strip()))
                    assert_equal(line1, line2, 'files differ')
            os.remove(orgfilename)
            os.remove(filename)
