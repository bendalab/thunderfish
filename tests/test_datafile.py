from nose.tools import assert_true, assert_equal, assert_almost_equal, assert_raises
import os
import sys
import numpy as np
import thunderfish.datafile as dfl

def setup_table():
    df = dfl.DataFile()
    df.append(["data", "info", "size"], "m", "%6.2f", [2.34, 56.7, 8.9])
    df.append("weight", "kg", "%.0f", 122.8)
    df.append_section("reaction")
    df.append("speed", "m/s", "%.3g", 98.7)
    df.append("jitter", "mm", "%.1f", 23)
    df.append("size", "g", "%.2e", 1.234)
    df.append_data(float('NaN'), 1)  # single value
    df.append_data((0.543, 45, 1.235e2)) # remaining row
    df.append_data((43.21, 6789.1, 3405, 1.235e-4), 1) # next row
    a = 0.5*np.arange(1, 6)*np.random.randn(5, 5) + 10.0 + np.arange(5)
    df.append_data(a.T, 0) # rest of table
    df[3:6,'weight'] = [11.0]*3
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
    assert_equal(df.shape, (8, 5), 'shape failed %d %d' % df.shape)

def test_write_load():
    df = setup_table()
    for units in ['auto', 'none', 'row', 'header']:
        for number_cols in [None, 'none', 'index', 'num', 'aa', 'AA']:
            for delimiter in [None, ';', '| ', '\t']:
                for tf in dfl.DataFile.formats[:-1]:
                    orgfilename = 'tabletest.' + dfl.DataFile.extensions[tf]
                    with open(orgfilename, 'w') as ff:
                        df.write(ff, table_format=tf, number_cols=number_cols, units=units,
                                 delimiter=delimiter)
                    sf = dfl.DataFile(orgfilename)
                    filename = 'tabletest-read.' + dfl.DataFile.extensions[tf]
                    with open(filename, 'w') as ff:
                        sf.write(ff, table_format=tf, number_cols=number_cols, units=units,
                                 delimiter=delimiter)
                    with open(orgfilename, 'r') as f1, open(filename, 'r') as f2:
                        for k, (line1, line2) in enumerate(zip(f1, f2)):
                            if line1 != line2:
                                print('%s: %s' % (tf, dfl.DataFile.descriptions[tf]))
                                print('files differ!')
                                print('original table:')
                                df.write(sys.stdout, table_format=tf, number_cols=number_cols,
                                         units=units, delimiter=delimiter)
                                print('')
                                print('read in table:')
                                sf.write(sys.stdout, table_format=tf, number_cols=number_cols,
                                         units=units, delimiter=delimiter)
                                print('')
                                print('line %2d "%s" from original table does not match\n        "%s" from read in table.' % (k+1, line1.rstrip('\n'), line2.rstrip('\n')))
                            assert_equal(line1, line2, 'files differ')
                    os.remove(orgfilename)
                    os.remove(filename)

def test_read_access():
    df = setup_table()
    df.clear()
    data = np.random.randn(10, df.columns())
    df.append_data(data)
    n = 1000
    # reading values by index:
    for c, r in zip(np.random.randint(0, df.columns(), n), np.random.randint(0, df.rows(), n)):
        assert_equal(df[r,c], data[r,c], 'element access by index failed')
    # reading values by column name:
    for c, r in zip(np.random.randint(0, df.columns(), n), np.random.randint(0, df.rows(), n)):
        assert_equal(df[r,df.keys()[c]], data[r,c], 'element access by column name failed')
    # reading row slices:
    for c in range(df.columns()):
        for r in np.random.randint(0, df.rows(), (n,2)):
            r0, r1 = np.sort(r)
            assert_true(np.array_equal(df[r0:r1,c], data[r0:r1,c]), 'slicing of rows failed')
    # reading column slices:
    for r in range(df.rows()):
        for c in np.random.randint(0, df.columns(), (n,2)):
            c0, c1 = np.sort(c)
            if c1-c0 < 2:
                continue
            assert_true(np.array_equal(df[r,c0:c1], data[r,c0:c1]), 'slicing of columns failed')
    # reading row and column slices:
    for c, r in zip(np.random.randint(0, df.columns(), (n,2)), np.random.randint(0, df.rows(), (n,2))):
        r0, r1 = np.sort(r)
        c0, c1 = np.sort(c)
        if c1-c0 < 2:
            continue
        assert_true(np.array_equal(df[r0:r1,c0:c1].array(), data[r0:r1,c0:c1]), 'slicing of rows and columns failed')
    # reading full column slices:
    for c in range(df.columns()):
        assert_true(np.array_equal(df[:,c], data[:,c]), 'slicing of full column failed')
        assert_true(np.array_equal(df.col(c)[:,0], data[:,c]), 'slicing of full column failed')
    for c, d in enumerate(df):
        assert_true(np.array_equal(d, data[:,c]), 'iterating of full column failed')
    # reading full row slices:
    for r in range(df.rows()):
        assert_true(np.array_equal(df[r,:], data[r,:]), 'slicing of full row failed')
        assert_true(np.array_equal(df.row(r)[0,:], data[r,:]), 'slicing of full row failed')


def test_write_access():
    df = setup_table()
    df.clear()
    data = np.random.randn(10, df.columns())
    df.append_data(data)
    n = 1000
    # writing and reading values by index:
    for c, r in zip(np.random.randint(0, df.columns(), n), np.random.randint(0, df.rows(), n)):
        v = np.random.randn()
        df[r,c] = v
        assert_equal(df[r,c], v, 'set item by index failed')
    # writing and reading row slices:
    for c in range(df.columns()):
        for r in np.random.randint(0, df.rows(), (n,2)):
            r0, r1 = np.sort(r)
            v = np.random.randn(r1-r0)
            df[r0:r1,c] = v
            assert_true(np.array_equal(df[r0:r1,c], v), 'slicing of rows failed')
    # writing and reading column slices:
    for r in range(df.rows()):
        for c in np.random.randint(0, df.columns(), (n,2)):
            c0, c1 = np.sort(c)
            if c1-c0 < 2:
                continue
            v = np.random.randn(c1-c0)
            df[r,c0:c1] = v
            assert_true(np.array_equal(df[r,c0:c1], v), 'slicing of columns failed')
    # writing and reading row and column slices:
    for c, r in zip(np.random.randint(0, df.columns(), (n,2)), np.random.randint(0, df.rows(), (n,2))):
        r0, r1 = np.sort(r)
        c0, c1 = np.sort(c)
        if c1-c0 < 2:
            continue
        v = np.random.randn(r1-r0, c1-c0)
        df[r0:r1,c0:c1] = v
        assert_true(np.array_equal(df[r0:r1,c0:c1].array(), v), 'slicing of rows and columns failed')

