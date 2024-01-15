from nose.tools import assert_equal
import os
import thunderfish.datawriter as dw


def test_formats():
    dw.available_formats()
    dw.data_modules['pickle'] == False
    dw.data_modules['numpy'] == False
    dw.data_modules['scipy'] == False
    dw.data_modules['audioio'] == False
    dw.available_formats()


def test_extensions():
    f = dw.format_from_extension(None)
    assert_equal(f, None)
    f = dw.format_from_extension('')
    assert_equal(f, None)
    f = dw.format_from_extension('test')
    assert_equal(f, None)
    f = dw.format_from_extension('test.')
    assert_equal(f, None)
    f = dw.format_from_extension('test.pkl')
    assert_equal(f, 'PICKLE')

    
def test_main():
    dw.main(['-c', '2', 'test.npz'])
    os.remove('test.npz')
    
