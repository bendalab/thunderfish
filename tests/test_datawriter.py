import pytest
import os
import numpy as np
import thunderfish.datawriter as dw


def test_formats():
    dw.available_formats()
    for fmt, lib, formats_func in dw.data_formats_funcs:
        if lib:
            dw.data_modules[lib] = False
            f = formats_func()
            assert len(f) == 0, f'{lib} format did not return empty list'
            dw.data_modules[lib] = True


def test_write():
    with pytest.raises(ValueError):
        dw.write_data('', np.zeros((1000, 2)), 48000)
    with pytest.raises(ValueError):
        dw.write_data('test', np.zeros((1000, 2)), 48000)
    with pytest.raises(IOError):
        dw.write_data('test', np.zeros((1000, 2)), 48000, format='xxx')
    dw.data_modules['pkl'] = False
    with pytest.raises(IOError):
        dw.write_data('test', np.zeros((1000, 2)), 48000, format='xxx')
    dw.data_modules['pkl'] = True
    for fmt, lib, formats_func in dw.data_formats_funcs:
        writer_func = dw.data_writer_funcs[fmt]
        with pytest.raises(ValueError):
            writer_func('', np.zeros((1000, 2)), 48000)
        if lib:
            dw.data_modules[lib] = False
            with pytest.raises(ImportError):
                writer_func('test.dat', np.zeros((1000, 2)), 48000)
            dw.data_modules[lib] = True

    
def test_extensions():
    f = dw.format_from_extension(None)
    assert f is None
    f = dw.format_from_extension('')
    assert f is None
    f = dw.format_from_extension('test')
    assert f is None
    f = dw.format_from_extension('test.')
    assert f is None
    f = dw.format_from_extension('test.pkl')
    assert f == 'PKL'

    
def test_main():
    dw.main('-c', '2', 'test.npz')
    os.remove('test.npz')
    
