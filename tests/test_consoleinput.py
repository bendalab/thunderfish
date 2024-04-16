import pytest
import os
import sys
from io import StringIO
import thunderfish.consoleinput as ci


def test_read():
    ips = 'foo\nbar\n\n3\nx\n4\n5\n100\n-2\n9\n'
    sys.stdin = StringIO(ips)
    x = ci.read('test')
    assert x == 'foo', 'read() foo'
    x = ci.read('test', 'a', str)
    assert x == 'bar', 'read() bar'
    x = ci.read('test', 'a', str)
    assert x == 'a', 'read() default'
    x = ci.read('test', 2, int)
    assert x == 3, 'read() 3'
    x = ci.read('test', 2, int)
    assert x == 4, 'read() 4'
    x = ci.read('test', 2, int, 1, 10)
    assert x == 5, 'read() 5'
    x = ci.read('test', 2, int, 1, 10)
    assert x == 9, 'read() 9'
    ci.save_inputs('inputs.txt')
    os.remove('inputs.txt')
    ci.clear_inputs()


def test_select():
    ips = 'b\nb\n\nx\nc\n'
    sys.stdin = StringIO(ips)
    x = ci.select('test', None, ['a', 'b', 'c'], ['apple', 'banana', 'citrus'])
    assert x == 'b', 'select() b'
    x = ci.select('test', 'a', ['a', 'b', 'c'], ['apple', 'banana', 'citrus'])
    assert x == 'b', 'select() b'
    x = ci.select('test', 'a', ['a', 'b', 'c'], ['apple', 'banana', 'citrus'])
    assert x == 'a', 'select() default'
    x = ci.select('test', 'a', ['a', 'b', 'c'], ['apple', 'banana', 'citrus'])
    assert x == 'c', 'select() x c'


def test_main():
    ips = '3\nb\n'
    sys.stdin = StringIO(ips)
    ci.main()

