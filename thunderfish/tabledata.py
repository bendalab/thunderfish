"""
Tables with hierarchical headers and units

## Classes

- `class TableData`: tables with a rich hierarchical header
  including units and column-specific formats. Kind of similar to a
  pandas data frame, but with intuitive numpy-style indexing and nicely
  formatted output to csv, html, and latex.


## Helper functions

- `write()`: shortcut for constructing and writing a TableData.
- `latex_unit()`: translate unit string into SIunit LaTeX code.
- `index2aa()`: convert an integer into an alphabetical representation.
- `aa2index()`: convert an alphabetical representation to an index.


## Configuration

- `add_write_table_config()`: add parameter specifying how to write a table to a file as a new section to a configuration.
- `write_table_args()`: translates a configuration to the respective parameter names for writing a table to a file.

"""

import sys
import os
import re
import math as m
import numpy as np
if sys.version_info[0] < 3:
    from io import BytesIO as StringIO
else:
    from io import StringIO
try:
    import pandas as pd
except ImportError:
    pass


__pdoc__ = {}
__pdoc__['TableData.__contains__'] = True
__pdoc__['TableData.__len__'] = True
__pdoc__['TableData.__iter__'] = True
__pdoc__['TableData.__next__'] = True
__pdoc__['TableData.__setupkey__'] = True
__pdoc__['TableData.__call__'] = True
__pdoc__['TableData.__getitem__'] = True
__pdoc__['TableData.__setitem__'] = True
__pdoc__['TableData.__delitem__'] = True
__pdoc__['TableData.__str__'] = True


class TableData(object):
    """
    Table with numpy-style indexing and a rich hierarchical header including units and formats.
    
    Parameters
    ----------
    data: string, stream, array
        - a filename: load table from file with name `data`.
        - a stream/file handle: load table from that stream.
        - 1-D or 2-D array of data: the data of the table.
          Requires als a specified `header`.
    header: list of string
        Header labels for each column.
    units: list of string, optional
        Unit strings for each column.
    formats: string or list of string, optional
        Format strings for each column. If only a single format string is
        given, then all columns are initialized with this format string.
    missing: string
        Missing data are indicated by this string.

    Manipulate table header
    -----------------------

    Each column of the table has a label (the name of the column), a
    unit, and a format specifier. Sections group columns into a hierarchy.

    - `__init__()`: initialize a TableData from data or a file.
    - `append()`: append column to the table.
    - `insert()`: insert a table column at a given position.
    - `remove()`: remove columns from the table.
    - `section()`: the section name of a specified column.
    - `set_section()`: set a section name.
    - `append_section()`: add sections to the table header.
    - `insert_section()`: insert a section at a given position of the table header.
    - `label()`: the name of a column.
    - `set_label()`: set the name of a column.
    - `unit()`: the unit of a column.
    - `set_unit()`: set the unit of a column.
    - `set_units()`: set the units of all columns.
    - `format()`: the format string of the column.
    - `set_format()`: set the format string of a column.
    - `set_formats()`: set the format strings of all columns.

    For example:
    ```
    tf = TableData('data.csv')
    ```
    loads a table directly from a file. See `load()` for details.
    ```
    tf = TableData(np.random.randn(4,3), header=['aaa', 'bbb', 'ccc'], units=['m', 's', 'g'], formats='%.2f')    
    ```
    results in
    ``` plain
    aaa    bbb    ccc
    m      s      g    
     1.45   0.01   0.16
    -0.74  -0.58  -1.34
    -2.06   0.08   1.47
    -0.43   0.60   1.38
    ```

    A more elaborate way to construct a table is:
    ```
    df = TableData()
    # first column with section names and 3 data values:
    df.append(["data", "partial information", "size"], "m", "%6.2f",
              [2.34, 56.7, 8.9])
    # next columns with single data values:
    df.append("full weight", "kg", "%.0f", 122.8)
    df.append_section("complete reaction")
    df.append("speed", "m/s", "%.3g", 98.7)
    df.append("median jitter", "mm", "%.1f", 23)
    df.append("size", "g", "%.2e", 1.234)
    # add a missing value to the second column:
    df.append_data(np.nan, 1)
    # fill up the remaining columns of the row:
    df.append_data((0.543, 45, 1.235e2))
    # append data to the next row starting at the second column:
    df.append_data((43.21, 6789.1, 3405, 1.235e-4), 1) # next row
    ```
    results in
    ``` plain
    data
    partial information  complete reaction
    size    full weight  speed     median jitter  size
    m       kg           m/s       mm             g       
      2.34          123      98.7           23.0  1.23e+00
     56.70            -     0.543           45.0  1.24e+02
      8.90           43  6.79e+03         3405.0  1.23e-04
    ```
    
    Table columns
    -------------

    Columns can be specified by an index or by the name of a column. In
    table headers with sections the colum can be specified by the
    section names and the column name separated by '>'.
    
    - `index()`: the index of a column.
    - `__contains__()`: check for existence of a column.
    - `find_col()`: find the start and end index of a column specification.
    - `column_spec()`: full specification of a column with all its section names.
    - `column_head()`: the name, unit, and format of a column.
    - `table_header()`: the header of the table without content.

    For example:
    ```
    df.index('complete reaction>size)   # returns 4
    'speed' in df                       # is True
    ```

    Iterating over columns
    ----------------------

    A table behaves like an ordered dictionary with column names as
    keys and the data of each column as values.
    Iterating over a table goes over columns.
    
    - `keys()`: list of unique column keys for all available columns.
    - `values()`: list of column data corresponding to keys().
    - `items()`: list of tuples with unique column specifications and the corresponding data.
    - `__len__()`: the number of columns.
    - `__iter__()`: initialize iteration over data columns.
    - `__next__()`: return data of next column as a list.
    - `data`: the table data as a list over columns each containing a list of data elements.

    For example:
    ```
    print('column specifications:')
    for c in range(df.columns()):
        print(df.column_spec(c))
    print('keys():')
    for c, k in enumerate(df.keys()):
        print('%d: %s' % (c, k))
    print('values():')
    for c, v in enumerate(df.values()):
        print(v)
    print('iterating over the table:')
    for v in df:
        print(v)
    ```
    results in
    ``` plain
    column specifications:
    data>partial information>size
    data>partial information>full weight
    data>complete reaction>speed
    data>complete reaction>median jitter
    data>complete reaction>size
    keys():
    0: data>partial information>size
    1: data>partial information>full weight
    2: data>complete reaction>speed
    3: data>complete reaction>median jitter
    4: data>complete reaction>size
    values():
    [2.34, 56.7, 8.9]
    [122.8, nan, 43.21]
    [98.7, 0.543, 6789.1]
    [23, 45, 3405]
    [1.234, 123.5, 0.0001235]
    iterating over the table:
    [2.34, 56.7, 8.9]
    [122.8, nan, 43.21]
    [98.7, 0.543, 6789.1]
    [23, 45, 3405]
    [1.234, 123.5, 0.0001235]
    ```

    Accessing data
    --------------

    In contrast to the iterator functions the [] operator treats the table as a
    2D-array where the first index indicates the row and the second index the column.

    Like a numpy aray the table can be sliced, and logical indexing can
    be used to select specific parts of the table.
    
    As for any function, columns can be specified as indices or strings.
    
    - `rows()`: the number of rows.
    - `columns()`: the number of columns.
    - `shape`: number of rows and columns.
    - `row()`: a single row of the table as TableData.
    - `row_dict()`: a single row of the table as dictionary.
    - `col()`: a single column of the table as TableData.
    - `__call__()`: a single column of the table as numpy array.
    - `__getitem__()`: data elements specified by slice.
    - `__setitem__()`: assign values to data elements specified by slice.
    - `__delitem__()`: delete data elements or whole columns or rows.
    - `array()`: the table data as a numpy array.
    - `data_frame()`: the table data as a pandas DataFrame.
    - `dicts()`: the table as a list of dictionaries.
    - `dict()`: the table as a dictionary.
    - `append_data()`: append data elements to successive columns.
    - `append_data_column()`: append data elements to a column.
    - `set_column()`: set the column where to add data.
    - `fill_data()`: fill up all columns with missing data.
    - `clear_data()`: clear content of the table but keep header.
    - `key_value()`: a data element returned as a key-value pair.
    
    - `sort()`: sort the table rows in place.
    - `statistics()`: descriptive statistics of each column.

    For example:
    ```
    # single column:    
    df('size')     # data of 'size' column as numpy array
    df[:,'size']   # data of 'size' column as numpy array
    df.col('size') # table with the single column 'size'

    # single row:    
    df[2,:]    # table with data of only the third row
    df.row(2)  # table with data of only the third row

    # slices:
    df[2:5,['size','jitter']]          # sub-table
    df[2:5,['size','jitter']].array()  # numpy array with data only

    # logical indexing:
    df[df('speed') > 100.0, 'size'] = 0.0 # set size to 0 if speed is > 100

    # delete:
    del df[3:6, 'weight']  # delete rows 3-6 from column 'weight'
    del df[3:5,:]          # delete rows 3-5 completeley
    del df[:,'speed']      # remove column 'speed' from table
    df.remove('weight')    # remove column 'weigth' from table

    # sort and statistics:
    df.sort(['weight', 'jitter'])
    df.statistics()
    ```
    statistics() returns a table with standard descriptive statistics:
    ``` plain
    statistics  data
    -           partial information  complete reaction
    -           size    full weight  speed     median jitter  size
    -           m       kg           m/s       mm             g       
    mean         22.65           83   2.3e+03         1157.7  4.16e+01
    std          24.23           40  3.18e+03         1589.1  5.79e+01
    min           2.34           43     0.543           23.0  1.23e-04
    quartile1     5.62           83      49.6           34.0  6.17e-01
    median        8.90          123      98.7           45.0  1.23e+00
    quartile3    32.80            -  3.44e+03         1725.0  6.24e+01
    max          56.70          123  6.79e+03         3405.0  1.24e+02
    count         3.00            2         3            3.0  3.00e+00
    ```

    Write and load tables
    ---------------------

    Table data can be written to a variety of text-based formats
    including comma separated values, latex and html files.  Which
    columns are written can be controlled by the hide() and show()
    functions. TableData can be loaded from all the written file formats
    (except html), also directly via the constructor.
    
    - `hide()`: hide a column or a range of columns.
    - `hide_all()`: hide all columns.
    - `hide_empty_columns()`: hide all columns that do not contain data.
    - `show()`: show a column or a range of columns.
    - `write()`: write the table to a file or stream.
    - `__str__()`: write table to a string.
    - `load()`: load table from file or stream.
    - `formats`: list of supported file formats for writing.
    - `descriptions`: dictionary with descriptions of the supported file formats.
    - `extensions`: dictionary with default filename extensions for each of the file formats.
    - `ext_formats`: dictionary mapping filename extensions to file formats.

    See documentation of the `write()` function for examples of the supported file formats.
    """
    
    formats = ['dat', 'ascii', 'csv', 'rtai', 'md', 'tex', 'html']
    """ list of strings: Supported output formats."""
    descriptions = {'dat': 'data text file', 'ascii': 'ascii-art table',
                    'csv': 'comma separated values', 'rtai': 'rtai-style table',
                    'md': 'markdown', 'tex': 'latex tabular',
                    'html': 'html markup'}
    """ dict: Decription of output formats corresponding to `formats`."""
    extensions = {'dat': 'dat', 'ascii': 'txt', 'csv': 'csv', 'rtai': 'dat',
                  'md': 'md', 'tex': 'tex', 'html': 'html'}
    """ dict: Default file extensions for the output `formats`. """
    ext_formats = {'dat': 'dat', 'DAT': 'dat', 'txt': 'dat', 'TXT': 'dat',
                   'csv': 'csv', 'CSV': 'csv', 'md': 'md', 'MD': 'md',
                   'tex': 'tex', 'TEX': 'tex', 'html': 'html', 'HTML': 'html'}
    """ dict: Mapping of file extensions to the output formats. """

    def __init__(self, data=None, header=None, units=None, formats=None,
                 missing='-'):
        self.data = []
        self.shape = (0, 0)
        self.header = []
        self.nsecs = 0
        self.units = []
        self.formats = []
        self.hidden = []
        self.setcol = 0
        self.addcol = 0
        if header is not None:
            if units is None:
                units = ['']*len(header)
            if formats is None:
                formats = ['%g']*len(header)
            elif not isinstance(formats, (list, tuple, np.ndarray)):
                formats = [formats]*len(header)
            for h, u, f in zip(header, units, formats):
                self.append(h, u, f)            
        if data is not None:
            if isinstance(data, TableData):
                self.shape = data.shape
                self.nsecs = data.nsecs
                self.setcol = data.setcol
                self.addcol = data.addcol
                for c in range(data.columns()):
                    self.header.append([])
                    for h in data.header[c]:
                        self.header[c].append(h)
                    self.units.append(data.units[c])
                    self.formats.append(data.formats[c])
                    self.hidden.append(data.hidden[c])
                    self.data.append([])
                    for d in data.data[c]:
                        self.data[c].append(d)
            elif isinstance(data, (list, tuple, np.ndarray)):
                if isinstance(data[0], (list, tuple, np.ndarray)):
                    # 2D list, rows first:
                    for row in data:
                        for c, val in enumerate(row):
                            self.data[c].append(val)
                else:
                    # 1D list:
                    for c, val in enumerate(data):
                        self.data[c].append(val)
            else:
                self.load(data, missing)
        
    def append(self, label, unit=None, formats=None, value=None, key=None, fac=None):
        """
        Append column to the table.

        Parameters
        ----------
        label: string or list of string
            Optional section titles and the name of the column.
        unit: string or None
            The unit of the column contents.
        formats: string or None
            The C-style format string used for printing out the column content, e.g.
            '%g', '%.2f', '%s', etc.
            If None, the format is set to '%g'.
        value: None, float, int, string, etc. or list thereof, or list of dict
            If not None, data for the column.
        key: None or key of a dictionary
            If not None and `value` is a list of dictionaries,
            extract from each dictionary in the list the value specified
            by `key` and assign the resulting list as data to the column.
        fac: float
            If not None, multiply the data values by this number.

        Returns
        -------
        index: int
            The index of the new column.
        """
        if self.addcol >= len(self.data):
            if isinstance(label, (list, tuple, np.ndarray)):
                self.header.append(list(reversed(label)))
            else:
                self.header.append([label])
            self.formats.append(formats or '%g')
            self.units.append(unit or '')
            self.hidden.append(False)
            self.data.append([])
            if self.nsecs < len(self.header[-1])-1:
                self.nsecs = len(self.header[-1])-1
        else:
            if isinstance(label, (list, tuple, np.ndarray)):
                self.header[self.addcol] = list(reversed(label)) + self.header[self.addcol]
            else:
                self.header[self.addcol] = [label] + self.header[self.addcol]
            self.units[self.addcol] = unit or ''
            self.formats[self.addcol] = formats or '%g'
            if self.nsecs < len(self.header[self.addcol])-1:
                self.nsecs = len(self.header[self.addcol])-1
        if value is not None:
            if isinstance(value, (list, tuple, np.ndarray)):
                if key and value and isinstance(value[0], dict):
                    value = [d[key] if key in d else float('nan') for d in value]
                self.data[-1].extend(value)
            else:
                self.data[-1].append(value)
        if fac:
            for k in range(len(self.data[-1])):
                self.data[-1][k] *= fac
        self.addcol = len(self.data)
        self.shape = (self.rows(), self.columns())
        return self.addcol-1
        
    def insert(self, column, label, unit=None, formats=None, value=None):
        """
        Insert a table column at a given position.

        .. WARNING::
           If no `value` is given, the inserted column is an empty list.

        Parameters
        ----------
        columns int or string
            Column before which to insert the new column.
            Column can be specified by index or name,
            see `index()` for details.
        label: string or list of string
            Optional section titles and the name of the column.
        unit: string or None
            The unit of the column contents.
        formats: string or None
            The C-style format string used for printing out the column content, e.g.
            '%g', '%.2f', '%s', etc.
            If None, the format is set to '%g'.
        value: None, float, int, string, etc. or list thereof
            If not None, data for the column.

        Returns
        -------
        index: int
            The index of the inserted column.
            
        Raises
        ------
        IndexError:
            If an invalid column was specified.
        """
        col = self.index(column)
        if col is None:
            if isinstance(column, (int, np.integer)):
                column = '%d' % column
            raise IndexError('Cannot insert before non-existing column ' + column)
        if isinstance(label, (list, tuple, np.ndarray)):
            self.header.insert(col, list(reversed(label)))
        else:
            self.header.insert(col, [label])
        self.formats.insert(col, formats or '%g')
        self.units.insert(col, unit or '')
        self.hidden.insert(col, False)
        self.data.insert(col, [])
        if self.nsecs < len(self.header[col])-1:
            self.nsecs = len(self.header[col])-1
        if value is not None:
            if isinstance(value, (list, tuple, np.ndarray)):
                self.data[col].extend(value)
            else:
                self.data[col].append(value)
        self.addcol = len(self.data)
        self.shape = (self.rows(), self.columns())
        return col

    def remove(self, columns):
        """
        Remove columns from the table.

        Parameters
        -----------
        columns: int or string or list of int or string
            Columns can be specified by index or name,
            see `index()` for details.

        Raises
        ------
        IndexError:
            If an invalid column was specified.
        """
        # fix columns:
        if not isinstance(columns, (list, tuple, np.ndarray)):
            columns = [ columns ]
        if not columns:
            return
        # remove:
        for col in columns:
            c = self.index(col)
            if c is None:
                if isinstance(col, (int, np.integer)):
                    col = '%d' % col
                raise IndexError('Cannot remove non-existing column ' + col)
                continue
            if c+1 < len(self.header):
                self.header[c+1].extend(self.header[c][len(self.header[c+1]):])
            del self.header[c]
            del self.units[c]
            del self.formats[c]
            del self.hidden[c]
            del self.data[c]
        if self.setcol > len(self.data):
            self.setcol = len(self.data)
        self.shape = (self.rows(), self.columns())

    def section(self, column, level):
        """
        The section name of a specified column.

        Parameters
        ----------
        column: None, int, or string
            A specification of a column.
            See self.index() for more information on how to specify a column.
        level: int
            The level of the section to be returned. The column label itself is level=0.

        Returns
        -------
        name: string
            The name of the section at the specified level containing the column.
        index: int
            The column index that contains this section (equal or smaller thant `column`).

        Raises
        ------
        IndexError:
            If `level` exceeds the maximum possible level.
        """
        if level < 0 or level > self.nsecs:
            raise IndexError('Invalid section level')
        column = self.index(column)
        while len(self.header[column]) <= level:
            column -= 1
        return self.header[column][level], column
    
    def set_section(self, label, column, level):
        """
        Set a section name.

        Parameters
        ----------
        label: string
            The new name to be used for the section.
        column: None, int, or string
            A specification of a column.
            See self.index() for more information on how to specify a column.
        level: int
            The level of the section to be set. The column label itself is level=0.
        """
        column = self.index(column)
        self.header[column][level] = label
        return column

    def append_section(self, label):
        """
        Add sections to the table header.

        Each column of the table has a header label. Columns can be
        grouped into sections. Sections can be nested arbitrarily.

        Parameters
        ----------
        label: string or list of string
            The name(s) of the section(s).

        Returns
        -------
        index: int
            The column index where the section was appended.
        """
        if self.addcol >= len(self.data):
            if isinstance(label, (list, tuple, np.ndarray)):
                self.header.append(list(reversed(label)))
            else:
                self.header.append([label])
            self.units.append('')
            self.formats.append('')
            self.hidden.append(False)
            self.data.append([])
        else:
            if isinstance(label, (list, tuple, np.ndarray)):
                self.header[self.addcol] = list(reversed(label)) + self.header[self.addcol]
            else:
                self.header[self.addcol] = [label] + self.header[self.addcol]
        if self.nsecs < len(self.header[self.addcol]):
            self.nsecs = len(self.header[self.addcol])
        self.addcol = len(self.data)-1
        self.shape = (self.rows(), self.columns())
        return self.addcol
        
    def insert_section(self, column, section):
        """
        Insert a section at a given position of the table header.

        Parameters
        ----------
        columns int or string
            Column before which to insert the new section.
            Column can be specified by index or name,
            see `index()` for details.
        section: string
            The name of the section.

        Returns
        -------
        index: int
            The index of the column where the section was inserted.
            
        Raises
        ------
        IndexError:
            If an invalid column was specified.
        """
        col = self.index(column)
        if col is None:
            if isinstance(column, (int, np.integer)):
                column = '%d' % column
            raise IndexError('Cannot insert at non-existing column ' + column)
        self.header[col].append(section)
        if self.nsecs < len(self.header[col])-1:
            self.nsecs = len(self.header[col])-1
        return col

    def label(self, column):
        """
        The name of a column.

        Parameters
        ----------
        column: None, int, or string
            A specification of a column.
            See self.index() for more information on how to specify a column.

        Returns
        -------
        name: string
            The column label.
        """
        column = self.index(column)
        return self.header[column][0]

    def set_label(self, label, column):
        """
        Set the name of a column.

        Parameters
        ----------
        label: string
            The new name to be used for the column.
        column: None, int, or string
            A specification of a column.
            See self.index() for more information on how to specify a column.
        """        
        column = self.index(column)
        self.header[column][0] = label
        return column

    def unit(self, column):
        """
        The unit of a column.

        Parameters
        ----------
        column: None, int, or string
            A specification of a column.
            See self.index() for more information on how to specify a column.

        Returns
        -------
        unit: string
            The unit.
        """
        column = self.index(column)
        return self.units[column]

    def set_unit(self, unit, column):
        """
        Set the unit of a column.

        Parameters
        ----------
        unit: string
            The new unit to be used for the column.
        column: None, int, or string
            A specification of a column.
            See self.index() for more information on how to specify a column.
        """
        column = self.index(column)
        self.units[column] = unit
        return column

    def set_units(self, units):
        """
        Set the units of all columns.

        Parameters
        ----------
        units: list of string
            The new units to be used.
        """
        for c, u in enumerate(units):
            self.units[c] = u

    def format(self, column):
        """
        The format string of the column.

        Parameters
        ----------
        column: None, int, or string
            A specification of a column.
            See self.index() for more information on how to specify a column.

        Returns
        -------
        format: string
            The format string.
        """
        column = self.index(column)
        return self.formats[column]

    def set_format(self, format, column):
        """
        Set the format string of a column.

        Parameters
        ----------
        format: string
            The new format string to be used for the column.
        column: None, int, or string
            A specification of a column.
            See self.index() for more information on how to specify a column.
        """
        column = self.index(column)
        self.formats[column] = format
        return column

    def set_formats(self, formats):
        """
        Set the format strings of all columns.

        Parameters
        ----------
        formats: string or list of string
            The new format strings to be used.
            If only a single format is specified,
            then all columns get the same format.
        """
        if isinstance(formats, (list, tuple, np.ndarray)):
            for c, f in enumerate(formats):
                self.formats[c] = f or '%g'
        else:
            for c in range(len(self.formats)):
                self.formats[c] = formats or '%g'

    def table_header(self):
        """
        The header of the table without content.

        Return
        ------
        data: TableData
            A TableData object with the same header but empty data.
        """
        data = TableData()
        sec_indices = [-1] * self.nsecs
        for c in range(self.columns()):
            data.append(*self.column_head(c))
            for l in range(self.nsecs):
                s, i = self.section(c, l+1)
                if i != sec_indices[l]:
                    data.header[-1].append(s)
                    sec_indices[l] = i
        data.nsecs = self.nsecs
        return data

    def column_head(self, column):
        """
        The name, unit, and format of a column.

        Parameters
        ----------
        column: None, int, or string
            A specification of a column.
            See self.index() for more information on how to specify a column.

        Returns
        -------
        name: string
            The column label.
        unit: string
            The unit.
        format: string
            The format string.
        """
        column = self.index(column)
        return self.header[column][0], self.units[column], self.formats[column]

    def column_spec(self, column):
        """
        Full specification of a column with all its section names.

        Parameters
        ----------
        column: int or string
            Specifies the column.
            See self.index() for more information on how to specify a column.

        Returns
        -------
        s: string
            Full specification of the column by all its section names and its header name.
        """
        c = self.index(column)
        fh = [self.header[c][0]]
        for l in range(self.nsecs):
            fh.append(self.section(c, l+1)[0])
        return '>'.join(reversed(fh))
    
    def find_col(self, column):
        """
        Find the start and end index of a column specification.
        
        Parameters
        ----------
        column: None, int, or string
            A specification of a column.
            See self.index() for more information on how to specify a column.

        Returns
        -------
        c0: int or None
            A valid column index or None that is specified by `column`.
        c1: int or None
            A valid column index or None of the column following the range specified
            by `column`.
        """

        def find_column_indices(ss, si, minns, maxns, c0, strict=True):
            if si >= len(ss):
                return None, None, None, None
            ns0 = 0
            for ns in range(minns, maxns+1):
                nsec = maxns-ns
                if ss[si] == '':
                    si += 1
                    continue
                for c in range(c0, len(self.header)):
                    if nsec < len(self.header[c]) and \
                        ( ( strict and self.header[c][nsec] == ss[si] ) or
                          ( not strict and ss[si] in self.header[c][nsec] ) ):
                        ns0 = ns
                        c0 = c
                        si += 1
                        if si >= len(ss):
                            c1 = len(self.header)
                            for c in range(c0+1, len(self.header)):
                                if nsec < len(self.header[c]):
                                    c1 = c
                                    break
                            return c0, c1, ns0, None
                        elif nsec > 0:
                            break
            return None, c0, ns0, si

        if column is None:
            return None, None
        if not isinstance(column, (int, np.integer)) and column.isdigit():
            column = int(column)
        if isinstance(column, (int, np.integer)):
            if column >= 0 and column < len(self.header):
                return column, column+1
            else:
                return None, None
        # find column by header:
        ss = column.rstrip('>').split('>')
        maxns = self.nsecs
        si0 = 0
        while si0 < len(ss) and ss[si0] == '':
            maxns -= 1
            si0 += 1
        if maxns < 0:
            maxns = 0
        c0, c1, ns, si = find_column_indices(ss, si0, 0, maxns, 0, True)
        if c0 is None and c1 is not None:
            c0, c1, ns, si = find_column_indices(ss, si, ns, maxns, c1, False)
        return c0, c1

    def index(self, column):
        """
        The index of a column.
        
        Parameters
        ----------
        column: None, int, or string
            A specification of a column.
            - None: no column is specified
            - int: the index of the column (first column is zero), e.g. `index(2)`.
            - a string representing an integer is converted into the column index,
              e.g. `index('2')`
            - a string specifying a column by its header.
              Header names of descending hierarchy are separated by '>'.

        Returns
        -------
        index: int or None
            A valid column index or None.
        """
        c0, c1 = self.find_col(column)
        return c0

    def __contains__(self, column):
        """
        Check for existence of a column. 

        Parameters
        ----------
        column: None, int, or string
            The column to be checked.
            See self.index() for more information on how to specify a column.

        Returns
        -------
        contains: bool
            True if `column` specifies an existing column key.
        """
        return self.index(column) is not None

    def keys(self):
        """
        List of unique column keys for all available columns.

        Returns
        -------
        keys: list of strings
            List of unique column specifications.
        """
        return [self.column_spec(c) for c in range(self.columns())]

    def values(self):
        """
        List of column data corresponding to keys().

        Returns
        -------
        data: list of list of values
            The data of the table. First index is columns!
        """
        return self.data

    def items(self):
        """
        Column names and corresponding data.

        Returns
        -------
        items: list of tuples
            Unique column specifications and the corresponding data.
        """
        return [(self.column_spec(c), self.data[c]) for c in range(self.columns())]
        
    def __len__(self):
        """
        The number of columns.
        
        Returns
        -------
        columns: int
            The number of columns contained in the table.
        """
        return self.columns()

    def __iter__(self):
        """
        Initialize iteration over data columns.
        """
        self.iter_counter = -1
        return self

    def __next__(self):
        """
        Next column of data.

        Returns
        -------
        data: list of values
            Table data of next column.
        """
        self.iter_counter += 1
        if self.iter_counter >= self.columns():
            raise StopIteration
        else:
            return self.data[self.iter_counter]

    def next(self):
        """
        Return next data columns.
        (python2 syntax)

        See also:
        ---------
        `__next__()`
        """
        return self.__next__()

    def rows(self):
        """
        The number of rows.
        
        Returns
        -------
        rows: int
            The number of rows contained in the table.
        """
        return max(map(len, self.data)) if self.data else 0
    
    def columns(self):
        """
        The number of columns.
        
        Returns
        -------
        columns: int
            The number of columns contained in the table.
        """
        return len(self.header)

    def row(self, index):
        """
        A single row of the table.

        Parameters
        ----------
        index: int
            The index of the row to be returned.

        Return
        ------
        data: TableData
            A TableData object with a single row.
        """
        data = TableData()
        sec_indices = [-1] * self.nsecs
        for c in range(self.columns()):
            data.append(*self.column_head(c))
            for l in range(self.nsecs):
                s, i = self.section(c, l+1)
                if i != sec_indices[l]:
                    data.header[-1].append(s)
                    sec_indices[l] = i
            data.data[-1] = [self.data[c][index]]
        data.nsecs = self.nsecs
        return data

    def row_dict(self, index):
        """
        A single row of the table.

        Parameters
        ----------
        index: int
            The index of the row to be returned.

        Return
        ------
        data: dict
            A dictionary with column header as key and corresponding data value of row `index`
            as value.
        """
        data = {}
        for c in range(self.columns()):
            data[self.label(c)] = self.data[c][index]
        return data

    def col(self, column):
        """
        A single column of the table.

        Parameters
        ----------
        column: None, int, or string
            The column to be returned.
            See self.index() for more information on how to specify a column.

        Return
        ------
        table: TableData
            A TableData object with a single column.
        """
        data = TableData()
        c = self.index(column)
        data.append(*self.column_head(c))
        data.data = [self.data[c]]
        data.nsecs = 0
        return data

    def __call__(self, column):
        """
        A single column of the table as a numpy array.

        Parameters
        ----------
        column: None, int, or string
            The column to be returned.
            See self.index() for more information on how to specify a column.

        Return
        ------
        data: 1-D array
            Content of the specified column as a numpy array.
        """
        c = self.index(column)
        return np.asarray(self.data[c])

    def __setupkey(self, key):
        """
        Helper function that turns a key into row and column indices.

        Returns
        -------
        rows: list of int, slice, None
            Indices of selected rows.
        cols: list of int
            Indices of selected columns.

        Raises
        ------
        IndexError:
            If an invalid column was specified.
        """
        if type(key) is not tuple:
            rows = key
            cols = range(self.columns())
        else:
            rows = key[0]
            cols = key[1]
        if isinstance(cols, slice):
            start = cols.start
            if start is not None:
                start = self.index(start)
                if start is None:
                    raise IndexError('"%s" is not a valid column index' % cols.start)
            stop = cols.stop
            if stop is not None:
                stop = self.index(stop)
                if stop is None:
                    raise IndexError('"%s" is not a valid column index' % cols.stop)
            cols = slice(start, stop, cols.step)
            cols = range(self.columns())[cols]
        else:
            if not isinstance(cols, (list, tuple, np.ndarray)):
                cols = [cols]
            c = [self.index(inx) for inx in cols]
            if None in c:
                raise IndexError('"%s" is not a valid column index' % cols[c.index(None)])
            cols = c
        if isinstance(rows, np.ndarray) and rows.dtype == np.dtype(bool):
            rows = np.where(rows)[0]
            if len(rows) == 0:
                rows = None
        return rows, cols

    def __getitem__(self, key):
        """
        Data elements specified by slice.

        Parameters
        -----------
        key:
            First key specifies row, (optional) second one the column.
            Columns can be specified by index or name,
            see `index()` for details.

        Returns
        -------
        data:
            - A single data value if a single row and a single column is specified.
            - An array of data elements if a single single column is specified.
            - A TableData object for multiple columns.
            - None if no row is selected (e.g. by a logical index that nowhere is True)

        Raises
        ------
        IndexError:
            If an invalid column was specified.
        """
        rows, cols = self.__setupkey(key)
        if len(cols) == 1:
            if rows is None:
                return None
            elif isinstance(rows, slice):
                return np.asarray(self.data[cols[0]][rows])
            elif isinstance(rows, (list, tuple, np.ndarray)):
                return np.asarray([self.data[cols[0]][r] for r in rows])
            else:
                return self.data[cols[0]][rows]
        else:
            data = TableData()
            sec_indices = [-1] * self.nsecs
            for c in cols:
                data.append(*self.column_head(c))
                for l in range(self.nsecs):
                    s, i = self.section(c, l+1)
                    if i != sec_indices[l]:
                        data.header[-1].append(s)
                        sec_indices[l] = i
                if rows is None:
                    continue
                if isinstance(rows, (list, tuple, np.ndarray)):
                    for r in rows:
                        data.data[-1].append(self.data[c][r])
                else:
                    if isinstance(self.data[c][rows], (list, tuple, np.ndarray)):
                        data.data[-1].extend(self.data[c][rows])
                    else:
                        data.data[-1].append(self.data[c][rows])
            data.nsecs = self.nsecs
            return data

    def __setitem__(self, key, value):
        """
        Assign values to data elements specified by slice.

        Parameters
        -----------
        key:
            First key specifies row, (optional) second one the column.
            Columns can be specified by index or name,
            see `index()` for details.
        value: TableData, list, ndarray, float, ...
            Value(s) used to assing to the table elements as specified by `key`.

        Raises
        ------
        IndexError:
            If an invalid column was specified.
        """
        rows, cols = self.__setupkey(key)
        if rows is None:
            return
        if isinstance(value, TableData):
            if isinstance(self.data[cols[0]][rows], (list, tuple, np.ndarray)):
                for k, c in enumerate(cols):
                    self.data[c][rows] = value.data[k]
            else:
                for k, c in enumerate(cols):
                    self.data[c][rows] = value.data[k][0]
        else:
            if len(cols) == 1:
                if isinstance(rows, (list, tuple, np.ndarray)):
                    if len(rows) == 1:
                        self.data[cols[0]][rows[0]] = value
                    elif isinstance(value, (list, tuple, np.ndarray)):
                        for k, r in enumerate(rows):
                            self.data[cols[0]][r] = value[k]
                    else:
                        for r in rows:
                            self.data[cols[0]][r] = value
                elif isinstance(value, (list, tuple, np.ndarray)):
                    self.data[cols[0]][rows] = value
                elif isinstance(rows, (int, np.integer)):
                    self.data[cols[0]][rows] = value
                else:
                    n = len(self.data[cols[0]][rows])
                    if n > 1:
                        self.data[cols[0]][rows] = [value]*n
                    else:
                        self.data[cols[0]][rows] = value
            else:
                if isinstance(self.data[0][rows], (list, tuple, np.ndarray)):
                    for k, c in enumerate(cols):
                        self.data[c][rows] = value[:,k]
                elif isinstance(value, (list, tuple, np.ndarray)):
                    for k, c in enumerate(cols):
                        self.data[c][rows] = value[k]
                else:
                    for k, c in enumerate(cols):
                        self.data[c][rows] = value

    def __delitem__(self, key):
        """
        Delete data elements or whole columns or rows.

        Parameters
        -----------
        key:
            First key specifies row, (optional) second one the column.
            Columns can be specified by index or name,
            see `index()` for details.
            If all rows are selected, then the specified columns are removed from the table.
            Otherwise only data values are removed.
            If all columns are selected than entire rows of data values are removed.
            Otherwise only data values in the specified rows are removed.

        Raises
        ------
        IndexError:
            If an invalid column was specified.
        """
        rows, cols = self.__setupkey(key)
        if rows is None:
            return
        row_indices = np.arange(self.rows(), dtype=np.int)[rows]
        if isinstance(row_indices, np.ndarray):
            if len(row_indices) == self.rows():
                # delete whole columns:
                self.remove(cols)
            elif len(row_indices) > 0:
                for r in reversed(sorted(row_indices)):
                    for c in cols:
                        del self.data[c][r]
                self.shape = (self.rows(), self.columns())
        else:
            for c in cols:
                del self.data[c][row_indices]
            self.shape = (self.rows(), self.columns())

    def array(self, row=None):
        """
        The table data as a numpy array.

        Parameter
        ---------
        row: int or None
            If specified, a 1D array of that row will be returned.

        Return
        ------
        data: 2D or 1D ndarray
            If no row is specified, the data content of the entire table
            as a 2D numpy array (rows first).
            If a row is specified, a 1D array of that row.
        """
        if row is None:
            return np.array(self.data).T
        else:
            return np.array([d[row] for d in self.data])

    def data_frame(self):
        """
        The table data as a pandas DataFrame.

        Return
        ------
        data: pandas.DataFrame
            A pandas DataFrame of the whole table.
        """
        return pd.DataFrame(self.dict())

    def dicts(self, raw_values=True, missing='-'):
        """
        The table as a list of dictionaries.

        Parameters
        ----------
        raw_values: bool
            If True, use raw table values as values,
            else format the values and add unit string.
        missing: string
            String indicating non-existing data elements.

        Returns
        -------
        table: list of dict
            For each row of the table a dictionary with header as key.
        """
        table = []
        for row in range(self.rows()):
            data = {}
            for col in range(len(self.header)):
                if raw_values:
                    v = self.data[col][row];
                else:
                    if isinstance(self.data[col][row], float) and m.isnan(self.data[col][row]):
                        v = missing
                    else:
                        u = ''
                        if not self.units[col] in '1-' and self.units[col] != 'a.u.':
                            u = self.units[col]
                        v = (self.formats[col] % self.data[col][row]) + u
                data[self.header[col][0]] = v
            table.append(data)
        return table

    def dict(self):
        """
        The table as a dictionary.

        Returns
        -------
        table: dict
            A dictionary with keys being the column headers and
            values the list of data elements of the corresponding column.
        """
        table = {k: v for k, v in self.items()}
        return table

    def append_data(self, data, column=None):
        """
        Append data elements to successive columns.

        The current column is set behid the added columns.

        Parameters
        ----------
        data: float, int, string, etc. or list thereof or list of list thereof
            Data values to be appended to successive column.
            - A single value is simply appened to the specified column of the table.
            - A 1D-list of values is appended to successive columns of the table
              starting with the specified column.
            - The columns of a 2D-list of values (second index) are appended
              to successive columns of the table starting with the specified column.
        column: None, int, or string
            The first column to which the data should be appended.
            If None, append to the current column.
            See self.index() for more information on how to specify a column.
        """
        column = self.index(column)
        if column is None:
            column = self.setcol
        if isinstance(data, (list, tuple, np.ndarray)):
            if isinstance(data[0], (list, tuple, np.ndarray)):
                # 2D list, rows first:
                for row in data:
                    for i, val in enumerate(row):
                        self.data[column+i].append(val)
                self.setcol = column + len(data[0])
            else:
                # 1D list:
                for val in data:
                    self.data[column].append(val)
                    column += 1
                self.setcol = column
        else:
            # single value:
            self.data[column].append(data)
            self.setcol = column+1
        self.shape = (self.rows(), self.columns())

    def append_data_column(self, data, column=None):
        """
        Append data elements to a column.

        The current column is incremented by one.

        Parameters
        ----------
        data: float, int, string, etc. or list thereof
            Data values to be appended to a column.
        column: None, int, or string
            The column to which the data should be appended.
            If None, append to the current column.
            See self.index() for more information on how to specify a column.
        """
        column = self.index(column)
        if column is None:
            column = self.setcol
        if isinstance(data, (list, tuple, np.ndarray)):
            self.data[column].extend(data)
            column += 1
            self.setcol = column
        else:
            self.data[column].append(data)
            self.setcol = column+1
        self.shape = (self.rows(), self.columns())

    def set_column(self, column):
        """
        Set the column where to add data.

        Parameters
        ----------
        column: int or string
            The column to which data elements should be appended.
            See self.index() for more information on how to specify a column.

        Raises
        ------
        IndexError:
            If an invalid column was specified.
        """
        col = self.index(column)
        if col is None:
            if isinstance(column, (int, np.integer)):
                column = '%d' % column
            raise IndexError('column ' + column + ' not found or invalid')
        self.setcol = col
        return col

    def fill_data(self):
        """
        Fill up all columns with missing data to have the same number of data elements.
        """
        # maximum rows:
        maxr = self.rows()
        # fill up:
        for c in range(len(self.data)):
            while len(self.data[c]) < maxr:
                self.data[c].append(np.nan)
        self.setcol = 0
        self.shape = (self.rows(), self.columns())

    def clear_data(self):
        """
        Clear content of the table but keep header.
        """
        for c in range(len(self.data)):
            self.data[c] = []
        self.setcol = 0
        self.shape = (self.rows(), self.columns())
                
    def sort(self, columns, reverse=False):
        """
        Sort the table rows in place.

        Parameters
        ----------
        columns: int or string or list of int or string
            A column specifier or a list of column specifiers of the columns
            to be sorted.
        reverse: boolean
            If `True` sort in descending order.

        Raises
        ------
        IndexError:
            If an invalid column was specified.
        """
        # fix columns:
        if not isinstance(columns, (list, tuple, np.ndarray)):
            columns = [ columns ]
        if not columns:
            return
        cols = []
        for col in columns:
            c = self.index(col)
            if c is None:
                if isinstance(col, (int, np.integer)):
                    col = '%d' % col
                raise IndexError('sort column ' + col + ' not found')
                continue
            cols.append(c)
        # get sorted row indices:
        row_inx = range(self.rows())
        row_inx = sorted(row_inx, key=lambda x : [float('-inf') if self.data[c][x] is np.nan \
                         or self.data[c][x] != self.data[c][x] \
                         else self.data[c][x] for c in cols], reverse=reverse)
        # sort table according to indices:
        for c in range(self.columns()):
            self.data[c] = [self.data[c][r] for r in row_inx]

    def statistics(self):
        """
        Descriptive statistics of each column.
        """
        ds = TableData()
        if self.nsecs > 0:
            ds.append_section('statistics')
            for l in range(1,self.nsecs):
                ds.append_section('-')
            ds.append('-', '-', '%-10s')
        else:
            ds.append('statistics', '-', '%-10s')
        ds.append_data('mean', 0)
        ds.append_data('std', 0)
        ds.append_data('min', 0)
        ds.append_data('quartile1', 0)
        ds.append_data('median', 0)
        ds.append_data('quartile3', 0)
        ds.append_data('max', 0)
        ds.append_data('count', 0)
        dc = 1
        for c in range(self.columns()):
            if len(self.data[c]) > 0 and isinstance(self.data[c][0], (float, int)):
                ds.hidden.append(False)
                ds.header.append(self.header[c])
                ds.units.append(self.units[c])
                # integer data still make floating point statistics:
                if isinstance(self.data[c][0], float):
                    f = self.formats[c]
                    i0 = f.find('.')
                    if i0 > 0:
                        p = int(f[i0+1:-1])
                        if p <= 0:
                            f = '%.1f'
                    ds.formats.append(f)
                else:
                    ds.formats.append('%.1f')
                # remove nans:
                data = np.asarray(self.data[c], np.float)
                data = data[np.isfinite(data)]
                # compute statistics:
                ds.data.append([])
                ds.append_data(np.mean(data), dc)
                ds.append_data(np.std(data), dc)
                ds.append_data(np.min(data), dc)
                q1, m, q3 = np.percentile(data, [25., 50., 75.])
                ds.append_data(q1, dc)
                ds.append_data(m, dc)
                ds.append_data(q3, dc)
                ds.append_data(np.max(data), dc)
                ds.append_data(len(data), dc)
                dc += 1
        ds.nsecs = self.nsecs
        ds.shape = (ds.rows(), ds.columns())
        return ds

    def key_value(self, row, col, missing='-'):
        """
        A data element returned as a key-value pair.

        Parameters
        ----------
        row: int
            Specifies the row from which the data element should be retrieved.
        col: None, int, or string
            A specification of a column.
            See self.index() for more information on how to specify a column.
        missing: string
            String indicating non-existing data elements.

        Returns
        -------
        key: string
            Header label of the column
        value: string
            A textual representation of the data element according to the format
            of the column, followed by the unit of the column.
        """
        col = self.index(col)
        if col is None:
            return ''
        if isinstance(self.data[col][row], float) and m.isnan(self.data[col][row]):
            v = missing
        else:
            u = ''
            if not self.units[col] in '1-' and self.units[col] != 'a.u.':
                u = self.units[col]
            v = (self.formats[col] % self.data[col][row]) + u
        return self.header[col][0], v

    def hide(self, column):
        """
        Hide a column or a range of columns.

        Hidden columns will not be printed out by the write() function.

        Parameters
        ----------
        column: int or string
            The column to be hidden.
            See self.index() for more information on how to specify a column.
        """
        c0, c1 = self.find_col(column)
        if c0 is not None:
            for c in range(c0, c1):
                self.hidden[c] = True

    def hide_all(self):
        """
        Hide all columns.

        Hidden columns will not be printed out by the write() function.
        """
        for c in range(len(self.hidden)):
            self.hidden[c] = True

    def hide_empty_columns(self, missing='-'):
        """
        Hide all columns that do not contain data.

        Hidden columns will not be printed out by the write() function.

        Parameters
        ----------
        missing: string
            String indicating missing data.
        """
        for c in range(len(self.data)):
            # check for empty column:
            isempty = True
            for v in self.data[c]:
                if isinstance(v, float):
                    if not m.isnan(v):
                        isempty = False
                        break
                else:
                    if v != missing:
                        isempty = False
                        break
            if isempty:
                self.hidden[c] = True

    def show(self, column):
        """
        Show a column or a range of columns.

        Undoes hiding of a column.

        Parameters
        ----------
        column: int or string
            The column to be shown.
            See self.index() for more information on how to specify a column.
        """
        c0, c1 = self.find_col(column)
        if c0 is not None:
            for c in range(c0, c1):
                self.hidden[c] = False

    def write(self, fh=sys.stdout, table_format=None, delimiter=None,
              unit_style=None, column_numbers=None, sections=None,
              align_columns=None, shrink_width=True, missing='-',
              center_columns=False, latex_label_command='', latex_merge_std=False):
        """
        Write the table to a file or stream.

        Parameters
        ----------
        fh: filename or stream
            If not a stream, the file with name `fh` is opened.
            If `fh` does not have an extension,
            the `table_format` is appended as an extension.
            Otherwise `fh` is used as a stream for writing.
        table_format: None or string
            The format to be used for output.
            One of 'out', 'dat', 'ascii', 'csv', 'rtai', 'md', 'tex', 'html'.
            If None or 'auto' then the format is set to the extension of the filename given by `fh`.
            If `fh` is a stream the format is set to 'dat'.
        delimiter: string
            String or character separating columns, if supported by the `table_format`.
            If None or 'auto' use the default for the specified `table_format`.
        unit_style: None or string
            - None or 'auto': use default of the specified `table_format`.
            - 'row': write an extra row to the table header specifying the units of the columns.
            - 'header': add the units to the column headers.
            - 'none': do not specify the units.
        column_numbers: string or None
            Add a row specifying the column index:
            - 'index': indices are integers, first column is 0.
            - 'num': indices are integers, first column is 1.
            - 'aa': use 'a', 'b', 'c', ..., 'z', 'aa', 'ab', ... for indexing
            - 'aa': use 'A', 'B', 'C', ..., 'Z', 'AA', 'AB', ... for indexing
            - None or 'none': do not add a row with column indices
            TableData.column_numbering is a list with the supported styles.
        sections: None or int
            Number of section levels to be printed.
            If `None` or 'auto' use default of selected `table_format`.
        align_columns: boolean
            - `True`: set width of column formats to make them align.
            - `False`: set width of column formats to 0 - no unnecessary spaces.
            - None or 'auto': Use default of the selected `table_format`.
        shrink_width: boolean
            If `True` disregard width specified by the format strings,
            such that columns can become narrower.
        missing: string
            Indicate missing data by this string.
        center_columns: boolean
            If True center all columns (markdown, html, and latex).
        latex_label_command: string
            LaTeX command for formatting header labels.
            E.g. 'textbf' for making the header labels bold.
        latex_merge_std: string
            Merge header of columns with standard deviations with previous column
            (LaTeX tables only).

        Returns
        -------
        file_name: string or None
            The full name of the file into which the data were written.

        Supported file formats
        ----------------------
        
        ## `dat`: data text file
        ``` plain
        # info           reaction     
        # size   weight  delay  jitter
        # m      kg      ms     mm    
           2.34     123   98.7      23
          56.70    3457   54.3      45
           8.90      43   67.9     345
        ```

        ## `ascii`: ascii-art table
        ``` plain
        |---------------------------------|
        | info           | reaction       |
        | size  | weight | delay | jitter |
        | m     | kg     | ms    | mm     |
        |-------|--------|-------|--------|
        |  2.34 |    123 |  98.7 |     23 |
        | 56.70 |   3457 |  54.3 |     45 |
        |  8.90 |     43 |  67.9 |    345 |
        |---------------------------------|
        ```

        ## `csv`: comma separated values
        ``` plain
        size/m,weight/kg,delay/ms,jitter/mm
        2.34,123,98.7,23
        56.70,3457,54.3,45
        8.90,43,67.9,345
        ```

        ## `rtai`: rtai-style table
        ``` plain
        RTH| info         | reaction     
        RTH| size | weight| delay| jitter
        RTH| m    | kg    | ms   | mm    
        RTD|  2.34|    123|  98.7|     23
        RTD| 56.70|   3457|  54.3|     45
        RTD|  8.90|     43|  67.9|    345
        ```

        ## `md`: markdown
        ``` plain
        | size/m | weight/kg | delay/ms | jitter/mm |
        |------:|-------:|------:|-------:|
        |  2.34 |    123 |  98.7 |     23 |
        | 56.70 |   3457 |  54.3 |     45 |
        |  8.90 |     43 |  67.9 |    345 |
        ```

        ## `tex`: latex tabular
        ``` tex
        \\begin{tabular}{rrrr}
          \\hline
          \\multicolumn{2}{l}{info} & \\multicolumn{2}{l}{reaction} \\
          \\multicolumn{1}{l}{size} & \\multicolumn{1}{l}{weight} & \\multicolumn{1}{l}{delay} & \\multicolumn{1}{l}{jitter} \\
          \\multicolumn{1}{l}{m} & \\multicolumn{1}{l}{kg} & \\multicolumn{1}{l}{ms} & \\multicolumn{1}{l}{mm} \\
          \\hline
          2.34 & 123 & 98.7 & 23 \\
          56.70 & 3457 & 54.3 & 45 \\
          8.90 & 43 & 67.9 & 345 \\
          \\hline
        \\end{tabular}
        ```

        ## `html`: html
        ``` html
        <table>
        <thead>
          <tr class="header">
            <th align="left" colspan="2">info</th>
            <th align="left" colspan="2">reaction</th>
          </tr>
          <tr class="header">
            <th align="left">size</th>
            <th align="left">weight</th>
            <th align="left">delay</th>
            <th align="left">jitter</th>
          </tr>
          <tr class="header">
            <th align="left">m</th>
            <th align="left">kg</th>
            <th align="left">ms</th>
            <th align="left">mm</th>
          </tr>
        </thead>
        <tbody>
          <tr class"odd">
            <td align="right">2.34</td>
            <td align="right">123</td>
            <td align="right">98.7</td>
            <td align="right">23</td>
          </tr>
          <tr class"even">
            <td align="right">56.70</td>
            <td align="right">3457</td>
            <td align="right">54.3</td>
            <td align="right">45</td>
          </tr>
          <tr class"odd">
            <td align="right">8.90</td>
            <td align="right">43</td>
            <td align="right">67.9</td>
            <td align="right">345</td>
          </tr>
        </tbody>
        </table>
        ```
        """
        # fix parameter:
        if table_format == 'auto':
            table_format = None
        if delimiter == 'auto':
            delimiter = None
        if unit_style == 'auto':
            unit_style = None
        if column_numbers == 'none':
            column_numbers = None
        if sections == 'auto':
            sections = None
        if align_columns == 'auto':
            align_columns = None
        # open file:
        own_file = False
        file_name = None
        if not hasattr(fh, 'write'):
            _, ext = os.path.splitext(fh)
            if table_format is None:
                if len(ext) > 1 and ext[1:] in self.ext_formats:
                    table_format = self.ext_formats[ext[1:]]
            elif not ext or not ext[1:].lower() in self.ext_formats:
                fh += '.' + self.extensions[table_format]
            file_name = fh
            fh = open(fh, 'w')
            own_file = True
        if table_format is None:
            table_format = 'dat'
        # set style:        
        if table_format[0] == 'd':
            align_columns = True
            begin_str = ''
            end_str = ''
            header_start = '# '
            header_sep = '  '
            header_close = ''
            header_end = '\n'
            data_start = '  '
            data_sep = '  '
            data_close = ''
            data_end = '\n'
            top_line = False
            header_line = False
            bottom_line = False
            if delimiter is not None:
                header_sep = delimiter
                data_sep = delimiter
            if sections is None:
                sections = 1000
        elif table_format[0] == 'a':
            align_columns = True
            begin_str = ''
            end_str = ''
            header_start = '| '
            header_sep = ' | '
            header_close = ''
            header_end = ' |\n'
            data_start = '| '
            data_sep = ' | '
            data_close = ''
            data_end = ' |\n'
            top_line = True
            header_line = True
            bottom_line = True
            if delimiter is not None:
                header_sep = delimiter
                data_sep = delimiter
            if sections is None:
                sections = 1000
        elif table_format[0] == 'c':
            # csv according to http://www.ietf.org/rfc/rfc4180.txt :
            column_numbers=None
            if unit_style is None:
                unit_style = 'header'
            if align_columns is None:
                align_columns = False
            begin_str = ''
            end_str = ''
            header_start=''
            header_sep = ','
            header_close = ''
            header_end='\n'
            data_start=''
            data_sep = ','
            data_close = ''
            data_end='\n'
            top_line = False
            header_line = False
            bottom_line = False
            if delimiter is not None:
                header_sep = delimiter
                data_sep = delimiter
            if sections is None:
                sections = 0
        elif table_format[0] == 'r':
            align_columns = True
            begin_str = ''
            end_str = ''
            header_start = 'RTH| '
            header_sep = '| '
            header_close = ''
            header_end = '\n'
            data_start = 'RTD| '
            data_sep = '| '
            data_close = ''
            data_end = '\n'
            top_line = False
            header_line = False
            bottom_line = False
            if sections is None:
                sections = 1000
        elif table_format[0] == 'm':
            if unit_style is None or unit_style == 'row':
                unit_style = 'header'
            align_columns = True
            begin_str = ''
            end_str = ''
            header_start='| '
            header_sep = ' | '
            header_close = ''
            header_end=' |\n'
            data_start='| '
            data_sep = ' | '
            data_close = ''
            data_end=' |\n'
            top_line = False
            header_line = True
            bottom_line = False
            if sections is None:
                sections = 0
        elif table_format[0] == 'h':
            align_columns = False
            begin_str = '<table>\n<thead>\n'
            end_str = '</tbody>\n</table>\n'
            if center_columns:
                header_start='  <tr>\n    <th align="center"'
                header_sep = '</th>\n    <th align="center"'
            else:
                header_start='  <tr>\n    <th align="left"'
                header_sep = '</th>\n    <th align="left"'
            header_close = '>'
            header_end='</th>\n  </tr>\n'
            data_start='  <tr>\n    <td'
            data_sep = '</td>\n    <td'
            data_close = '>'
            data_end='</td>\n  </tr>\n'
            top_line = False
            header_line = False
            bottom_line = False
            if sections is None:
                sections = 1000
        elif table_format[0] == 't':
            if align_columns is None:
                align_columns = False
            begin_str = '\\begin{tabular}'
            end_str = '\\end{tabular}\n'
            header_start='  '
            header_sep = ' & '
            header_close = ''
            header_end=' \\\\\n'
            data_start='  '
            data_sep = ' & '
            data_close = ''
            data_end=' \\\\\n'
            top_line = True
            header_line = True
            bottom_line = True
            if sections is None:
                sections = 1000
        else:
            if align_columns is None:
                align_columns = True
            begin_str = ''
            end_str = ''
            header_start = ''
            header_sep = '  '
            header_close = ''
            header_end = '\n'
            data_start = ''
            data_sep = '  '
            data_close = ''
            data_end = '\n'
            top_line = False
            header_line = False
            bottom_line = False
            if sections is None:
                sections = 1000
        # check units:
        if unit_style is None:
            unit_style = 'row'
        have_units = False
        for u in self.units:
            if u and u != '1' and u != '-':
                have_units = True
                break
        if not have_units:
            unit_style = 'none'
        # find std columns:
        stdev_col = np.zeros(len(self.header), dtype=np.bool)
        for c in range(len(self.header)-1):
            if self.header[c+1][0].lower() in ['sd', 'std', 's.d.', 'stdev'] and \
               not self.hidden[c+1]:
                stdev_col[c] = True
        # begin table:
        fh.write(begin_str)
        if table_format[0] == 't':
            fh.write('{')
            merged = False
            for h, f, s in zip(self.hidden, self.formats, stdev_col):
                if merged:
                    fh.write('l')
                    merged = False
                    continue
                if h:
                    continue
                if latex_merge_std and s:
                    fh.write('r@{$\\,\\pm\\,$}')
                    merged = True
                elif center_columns:
                    fh.write('c')
                elif f[1] == '-':
                    fh.write('l')
                else:
                    fh.write('r')
            fh.write('}\n')
        # retrieve column formats and widths:
        widths = []
        widths_pos = []
        for c, f in enumerate(self.formats):
            w = 0
            # position of width specification:
            i0 = 1
            if f[1] == '-' :
                i0 = 2
            i1 = f.find('.')
            if not shrink_width:
                if f[i0:i1]:
                    w = int(f[i0:i1])
            widths_pos.append((i0, i1))
            # adapt width to header label:
            hw = len(self.header[c][0])
            if unit_style == 'header' and self.units[c] and\
               self.units[c] != '1' and self.units[c] != '-':
                hw += 1 + len(self.units[c])
            if w < hw:
                w = hw
            # adapt width to data:
            if f[-1] == 's':
                for v in self.data[c]:
                    if not isinstance(v, float) and w < len(v):
                        w = len(v)
            else:
                fs = f[:i0] + str(0) + f[i1:]
                for v in self.data[c]:
                    if isinstance(v, float) and m.isnan(v):
                        s = missing
                    else:
                        s = fs % v
                    if w < len(s):
                        w = len(s)
            widths.append(w)
        # adapt width to sections:
        sec_indices = [0] * self.nsecs
        sec_widths = [0] * self.nsecs
        sec_columns = [0] * self.nsecs
        for c in range(len(self.header)):
            w = widths[c]
            for l in range(min(self.nsecs, sections)):
                if 1+l < len(self.header[c]):
                    if c > 0 and sec_columns[l] > 0 and \
                       1+l < len(self.header[sec_indices[l]]) and \
                       len(self.header[sec_indices[l]][1+l]) > sec_widths[l]:
                        dw = len(self.header[sec_indices[l]][1+l]) - sec_widths[l]
                        nc = sec_columns[l]
                        ddw = np.zeros(nc, dtype=int) + dw // nc
                        ddw[:dw % nc] += 1
                        wk = 0
                        for ck in range(sec_indices[l], c):
                            if not self.hidden[ck]:
                                widths[ck] += ddw[wk]
                                wk += 1
                    sec_widths[l] = 0
                    sec_indices[l] = c
                if not self.hidden[c]:
                    if sec_widths[l] > 0:
                        sec_widths[l] += len(header_sep)
                    sec_widths[l] += w
                    sec_columns[l] += 1
        # set width of format string:
        formats = []
        for c, (f, w) in enumerate(zip(self.formats, widths)):
            formats.append(f[:widths_pos[c][0]] + str(w) + f[widths_pos[c][1]:])
        # top line:
        if top_line:
            if table_format[0] == 't':
                fh.write('  \\hline \\\\[-2ex]\n')
            else:
                first = True
                fh.write(header_start.replace(' ', '-'))
                for c in range(len(self.header)):
                    if self.hidden[c]:
                        continue
                    if not first:
                        fh.write('-'*len(header_sep))
                    first = False
                    fh.write(header_close)
                    w = widths[c]
                    fh.write(w*'-')
                fh.write(header_end.replace(' ', '-'))
        # section and column headers:
        nsec0 = self.nsecs-sections
        if nsec0 < 0:
            nsec0 = 0
        for ns in range(nsec0, self.nsecs+1):
            nsec = self.nsecs-ns
            first = True
            last = False
            merged = False
            fh.write(header_start)
            for c in range(len(self.header)):
                if nsec < len(self.header[c]):
                    # section width and column count:
                    sw = -len(header_sep)
                    columns = 0
                    if not self.hidden[c]:
                        sw = widths[c]
                        columns = 1
                    for k in range(c+1, len(self.header)):
                        if nsec < len(self.header[k]):
                            break
                        if self.hidden[k]:
                            continue
                        sw += len(header_sep) + widths[k]
                        columns += 1
                    else:
                        last = True
                        if len(header_end.strip()) == 0:
                            sw = 0  # last entry needs no width
                    if columns == 0:
                        continue
                    if not first and not merged:
                        fh.write(header_sep)
                    first = False
                    if table_format[0] == 'c':
                        sw -= len(header_sep)*(columns-1)
                    elif table_format[0] == 'h':
                        if columns>1:
                            fh.write(' colspan="%d"' % columns)
                    elif table_format[0] == 't':
                        if merged:
                            merged = False
                            continue
                        if latex_merge_std and nsec == 0 and stdev_col[c]:
                            merged = True
                            fh.write('\\multicolumn{%d}{c}{' % (columns+1))
                        elif center_columns:
                            fh.write('\\multicolumn{%d}{c}{' % columns)
                        else:
                            fh.write('\\multicolumn{%d}{l}{' % columns)
                        if latex_label_command:
                            fh.write('\\%s{' % latex_label_command)
                    fh.write(header_close)
                    hs = self.header[c][nsec]
                    if nsec == 0 and unit_style == 'header':
                        if self.units[c] and self.units[c] != '1' and self.units[c] != '-':
                            hs += '/' + self.units[c]
                    if align_columns and not table_format[0] in 'th':
                        f = '%%-%ds' % sw
                        fh.write(f % hs)
                    else:
                        fh.write(hs)
                    if table_format[0] == 'c':
                        if not last:
                            fh.write(header_sep*(columns-1))
                    elif table_format[0] == 't':
                        if latex_label_command:
                            fh.write('}')
                        fh.write('}')
            fh.write(header_end)
        # units:
        if unit_style == 'row':
            first = True
            merged = False
            fh.write(header_start)
            for c in range(len(self.header)):
                if self.hidden[c] or merged:
                    merged = False
                    continue
                if not first:
                    fh.write(header_sep)
                first = False
                fh.write(header_close)
                unit = self.units[c]
                if not unit:
                    unit = '-'
                if table_format[0] == 't':
                    if latex_merge_std and stdev_col[c]:
                        merged = True
                        fh.write('\\multicolumn{2}{c}{%s}' % latex_unit(unit))
                    elif center_columns:
                        fh.write('\\multicolumn{1}{c}{%s}' % latex_unit(unit))
                    else:
                        fh.write('\\multicolumn{1}{l}{%s}' % latex_unit(unit))
                else:
                    if align_columns and not table_format[0] in 'h':
                        f = '%%-%ds' % widths[c]
                        fh.write(f % unit)
                    else:
                        fh.write(unit)
            fh.write(header_end)
        # column numbers:
        if column_numbers is not None:
            first = True
            fh.write(header_start)
            for c in range(len(self.header)):
                if self.hidden[c]:
                    continue
                if not first:
                    fh.write(header_sep)
                first = False
                fh.write(header_close)
                i = c
                if column_numbers == 'num':
                    i = c+1
                aa = index2aa(c, 'a')
                if column_numbers == 'AA':
                    aa = index2aa(c, 'A')
                if table_format[0] == 't':
                    if column_numbers == 'num' or column_numbers == 'index':
                        fh.write('\\multicolumn{1}{l}{%d}' % i)
                    else:
                        fh.write('\\multicolumn{1}{l}{%s}' % aa)
                else:
                    if column_numbers == 'num' or column_numbers == 'index':
                        if align_columns:
                            f = '%%%dd' % widths[c]
                            fh.write(f % i)
                        else:
                            fh.write('%d' % i)
                    else:
                        if align_columns:
                            f = '%%-%ds' % widths[c]
                            fh.write(f % aa)
                        else:
                            fh.write(aa)
            fh.write(header_end)
        # header line:
        if header_line:
            if table_format[0] == 'm':
                fh.write('|')
                for c in range(len(self.header)):
                    if self.hidden[c]:
                        continue
                    w = widths[c]+2
                    if center_columns:
                        fh.write(':' + (w-2)*'-' + ':|')
                    elif formats[c][1] == '-':
                        fh.write(w*'-' + '|')
                    else:
                        fh.write((w-1)*'-' + ':|')
                fh.write('\n')
            elif table_format[0] == 't':
                fh.write('  \\hline \\\\[-2ex]\n')
            else:
                first = True
                fh.write(header_start.replace(' ', '-'))
                for c in range(len(self.header)):
                    if self.hidden[c]:
                        continue
                    if not first:
                        fh.write(header_sep.replace(' ', '-'))
                    first = False
                    fh.write(header_close)
                    w = widths[c]
                    fh.write(w*'-')
                fh.write(header_end.replace(' ', '-'))
        # start table data:
        if table_format[0] == 'h':
            fh.write('</thead>\n<tbody>\n')
        # data:
        for k in range(self.rows()):
            first = True
            merged = False
            fh.write(data_start)
            for c, f in enumerate(formats):
                if self.hidden[c] or merged:
                    merged = False
                    continue
                if not first:
                    fh.write(data_sep)
                first = False
                if table_format[0] == 'h':
                    if center_columns:
                        fh.write(' align="center"')
                    elif f[1] == '-':
                        fh.write(' align="left"')
                    else:
                        fh.write(' align="right"')
                fh.write(data_close)
                if k >= len(self.data[c]) or \
                   (isinstance(self.data[c][k], float) and m.isnan(self.data[c][k])):
                    # missing data:
                    if table_format[0] == 't' and latex_merge_std and stdev_col[c]:
                        merged = True
                        fh.write('\\multicolumn{2}{c}{%s}' % missing)
                    elif align_columns:
                        if f[1] == '-':
                            fn = '%%-%ds' % widths[c]
                        else:
                            fn = '%%%ds' % widths[c]
                        fh.write(fn % missing)
                    else:
                        fh.write(missing)
                else:
                    # data value:
                    ds = f % self.data[c][k]
                    if not align_columns:
                        ds = ds.strip()
                    fh.write(ds)
            fh.write(data_end)
        # bottom line:
        if bottom_line:
            if table_format[0] == 't':
                fh.write('  \\hline\n')
            else:
                first = True
                fh.write(header_start.replace(' ', '-'))
                for c in range(len(self.header)):
                    if self.hidden[c]:
                        continue
                    if not first:
                        fh.write('-'*len(header_sep))
                    first = False
                    fh.write(header_close)
                    w = widths[c]
                    fh.write(w*'-')
                fh.write(header_end.replace(' ', '-'))
        # end table:
        fh.write(end_str)
        # close file:
        if own_file:
            fh.close()
        # return file name:
        return file_name

            
    def __str__(self):
        """
        Write table to a string.
        """
        stream = StringIO()
        self.write(stream, table_format='out')
        return stream.getvalue()
                

    def load(self, fh, missing='-'):
        """
        Load table from file or stream.

        File type and properties are automatically inferred.

        Parameters
        ----------
        fh: filename or stream
            If not a stream, the file with name `fh` is opened for reading.
        missing: string
            Missing data are indicated by this string.

        Raises
        ------
        FileNotFoundError:
            If `fh` is a path that does not exist.
        """

        def read_key_line(line, sep, table_format):
            if sep is None:
                cols, indices = zip(*[(m.group(0), m.start()) for m in re.finditer(r'( ?[\S]+)+(?=[ ][ ]+|\Z)', line.strip())])
            elif table_format == 'csv':
                cols, indices = zip(*[(c.strip(), i) for i, c in enumerate(line.strip().split(sep)) if c.strip()])
                return cols, indices
            else:
                seps = r'[^'+re.escape(sep)+']+'
                cols, indices = zip(*[(m.group(0), m.start()) for m in re.finditer(seps, line.strip())])
            colss = []
            indicess = []
            if table_format == 'tex':
                i = 0
                for c in cols:
                    if 'multicolumn' in c:
                        fields = c.split('{')
                        n = int(fields[1].strip().rstrip('}').rstrip())
                        colss.append(fields[3].strip().rstrip('}').rstrip())
                        indicess.append(i)
                        i += n
                    else:
                        colss.append(c.strip())
                        indicess.append(i)
                        i += 1
            else:
                for k, (c, i) in enumerate(zip(cols, indices)):
                    if k == 0:
                        c = c.lstrip('|')
                    if k == len(cols)-1:
                        c = c.rstrip('|')
                    cs = c.strip()
                    colss.append(cs)
                    indicess.append(i)
            return colss, indicess

        def read_data_line(line, sep, post, precd, alld, numc, exped, fixed, strf, missing):
            # read line:
            cols = []
            if sep is None:
                cols = [m.group(0) for m in re.finditer(r'\S+', line.strip())]
            else:
                seps = r'[^'+re.escape(sep)+']+'
                cols = [m.group(0).strip() for m in re.finditer(seps, line.strip())]
                cols[0] = cols[0].lstrip('|').lstrip()
                cols[-1] = cols[-1].rstrip('|').rstrip()
            cols = [c for c in cols if c not in '|']
            # read columns:
            for k, c in enumerate(cols):
                c = c.strip()
                try:
                    v = float(c)
                    ad = 0
                    ve = c.split('e')
                    if len(ve) <= 1:
                        exped[k] = False
                    else:
                        ad = len(ve[1])+1
                    vc = ve[0].split('.')
                    ad += len(vc[0])
                    prec = len(vc[0].lstrip('-').lstrip('+').lstrip('0')) 
                    if len(vc) == 2:
                        if numc[k] and post[k] != len(vc[1]):
                            fixed[k] = False
                        if post[k] < len(vc[1]):
                            post[k] = len(vc[1])
                        ad += len(vc[1])+1
                        prec += len(vc[1].rstrip('0'))
                    if precd[k] < prec:
                        precd[k] = prec
                    if alld[k] < ad:
                        alld[k] = ad
                    numc[k] = True
                except ValueError:
                    if c == missing:
                        v = np.nan
                    else:
                        strf[k] = True
                        if alld[k] < len(c):
                            alld[k] = len(c)
                        v = c
                self.append_data(v, k)

        # initialize:
        self.data = []
        self.shape = (0, 0)
        self.header = []
        self.nsecs = 0
        self.units = []
        self.formats = []
        self.hidden = []
        self.setcol = 0
        self.addcol = 0
        # open file:
        own_file = False
        if not hasattr(fh, 'readline'):
            fh = open(fh, 'r')
            own_file = True
        # read inital lines of file:
        key = []
        data = []
        target = data
        comment = False
        table_format='dat'        
        for line in fh:
            line = line.rstrip()
            if line:
                if r'\begin{tabular' in line:
                    table_format='tex'
                    target = key
                    continue
                if table_format == 'tex':
                    if r'\end{tabular' in line:
                        break
                    if r'\hline' in line:
                        if key:
                            target = data
                        continue
                    line = line.rstrip(r'\\')
                if line[0] == '#':
                    comment = True
                    table_format='dat'        
                    target = key
                    line = line.lstrip('#')
                elif comment:
                    target = data
                if line[0:3] == 'RTH':
                    target = key
                    line = line[3:]
                    table_format='rtai'
                elif line[0:3] == 'RTD':
                    target = data
                    line = line[3:]
                    table_format='rtai'        
                if (line[0:3] == '|--' or line[0:3] == '|:-') and \
                   (line[-3:] == '--|' or line[-3:] == '-:|'):
                    if not data and not key:
                        table_format='ascii'
                        target = key
                        continue
                    elif not key:
                        table_format='md'
                        key = data
                        data = []
                        target = data
                        continue
                    elif not data:
                        target = data
                        continue
                    else:
                        break
                target.append(line)
            else:
                break
            if len(data) > 5:
                break
        # find column separator of data and number of columns:
        col_seps = ['|', ',', ';', ':', '\t', '&', None]
        colstd = np.zeros(len(col_seps))
        colnum = np.zeros(len(col_seps), dtype=int)
        for k, sep in enumerate(col_seps):
            cols = []
            s = 5 if len(data) >= 8 else len(data) - 3
            if s < 0 or key:
                s = 0
            for line in data[s:]:
                cs = line.strip().split(sep)
                if not cs[0]:
                    cs = cs[1:]
                if cs and not cs[-1]:
                    cs = cs[:-1]
                cols.append(len(cs))
            colstd[k] = np.std(cols)
            colnum[k] = np.median(cols)
        if np.max(colnum) < 2:
            sep = None
            colnum = 1
        else:
            ci = np.where(np.array(colnum)>1.5)[0]
            ci = ci[np.argmin(colstd[ci])]
            sep = col_seps[ci]
            colnum = int(colnum[ci])
        # fix key:
        if not key and sep is not None and sep in ',;:\t|':
            table_format = 'csv'
        # read key:
        key_cols = []
        key_indices = []
        for line in key:
            cols, indices = read_key_line(line, sep, table_format)
            key_cols.append(cols)
            key_indices.append(indices)
        if not key_cols:
            # no obviously marked table key:
            key_num = 0
            for line in data:
                cols, indices = read_key_line(line, sep, table_format)
                numbers = 0
                for c in cols:
                    try:
                        v = float(c)
                        numbers += 1
                    except ValueError:
                        pass
                if numbers == 0:
                    key_cols.append(cols)
                    key_indices.append(indices)
                    key_num += 1
                else:
                    break
            data = data[key_num:]
        kr = len(key_cols)-1
        # check for key with column indices:
        if kr >= 0:
            cols = key_cols[kr]
            numrow = True
            try:
                pv = int(cols[0])
                for c in cols[1:]:
                    v = int(c)
                    if v != pv+1:
                        numrow = False
                        break
                    pv = v
            except ValueError:
                try:
                    pv = aa2index(cols[0])
                    for c in cols[1:]:
                        v = aa2index(c)
                        if v != pv+1:
                            numrow = False
                            break
                        pv = v
                except ValueError:
                    numrow = False
            if numrow:
                kr -= 1
        # check for unit line:
        units = None
        if kr > 0 and len(key_cols[kr]) == len(key_cols[kr-1]):
            units = key_cols[kr]
            kr -= 1
        # column labels:
        if kr >= 0:
            if units is None:
                # units may be part of the label:
                labels = []
                units = []
                for c in key_cols[kr]:
                    if c[-1] == ')':
                        lu = c[:-1].split('(')
                        if len(lu) >= 2:
                            labels.append(lu[0].strip())
                            units.append('('.join(lu[1:]).strip())
                            continue
                    lu = c.split('/')
                    if len(lu) >= 2:
                        labels.append(lu[0].strip())
                        units.append('/'.join(lu[1:]).strip())
                    else:
                        labels.append(c)
                        units.append('')
            else:
                labels = key_cols[kr]
            indices = key_indices[kr]
            # init table columns:
            for k in range(colnum):
                self.append(labels[k], units[k], '%g')
        # read in sections:
        while kr > 0:
            kr -= 1
            for sec_label, sec_inx in zip(key_cols[kr], key_indices[kr]):
                col_inx = indices.index(sec_inx)
                self.header[col_inx].append(sec_label)
                if self.nsecs < len(self.header[col_inx])-1:
                    self.nsecs = len(self.header[col_inx])-1
        # read data:
        post = np.zeros(colnum)
        precd = np.zeros(colnum)
        alld = np.zeros(colnum)
        numc = [False] * colnum
        exped = [True] * colnum
        fixed = [True] * colnum
        strf = [False] * colnum
        for line in data:
            read_data_line(line, sep, post, precd, alld, numc, exped, fixed, strf, missing)
        # read remaining data:
        for line in fh:
            line = line.rstrip()
            if table_format == 'tex':
                if r'\end{tabular' in line or r'\hline' in line:
                    break
                line = line.rstrip(r'\\')
            if (line[0:3] == '|--' or line[0:3] == '|:-') and \
                (line[-3:] == '--|' or line[-3:] == '-:|'):
                break
            if line[0:3] == 'RTD':
                line = line[3:]
            read_data_line(line, sep, post, precd, alld, numc, exped, fixed, strf, missing)
        # set formats:
        for k in range(len(alld)):
            if strf[k]:
                self.set_format('%%-%ds' % alld[k], k)
            elif exped[k]:
                self.set_format('%%%d.%de' % (alld[k], post[k]), k)
            elif fixed[k]:
                self.set_format('%%%d.%df' % (alld[k], post[k]), k)
            else:
                self.set_format('%%%d.%dg' % (alld[k], precd[k]), k)
        # close file:
        if own_file:
            fh.close()


def write(fh, data, header, units=None, formats=None, table_format=None, delimiter=None,
              unit_style=None, column_numbers=None, sections=None,
              align_columns=None, shrink_width=True, missing='-',
              center_columns=False, latex_label_command='', latex_merge_std=False):
    """
    Construct table and write to file.

    Parameters
    ----------
    fh: filename or stream
        If not a stream, the file with name `fh` is opened.
        If `fh` does not have an extension,
        the `table_format` is appended as an extension.
        Otherwise `fh` is used as a stream for writing.
    data: 1-D or 2-D array of data
          The data of the table.
    header: list of string
        Header labels for each column.
    units: list of string, optional
        Unit strings for each column.
    formats: string or list of string, optional
        Format strings for each column. If only a single format string is
        given, then all columns are initialized with this format string.

    See `TableData.write()` for a description of all other parameters.

    Example
    -------
    ```
    write(sys.stdout, np.random.randn(4,3), ['aaa', 'bbb', 'ccc'], units=['m', 's', 'g'], formats='%.2f')
    ```
    """
    td = TableData(data, header, units, formats)
    td.write(fh, table_format=table_format, unit_style=unit_style,
             column_numbers=column_numbers, missing=missing, shrink_width=shrink_width,
             delimiter=delimiter, align_columns=align_columns, sections=sections,
             latex_label_command=latex_label_command, latex_merge_std=latex_merge_std)

    
def add_write_table_config(cfg, table_format=None, delimiter=None,
                           unit_style=None, column_numbers=None, sections=None,
                           align_columns=None, shrink_width=True, missing='-',
                           center_columns=False, latex_label_command='', latex_merge_std=False):
    """ Add parameter specifying how to write a table to a file as a new section to a configuration.

    Parameters
    ----------
    cfg: ConfigFile
        The configuration.
    """

    cfg.add_section('File format for storing analysis results:')
    cfg.add('fileFormat', table_format or 'auto', '', 'Default file format used to store analysis results.\nOne of %s.' % ', '.join(TableData.formats))
    cfg.add('fileDelimiter', delimiter or 'auto', '', 'String used to separate columns or "auto".')
    cfg.add('fileUnitStyle', unit_style or 'auto', '', 'Add units as extra row ("row"), add units to header label separated by "/" ("header"), do not print out units ("none"), or "auto".')
    cfg.add('fileColumnNumbers', column_numbers or 'none', '', 'Add line with column indices ("index", "num", "aa", "AA", or "none")')
    cfg.add('fileSections', sections or 'auto', '', 'Maximum number of section levels or "auto"')
    cfg.add('fileAlignColumns', align_columns or 'auto', '', 'If True, write all data of a column using the same width, if False write the data without any white space, or "auto".')
    cfg.add('fileShrinkColumnWidth', shrink_width, '', 'Allow to make columns narrower than specified by the corresponding format strings.')
    cfg.add('fileMissing', missing, '', 'String used to indicate missing data values.')
    cfg.add('fileCenterColumns', center_columns, '', 'Center content of all columns instead of left align columns of strings and right align numbers (markdown, html, and latex).')
    cfg.add('fileLaTeXLabelCommand', latex_label_command, '', 'LaTeX command name for formatting column labels of the table header.')
    cfg.add('fileLaTeXMergeStd', latex_merge_std, '', 'Merge header of columns with standard deviations with previous column (LaTeX tables only).')


def write_table_args(cfg):
    """ Translates a configuration to the respective parameter names for writing a table to a file.
    
    The return value can then be passed as key-word arguments to TableData.write().

    Parameters
    ----------
    cfg: ConfigFile
        The configuration.

    Returns
    -------
    a: dict
        Dictionary with names of arguments of the `TableData.write` function
        and their values as supplied by `cfg`.
    """

    d = cfg.map({'table_format': 'fileFormat',
                 'delimiter': 'fileDelimiter',
                 'unit_style': 'fileUnitStyle',
                 'column_numbers': 'fileColumnNumbers',
                 'sections': 'fileSections',
                 'align_columns': 'fileAlignColumns',
                 'shrink_width': 'fileShrinkColumnWidth',
                 'missing': 'fileMissing',
                 'center_columns': 'fileCenterColumns',
                 'latex_label_command': 'fileLaTeXLabelCommand',
                 'latex_merge_std': 'fileLaTeXMergeStd'})
    if 'sections' in d:
        if d['sections'] != 'auto':
            d['sections'] = int(d['sections'])
    return d


def latex_unit(unit):
    """ Translate unit string into SIunit LaTeX code.
    
    Parameters
    ----------
    unit: string
        String enoting a unit.
        
    Returns
    -------
    unit: string
        Unit string as valid LaTeX code.
    """
    si_prefixes = {'y': '\\yocto',
                  'z': '\\zepto',
                  'a': '\\atto',
                  'f': '\\femto',
                  'p': '\\pico',
                  'n': '\\nano',
                  'u': '\\micro',
                  'm': '\\milli',
                  'c': '\\centi',
                  'd': '\\deci',
                  'h': '\\hecto',
                  'k': '\\kilo',
                  'M': '\\mega',
                  'G': '\\giga',
                  'T': '\\tera',
                  'P': '\\peta',
                  'E': '\\exa',
                  'Z': '\\zetta',
                  'Y': '\\yotta' }
    si_units = {'m': '\\metre',
               'g': '\\gram',
               's': '\\second',
               'A': '\\ampere',
               'K': '\\kelvin',
               'mol': '\\mole',
               'cd': '\\candela',
               'Hz': '\\hertz',
               'N': '\\newton',
               'Pa': '\\pascal',
               'J': '\\joule',
               'W': '\\watt',
               'C': '\\coulomb',
               'V': '\\volt',
               'F': '\\farad',
               'O': '\\ohm',
               'S': '\\siemens',
               'Wb': '\\weber',
               'T': '\\tesla',
               'H': '\\henry',
               'C': '\\celsius',
               'lm': '\\lumen',
               'lx': '\\lux',
               'Bq': '\\becquerel',
               'Gv': '\\gray',
               'Sv': '\\sievert'}
    other_units = {"'": '\\arcminute',
               "''": '\\arcsecond',
               'a': '\\are',
               'd': '\\dday',
               'eV': '\\electronvolt',
               'ha': '\\hectare',
               'h': '\\hour',
               'L': '\\liter',
               'l': '\\litre',
               'min': '\\minute',
               'Np': '\\neper',
               'rad': '\\rad',
               't': '\\ton',
               '%': '\\%'}
    unit_powers = {'^2': '\\squared',
              '^3': '\\cubed',
              '/': '\\per',
              '^-1': '\\power{}{-1}',
              '^-2': '\\rpsquared',
              '^-3': '\\rpcubed'}
    if '\\' in unit:   # this string is already translated!
        return unit
    units = ''
    j = len(unit)
    while j >= 0:
        for k in range(-3, 0):
            if j+k < 0:
                continue
            uss = unit[j+k:j]
            if uss in unit_powers:
                units = unit_powers[uss] + units
                break
            elif uss in other_units:
                units = other_units[uss] + units
                break
            elif uss in si_units:
                units = si_units[uss] + units
                j = j+k
                k = 0
                if j-1 >= 0:
                    uss = unit[j-1:j]
                    if uss in si_prefixes:
                        units = si_prefixes[uss] + units
                        k = -1
                break
        else:
            k = -1
            units = unit[j+k:j] + units
        j = j + k
    return units


def index2aa(n, a='a'):
    """
    Convert an integer into an alphabetical representation.

    The integer number is converted into 'a', 'b', 'c', ..., 'z',
    'aa', 'ab', 'ac', ..., 'az', 'ba', 'bb', ...

    Inspired by https://stackoverflow.com/a/37604105

    Parameters
    ----------
    n: int
        An integer to be converted into alphabetical representation.
    a: string ('a' or 'A')
        Use upper or lower case characters.

    Returns
    -------
    ns: string
        Alphabetical represtnation of an integer.
    """
    d, m = divmod(n, 26)
    bm = chr(ord(a)+m)
    return index2aa(d-1, a) + bm if d else bm


def aa2index(s):
    """
    Convert an alphabetical representation to an index.

    The alphabetical representation 'a', 'b', 'c', ..., 'z',
    'aa', 'ab', 'ac', ..., 'az', 'ba', 'bb', ...
    is converted to an index starting with 0.

    Parameters
    ----------
    s: string
        Alphabetical representation of an index.

    Returns
    -------
    index: int
        The corresponding index.

    Raises
    ------
    ValueError:
        Invalid character in input string.
    """
    index = 0
    maxc = ord('z') - ord('a') + 1
    for c in s.lower():
        index *= maxc
        if ord(c) < ord('a') or ord(c) > ord('z'):
            raise ValueError('invalid character "%s" in string.' % c)
        index += ord(c) - ord('a') + 1
    return index-1

        
class IndentStream(object):
    """
    Filter an output stream and start each newline with a number of spaces.
    """
    def __init__(self, stream, indent=4):
        self.stream = stream
        self.indent = indent
        self.pending = True

    def __getattr__(self, attr_name):
        return getattr(self.stream, attr_name)

    def write(self, data):
        if not data:
            return
        if self.pending:
            self.stream.write(' '*self.indent)
            self.pending = False
        substr = data.rstrip('\n')
        rn = len(data) - len(substr)
        if len(substr) > 0:
            self.stream.write(substr.replace('\n', '\n'+' '*self.indent))
        if rn > 0:
            self.stream.write('\n'*rn)
            self.pending = True

    def flush(self):
        self.stream.flush()

        
if __name__ == "__main__":
    import os
    
    print("Checking tabledata module ...")
    print('')

    # setup a table:
    df = TableData()
    df.append(["data", "partial information", "ID"], "", "%-s", list('ABCDEFGH'))
    df.append("size", "m", "%6.2f", [2.34, 56.7, 8.9])
    df.append("full weight", "kg", "%.0f", 122.8)
    df.append_section("complete reaction")
    df.append("speed", "m/s", "%.3g", 98.7)
    df.append("median jitter", "mm", "%.1f", 23)
    df.append("size", "g", "%.2e", 1.234)
    df.append_data(np.nan, 2)  # single value
    df.append_data((0.543, 45, 1.235e2)) # remaining row
    df.append_data((43.21, 6789.1, 3405, 1.235e-4), 2) # next row
    a = 0.5*np.arange(1, 6)*np.random.randn(5, 5) + 10.0 + np.arange(5)
    df.append_data(a.T, 1) # rest of table
    df[3:6,'weight'] = [11.0]*3
    
    # write out in all formats:
    for tf in TableData.formats:
        print('    - `%s`: %s' % (tf, TableData.descriptions[tf]))
        print('      ```')
        iout = IndentStream(sys.stdout, 4+2)
        df.write(iout, table_format=tf)
        print('      ```')
        print('')
