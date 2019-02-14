"""
# Data file
class DataFile for reading and writing of data tables.
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
    

class DataFile:
    """
    Tables of data with a rich hierarchical header with units and formats.

    Indexing
    --------
    The DataFile is a mapping of keys (the column headers)
    to values (the column data).
    Iterating over the table goes over columns.

    The only exception is the [] operator that treats the table as a
    2D-array: rows first, columns second.

    File formats for writing
    ------------------------
    - `dat`: data text file
      ```
      # info           reaction     
      # size   weight  delay  jitter
      # m      kg      ms     mm    
         2.34     123   98.7      23
        56.70    3457   54.3      45
         8.90      43   67.9     345
      ```

    - `ascii`: ascii-art table
      ```
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

    - `csv`: comma separated values
      ```
      size/m,weight/kg,delay/ms,jitter/mm
      2.34,123,98.7,23
      56.70,3457,54.3,45
      8.90,43,67.9,345
      ```

    - `rtai`: rtai-style table
      ```
      RTH| info         | reaction     
      RTH| size | weight| delay| jitter
      RTH| m    | kg    | ms   | mm    
      RTD|  2.34|    123|  98.7|     23
      RTD| 56.70|   3457|  54.3|     45
      RTD|  8.90|     43|  67.9|    345
      ```

    - `md`: markdown
      ```
      | size/m | weight/kg | delay/ms | jitter/mm |
      |------:|-------:|------:|-------:|
      |  2.34 |    123 |  98.7 |     23 |
      | 56.70 |   3457 |  54.3 |     45 |
      |  8.90 |     43 |  67.9 |    345 |
      ```
      
    - `tex`: latex tabular
      ```
      \begin{tabular}{rrrr}
        \hline
        \multicolumn{2}{l}{info} & \multicolumn{2}{l}{reaction} \\
        \multicolumn{1}{l}{size} & \multicolumn{1}{l}{weight} & \multicolumn{1}{l}{delay} & \multicolumn{1}{l}{jitter} \\
        \multicolumn{1}{l}{m} & \multicolumn{1}{l}{kg} & \multicolumn{1}{l}{ms} & \multicolumn{1}{l}{mm} \\
        \hline
        2.34 & 123 & 98.7 & 23 \\
        56.70 & 3457 & 54.3 & 45 \\
        8.90 & 43 & 67.9 & 345 \\
        \hline
      \end{tabular}
      ```

    - `html`: html
      ```
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
    
    formats = ['dat', 'ascii', 'csv', 'rtai', 'md', 'tex', 'html']
    descriptions = {'dat': 'data text file', 'ascii': 'ascii-art table', 'csv': 'comma separated values', 'rtai': 'rtai-style table', 'md': 'markdown', 'tex': 'latex tabular', 'html': 'html markup'}
    extensions = {'dat': 'dat', 'ascii': 'txt', 'csv': 'csv', 'rtai': 'dat', 'md': 'md', 'tex': 'tex', 'html': 'html'}
    ext_formats = {'dat': 'dat', 'DAT': 'dat', 'txt': 'dat', 'TXT': 'dat', 'csv': 'csv', 'CSV': 'csv', 'md': 'md', 'MD': 'md', 'tex': 'tex', 'TEX': 'tex', 'html': 'html', 'HTML': 'html'}
    column_numbering = ['num', 'index', 'aa', 'AA']

    def __init__(self, filename=None):
        self.data = []
        self.shape = (0, 0)
        self.header = []
        self.nsecs = 0
        self.units = []
        self.formats = []
        self.hidden = []
        self.setcol = 0
        self.addcol = 0
        if filename is not None:
            self.load(filename)

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
        
    def append(self, label, unit, formats, value=None):
        """
        Append column to the table.

        Parameters
        ----------
        label: string or list of string
            Optional section titles and the name of the column.
        unit: string
            The unit of the column contents.
        formats: string
            The C-style format string used for printing out the column content, e.g.
            '%g', '%.2f', '%s', etc.
        val: None, float, int, string, etc. or list thereof
            If not None, data for the column.

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
            self.formats.append(formats)
            self.units.append(unit)
            self.hidden.append(False)
            self.data.append([])
            if self.nsecs < len(self.header[-1])-1:
                self.nsecs = len(self.header[-1])-1
        else:
            if isinstance(label, (list, tuple, np.ndarray)):
                self.header[self.addcol] = list(reversed(label)) + self.header[self.addcol]
            else:
                self.header[self.addcol] = [label] + self.header[self.addcol]
            self.units[self.addcol] = unit
            self.formats[self.addcol] = formats
            if self.nsecs < len(self.header[self.addcol])-1:
                self.nsecs = len(self.header[self.addcol])-1
        if value is not None:
            if isinstance(value, (list, tuple, np.ndarray)):
                self.data[-1].extend(value)
            else:
                self.data[-1].append(value)
        self.addcol = len(self.data)
        self.shape = (self.rows(), self.columns())
        return self.addcol-1
        
    def insert(self, column, label, unit, formats, value=None):
        """
        Insert a table column at a given position.

        Parameters
        ----------
        columns int or string
            Column before which to insert the new column.
            Column can be specified by index or name, see index() for details.
        label: string or list of string
            Optional section titles and the name of the column.
        unit: string
            The unit of the column contents.
        formats: string
            The C-style format string used for printing out the column content, e.g.
            '%g', '%.2f', '%s', etc.
        val: None, float, int, string, etc. or list thereof
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
        self.formats.insert(col, formats)
        self.units.insert(col, unit)
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

    def section(self, column, level):
        """
        The section name of a specified column.

        Parameters
        ----------
        col: None, int, or string
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
        col: None, int, or string
            A specification of a column.
            See self.index() for more information on how to specify a column.
        level: int
            The level of the section to be set. The column label itself is level=0.
        """
        column = self.index(column)
        self.header[column][level] = label
        return column

    def label(self, column):
        """
        The column name.

        Parameters
        ----------
        col: None, int, or string
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
        Set the column name.

        Parameters
        ----------
        label: string
            The new name to be used for the column.
        col: None, int, or string
            A specification of a column.
            See self.index() for more information on how to specify a column.
        """        
        column = self.index(column)
        self.header[column][0] = label
        return column

    def unit(self, column):
        """
        The unit of the column.

        Parameters
        ----------
        col: None, int, or string
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
        The unit of the column.

        Parameters
        ----------
        unit: string
            The new unit to be used for the column.
        col: None, int, or string
            A specification of a column.
            See self.index() for more information on how to specify a column.
        """
        column = self.index(column)
        self.units[column] = unit
        return column

    def format(self, column):
        """
        The format string of the column.

        Parameters
        ----------
        col: None, int, or string
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
        The format string of the column.

        Parameters
        ----------
        format: string
            The new format string to be used for the column.
        col: None, int, or string
            A specification of a column.
            See self.index() for more information on how to specify a column.
        """
        column = self.index(column)
        self.formats[column] = format
        return column

    def column_spec(self, column):
        """
        Full specification of a column.

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

    def keys(self):
        """
        List of unique column keys for all available columns.
        """
        return [self.column_spec(c) for c in range(self.columns())]

    def values(self):
        """
        List of column data corresponding to keys().
        """
        return self.data

    def items(self):
        """
        List of tuples with unique column specifications and the corresponding data.
        """
        return [(self.column_spec(c), self.data[c]) for c in range(self.columns())]

    def column_head(self, column):
        """
        The name, unit, and format of a column.

        Parameters
        ----------
        col: None, int, or string
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

    def table_header(self):
        """
        The header of the table without content.

        Return
        ------
        data: DataFile
            A DataFile object with the same header but empty data.
        """
        data = DataFile()
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
        Return next column as a list.
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
        """
        return self.__next__()

    def columns(self):
        """
        The number of columns.
        
        Returns
        -------
        columns: int
            The number of columns contained in the table.
        """
        return len(self.header)

    def rows(self):
        """
        The number of rows.
        
        Returns
        -------
        rows: int
            The number of rows contained in the table.
        """
        return max(map(len, self.data))

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
        table: DataFile
            A DataFile object with a single column.
        """
        data = DataFile()
        c = self.index(column)
        data.append(*self.column_head(c))
        data.data = [self.data[c]]
        data.nsecs = 0
        return data

    def row(self, index):
        """
        A single row of the table.

        Parameters
        ----------
        index: int
            The index of the row to be returned.

        Return
        ------
        data: DataFile
            A DataFile object with a single row.
        """
        data = DataFile()
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

    def __setupkey(self, key):
        """
        Helper function that turns a key into row and column indices.
        """
        if type(key) is not tuple:
            rows = key
            cols = range(self.columns())
        else:
            rows = key[0]
            cols = key[1]
        if isinstance(cols, slice):
            start = self.index(cols.start)
            stop = self.index(cols.stop)
            cols = slice(start, stop, cols.step)
            cols = range(self.columns())[cols]
        elif isinstance(cols, (list, tuple, np.ndarray)):
            cols = [self.index(inx) for inx in cols]
        else:
            cols = [self.index(cols)]
        return rows, cols

    def __getitem__(self, key):
        """
        Data elements specified by slice.

        Parameters
        -----------
        key:
            First key specifies row, (optional) second one the column.
            Columns can be specified by index or name, see index() for details.

        Returns
        -------
        data:
            - A single data value if a single row and a single column is specified.
            - A list of data elements if a single row or a single column is specified.
            - A DataFile object for multiple rows and columns.
        """
        rows, cols = self.__setupkey(key)
        if len(cols) == 1:
            return self.data[cols[0]][rows]
        else:
            if hasattr(self.data[0][rows], '__len__'):
                data = DataFile()
                sec_indices = [-1] * self.nsecs
                for c in cols:
                    data.append(*self.column_head(c))
                    for l in range(self.nsecs):
                        s, i = self.section(c, l+1)
                        if i != sec_indices[l]:
                            data.header[-1].append(s)
                            sec_indices[l] = i
                    data.data[-1] = self.data[c][rows]
                data.nsecs = self.nsecs
                return data
            else:
                return [self.data[i][rows] for i in cols]

    def __setitem__(self, key, value):
        """
        Assign values to data elements specified by slice.

        Parameters
        -----------
        key:
            First key specifies row, (optional) second one the column.
            Columns can be specified by index or name, see index() for details.
        value: DataFile, list, ndarray, float, ...
            Value(s) used to assing to the table elements as specified by `key`.
        """
        rows, cols = self.__setupkey(key)
        if isinstance(value, DataFile):
            if hasattr(self.data[cols[0]][rows], '__len__'):
                for k, c in enumerate(cols):
                    self.data[c][rows] = value.data[k]
            else:
                for k, c in enumerate(cols):
                    self.data[c][rows] = value.data[k][0]
        else:
            if len(cols) == 1:
                self.data[cols[0]][rows] = value
            else:
                if hasattr(self.data[0][rows], '__len__'):
                    for k, c in enumerate(cols):
                        self.data[c][rows] = value[:,k]
                else:
                    for k, c in enumerate(cols):
                        self.data[c][rows] = value[k]

    def __delitem__(self, key):
        """
        Delete data elements or whole columns.

        Parameters
        -----------
        key:
            First key specifies row, (optional) second one the column.
            Columns can be specified by index or name, see index() for details.
            If all rows are selected, then the specified columns are removed from the table.
            Otherwise only data values are removed.
            If all columns are selected than entire rows of data values are removed.
            Otherwise only data values in the specified rows are removed and the columns
            are filled up with missing values.
        """
        rows, cols = self.__setupkey(key)
        if hasattr(self.data[cols[0]][rows], '__len__') and \
           len(self.data[cols[0]][rows]) == len(self.data[cols[0]]) :
            # delete whole columns:
            self.remove(cols)
        else:
            if len(cols) == len(self.header):
                # delete whole row:
                for c in cols:
                    del self.data[c][rows]
                self.shape = (self.rows(), self.columns())
            else:
                # delete part of a row:
                for c in cols:
                    del self.data[c][rows]
                    self.data[c].extend([float('NaN')]*(self.rows()-len(self.data[c])))

    def remove(self, columns):
        """
        Remove columns from the table.

        Parameters
        -----------
        columns: int or string or list of int or string
            Columns can be specified by index or name, see index() for details.

        Raises
        ------
        IndexError:
            If an invalid column was specified.
        """
        # fix columns:
        if not isinstance(columns, (list, tuple, np.ndarray)):
            columns = [ columns ]
        if len(columns) == 0:
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

    def array(self):
        """
        The table data as an numpy array.

        Return
        ------
        data: ndarray
            The data content of the entire table as a 2D numpy array (rows first).
        """
        return np.array(self.data).T

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
        s: string
            A string composed of the header label of the column, an '=' character,
            a textual representation of the data element according to the format
            of the column, followed by the unit of the column.
        """
        col = self.index(col)
        if col is None:
            return ''
        if isinstance(self.data[col][row], float) and m.isnan(self.data[col][row]):
            v = missing
        else:
            u = self.units[col] if self.units[col] != '1' else ''
            v = (self.formats[col] % self.data[col][row]) + u
        return self.header[col][0] + '=' + v

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

    def append_data(self, value, column=None):
        """
        Append data elements to successive table columns.

        Parameters
        ----------
        value: float, int, string, etc. or list thereof or list of list thereof
            Data values to be appended to a column.
            - A single value is simply appened to the specified column of the table.
            - A 1D-list of values is appended to successive columns of the table
              starting with the specified columns.
            - The columns of a 2D-list of values (second index) are appended
              to successive columns of the table starting with the specified columns.
        column: None, int, or string
            The first column to which the data should be appended.
            If None, append to the current column.
            See self.index() for more information on how to specify a column.
        """
        column = self.index(column)
        if column is None:
            column = self.setcol
        if isinstance(value, (list, tuple, np.ndarray)):
            if isinstance(value[0], (list, tuple, np.ndarray)):
                # 2D list, rows first:
                for row in value:
                    for i, val in enumerate(row):
                        self.data[column+i].append(val)
                self.setcol = column + len(value[0])
            else:
                # 1D list:
                for val in value:
                    self.data[column].append(val)
                    column += 1
                self.setcol = column
        else:
            # single value:
            self.data[column].append(value)
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
                self.data[c].append(float('NaN'))
        self.setcol = 0
        self.shape = (self.rows(), self.columns())

    def clear(self):
        """
        Clear content of the table but keep header.
        """
        for c in range(len(self.data)):
            self.data[c] = []
        self.setcol = 0
        self.shape = (self.rows(), self.columns())

    def statistics(self):
        """
        Descriptive statistics of each column.
        """
        ds = DataFile()
        if self.nsecs > 0:
            ds.append_section('statistics')
            for l in range(1,self.nsecs):
                ds.append_section('-')
            ds.append('-', '-', '%-10s')
        else:
            ds.append('statistics', '-', '%-10s')
        ds.header.extend(self.header)
        ds.units.extend(self.units)
        ds.formats.extend(self.formats)
        ds.nsecs = self.nsecs
        ds.hidden = [False] * ds.columns()
        for c in range(self.columns()):
            ds.data.append([])
        ds.append_data('mean', 0)
        ds.append_data('std', 0)
        ds.append_data('min', 0)
        ds.append_data('quartile1', 0)
        ds.append_data('median', 0)
        ds.append_data('quartile3', 0)
        ds.append_data('max', 0)
        ds.append_data('count', 0)
        for c in range(self.columns()):
            ds.append_data(np.nanmean(self.data[c]), c+1)
            ds.append_data(np.nanstd(self.data[c]), c+1)
            ds.append_data(np.nanmin(self.data[c]), c+1)
            q1, m, q3 = np.percentile(self.data[c], [25., 50., 75.])
            ds.append_data(q1, c+1)
            ds.append_data(m, c+1)
            ds.append_data(q3, c+1)
            ds.append_data(np.nanmax(self.data[c]), c+1)
            ds.append_data(np.count_nonzero(~np.isnan(self.data[c])), c+1)
        ds.shape = (ds.rows(), ds.columns())
        return ds
                
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
        if len(columns) == 0:
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

    def write(self, fh=sys.stdout, table_format=None, units=None, number_cols=None,
              missing='-', shrink=True, delimiter=None, format_width=None, sections=None):
        """
        Write the table into a stream.

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
            If None then the format is set to the extension of the filename given by `fh`.
            If `fh` is a stream the format is set to 'dat'.
        units: None or string
            - None: use default of the specified `table_format`.
            - 'row': write an extra row to the table header specifying the units of the columns.
            - 'header': add the units to the column headers.
            - 'none': do not specify the units.
        number_cols: string or None
            Add a row specifying the column index:
            - 'index': indices are integers, first column is 0.
            - 'num': indices are integers, first column is 1.
            - 'aa': use 'a', 'b', 'c', ..., 'z', 'aa', 'ab', ... for indexing
            - 'aa': use 'A', 'B', 'C', ..., 'Z', 'AA', 'AB', ... for indexing
            - None or 'none': do not add a row with column indices
        missing: string
            Indicate missing data by this string.
        shrink: boolean
            If `True` disregard width specified by the format strings,
            such that columns can become narrower.
        delimiter: string
            String or character separating columns, if supported by the `table_format`.
            If None use the default for the specified `table_format`.
        format_width: boolean
            - `True`: set width of column formats to make them align.
            - `False`: set width of column formats to 0 - no unnecessary spaces.
            - None: Use default of the selected `table_format`.
        sections: None or int
            Number of section levels to be printed.
            If `None` use default of selected `table_format`.
        """

        # open file:
        own_file = False
        if not hasattr(fh, 'write'):
            _, ext = os.path.splitext(fh)
            if table_format is None:
                if len(ext) > 1:
                    table_format = self.ext_formats[ext[1:]]
            elif len(ext) == 0:
                fh += '.' + self.extensions[table_format]
            fh = open(fh, 'w')
            own_file = True
        if table_format is None:
            table_format = 'dat'
        # set style:        
        if table_format[0] == 'd':
            format_width = True
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
            format_width = True
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
            number_cols=None
            if units is None:
                units = 'header'
            if format_width is None:
                format_width = False
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
            format_width = True
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
            if units is None:
                units = 'header'
            format_width = True
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
            format_width = False
            begin_str = '<table>\n<thead>\n'
            end_str = '</tbody>\n</table>\n'
            header_start='  <tr class="header">\n    <th align="left"'
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
            if format_width is None:
                format_width = False
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
            if format_width is None:
                format_width = True
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
        if units is None:
            units = 'row'

        # begin table:
        fh.write(begin_str)
        if table_format[0] == 't':
            fh.write('{')
            for f in self.formats:
                if f[1] == '-':
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
            if not shrink:
                if len(f[i0:i1]) > 0:
                    w = int(f[i0:i1])
            widths_pos.append((i0, i1))
            # adapt width to header label:
            hw = len(self.header[c][0])
            if units == 'header':
                hw += 1 + len(self.units[c])
            if w < hw:
                w = hw
            # adapt width to data:
            if f[-1] == 's':
                for v in self.data[c]:
                    if w < len(v):
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
        for c in range(len(self.header)):
            w = widths[c]
            for l in range(min(self.nsecs, sections)):
                if 1+l < len(self.header[c]):
                    if c > 0 and len(self.header[sec_indices[l]][1+l]) > sec_widths[l]:
                        dw = len(self.header[sec_indices[l]][1+l]) - sec_widths[l]
                        nc = c - sec_indices[l]
                        ddw = np.zeros(nc, dtype=int) + dw // nc
                        ddw[:dw % nc] += 1
                        for k in range(nc):
                            widths[sec_indices[l]+k] += ddw[k]
                    sec_widths[l] = 0
                    sec_indices[l] = c
                if sec_widths[l] > 0:
                    sec_widths[l] += len(header_sep)
                sec_widths[l] += w
        # set width of format string:
        formats = []
        for c, (f, w) in enumerate(zip(self.formats, widths)):
            formats.append(f[:widths_pos[c][0]] + str(w) + f[widths_pos[c][1]:])
        # top line:
        if top_line:
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
        # section and column headers:
        nsec0 = self.nsecs-sections
        if nsec0 < 0:
            nsec0 = 0
        for ns in range(nsec0, self.nsecs+1):
            nsec = self.nsecs-ns
            first = True
            last = False
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
                    if not first:
                        fh.write(header_sep)
                    first = False
                    if table_format[0] == 'c':
                        sw -= len(header_sep)*(columns-1)
                    elif table_format[0] == 'h':
                        if columns>1:
                            fh.write(' colspan="%d"' % columns)
                    elif table_format[0] == 't':
                        fh.write('\\multicolumn{%d}{l}{' % columns)
                    fh.write(header_close)
                    hs = self.header[c][nsec]
                    if nsec == 0 and units == 'header':
                        if units and self.units[c] != '1':
                            hs += '/' + self.units[c]
                    if format_width and not table_format[0] in 'th':
                        f = '%%-%ds' % sw
                        fh.write(f % hs)
                    else:
                        fh.write(hs)
                    if table_format[0] == 'c':
                        if not last:
                            fh.write(header_sep*(columns-1))
                    elif table_format[0] == 't':
                        fh.write('}')
            fh.write(header_end)
        # units:
        if units == 'row':
            first = True
            fh.write(header_start)
            for c in range(len(self.header)):
                if self.hidden[c]:
                    continue
                if not first:
                    fh.write(header_sep)
                first = False
                fh.write(header_close)
                if table_format[0] == 't':
                    fh.write('\\multicolumn{1}{l}{%s}' % self.units[c])
                else:
                    if format_width and not table_format[0] in 'h':
                        f = '%%-%ds' % widths[c]
                        fh.write(f % self.units[c])
                    else:
                        fh.write(self.units[c])
            fh.write(header_end)
        # column numbers:
        if number_cols is not None and number_cols not in 'none':
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
                if number_cols == 'num':
                    i = c+1
                aa = index2aa(c, 'a')
                if number_cols == 'AA':
                    aa = index2aa(c, 'A')
                if table_format[0] == 't':
                    if number_cols == 'num' or number_cols == 'index':
                        fh.write('\\multicolumn{1}{l}{%d}' % i)
                    else:
                        fh.write('\\multicolumn{1}{l}{%s}' % aa)
                else:
                    if number_cols == 'num' or number_cols == 'index':
                        if format_width:
                            f = '%%%dd' % widths[c]
                            fh.write(f % i)
                        else:
                            fh.write('%d' % i)
                    else:
                        if format_width:
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
                    if formats[c][1] == '-':
                        fh.write(w*'-' + '|')
                    else:
                        fh.write((w-1)*'-' + ':|')
                fh.write('\n')
            elif table_format[0] == 't':
                fh.write('  \\hline\n')
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
            if table_format[0] == 'h':
                eo = 'even' if k % 2 == 1 else 'odd'
                fh.write('  <tr class "%s">\n    <td' % eo)
            else:
                fh.write(data_start)
            for c, f in enumerate(formats):
                if self.hidden[c]:
                    continue
                if not first:
                    fh.write(data_sep)
                first = False
                if table_format[0] == 'h':
                    if f[1] == '-':
                        fh.write(' align="left"')
                    else:
                        fh.write(' align="right"')
                fh.write(data_close)
                if k >= len(self.data[c]) or \
                   (isinstance(self.data[c][k], float) and m.isnan(self.data[c][k])):
                    if format_width:
                        if f[1] == '-':
                            fn = '%%-%ds' % widths[c]
                        else:
                            fn = '%%%ds' % widths[c]
                        fh.write(fn % missing)
                    else:
                        fh.write(missing)
                else:
                    ds = f % self.data[c][k]
                    if not format_width:
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

            
    def __str__(self):
        stream = StringIO()
        self.write(stream, table_format='out')
        return stream.getvalue()
                

    def load(self, fh, missing='-'):
        """
        Load table from file.

        File type and properties are automatically inferred.

        Parameters
        ----------
        fh: filename or stream
            If not a stream, the file with name `fh` is opened for reading.
        missing: string
            Missing data are indicated by this string.
        """

        def read_key_line(line, sep, table_format):
            if sep is None:
                cols, indices = zip(*[(m.group(0), m.start()) for m in re.finditer(r'( ?[\S]+)+(?=[ ][ ]+|\Z)', line.strip())])
            elif table_format == 'csv':
                cols, indices = zip(*[(c.strip(), i) for i, c in enumerate(line.strip().split(sep)) if len(c.strip()) > 0])
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
                seps = r'[^\s'+re.escape(sep)+']+'
                cols = [m.group(0).strip() for m in re.finditer(seps, line.strip())]
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
                        v = float('NaN')
                    else:
                        strf[k] = True
                        if alld[k] < len(c):
                            alld[k] = len(c)
                        v = c
                self.append_data(v, k)

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
            if len(line) > 0:
                if r'\begin{tabular' in line:
                    table_format='tex'
                    target = key
                    continue
                if table_format == 'tex':
                    if r'\end{tabular' in line:
                        break
                    if r'\hline' in line:
                        if len(key) > 0:
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
                    if len(data) == 0 and len(key) == 0:
                        table_format='ascii'
                        target = key
                        continue
                    elif len(key) == 0:
                        table_format='md'
                        key = data
                        data = []
                        target = data
                        continue
                    elif len(data) == 0:
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
            if s < 0 or len(key) > 0:
                s = 0
            for line in data[s:]:
                cs = line.strip().split(sep)
                if len(cs[0]) == 0:
                    cs = cs[1:]
                if len(cs) > 0 and len(cs[-1]) == 0:
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
        if len(key) == 0 and sep is not None and sep in ',;:\t|':
            table_format = 'csv'
        # read key:
        key_cols = []
        key_indices = []
        for line in key:
            cols, indices = read_key_line(line, sep, table_format)
            key_cols.append(cols)
            key_indices.append(indices)
        if len(key_cols) == 0:
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
        if len(data) == 0:
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
    
    print("Checking datafile module ...")
    print('')

    # setup a table:
    df = DataFile()
    df.append(["data", "partial information", "size"], "m", "%6.2f", [2.34, 56.7, 8.9])
    df.append("full weight", "kg", "%.0f", 122.8)
    df.append_section("complete reaction")
    df.append("speed", "m/s", "%.3g", 98.7)
    df.append("median jitter", "mm", "%.1f", 23)
    df.append("size", "g", "%.2e", 1.234)
    df.append_data(float('NaN'), 1)  # single value
    df.append_data((0.543, 45, 1.235e2)) # remaining row
    df.append_data((43.21, 6789.1, 3405, 1.235e-4), 1) # next row
    a = 0.5*np.arange(1, 6)*np.random.randn(5, 5) + 10.0 + np.arange(5)
    df.append_data(a.T, 0) # rest of table
    df[3:6,'weight'] = [11.0]*3
    
    # write out in all formats:
    for tf in DataFile.formats:
        print('    - `%s`: %s' % (tf, DataFile.descriptions[tf]))
        print('      ```')
        iout = IndentStream(sys.stdout, 4+2)
        df.write(iout, table_format=tf, units=None, number_cols=None, delimiter=None, sections=None)
        print('      ```')
        print('')
        
    # some infos about the data:
    print('data len: %d' % len(df))
    print('data rows: %d' % df.rows())
    print('data columns: %d' % df.columns())
    print('data shape: (%d, %d)' % (df.shape[0],df.shape[1]))
    print('')
    print('column specifications:')
    for c in range(df.columns()):
        print(df.column_spec(c))
    print('keys:')
    print(df.keys())
    print('values:')
    print(df.values())
    print('items:')
    print(df.items())
    print('')

    # sorting:
    print(df)
    df.sort(['weight', 'jitter'], reverse=False)
    print(df)

    # data access:
    print(df)
    print('')
    #print(df[2:,'weight':'reaction>size'])
    print(df[2:5,['size','jitter']])
    print(df[2:5,['size','jitter']].array())
    print('')

    # iterate over columns:
    for a in df:
        print(a)
    print('')

    # single column:    
    print(df[:,'size'])
    print(df.col('size'))

    # single row:    
    print(df[2,:])
    print(df.row(2))

    # table header:
    print(df.table_header())

    # statistics
    print(df.statistics())

    # assignment:
    print(df)
    df[2,3] = 100.0
    df[3:7,'size'] = 3.05+0.1*np.arange(4)
    df[1,:4] = 10.02+2.0*np.arange(4)
    df[3:7,['weight', 'jitter']] = 30.0*(np.ones((4,2))+np.arange(2))
    print(df)
    df[2,3] = df[3,2]
    df[:,'size'] = df.col('weight')
    df[6,:] = df.row(7)
    df[3:7,['speed', 'reaction>jitter']] = df[0:4,['size','speed']]
    print(df)

    # delete:
    del df[3:6, 'weight']
    del df[3:5,:]
    del df[:,'speed']
    print(df)
    df.remove('weight')
    print(df)

    # insert:
    df.insert(1, "s.d.", "m", "%7.3f", np.random.randn(df.rows()))
    print(df)
    
    # contains:
    print('jitter' in df)
    print('velocity' in df)
    print('reaction>size' in df)
