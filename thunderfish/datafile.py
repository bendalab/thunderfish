"""
# Data file
class DataFile for reading and writing of data tables.
"""

import sys
import re
import math as m
import numpy as np
if sys.version_info[0] < 3:
    from io import BytesIO as StringIO
else:
    from io import StringIO
    

class DataFile:
    """
    Tables of data with a rich header.

    Indexing
    --------
    With the [] operator: rows first, columns second
    len() returns number of rows.

    But: iterating over the table goes over columns!

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
    extensions = {'dat': 'dat', 'ascii': 'txt', 'csv': 'csv', 'rtai': 'dat', 'md': 'ms', 'tex': 'tex', 'html': 'html'}
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
        self.indices = None
        if filename is not None:
            with open(filename, 'r') as sf:
                self.load(sf)

    def add_section(self, label):
        """
        Add a new section to the table header.

        Each column of the table has a header label. Columns can be
        grouped into sections. Sections can be nested arbitrarily.

        Parameters
        ----------
        label: string
            The name of the section.
        """
        if self.addcol >= len(self.data):
            self.header.append([label])
            self.units.append('')
            self.formats.append('')
            self.hidden.append(False)
            self.data.append([])
        else:
            self.header[self.addcol] = [label] + self.header[self.addcol]
        if self.nsecs < len(self.header[self.addcol]):
            self.nsecs = len(self.header[self.addcol])
        self.addcol = len(self.data)-1
        self.shape = (self.rows(), self.columns())
        return self.addcol
        
    def add_column(self, label, unit, formats, value=None):
        """
        Add a new column to the table header.

        Parameters
        ----------
        label: string
            The name of the column.
        unit: string
            The unit of the column contents.
        formats: string
            The C-style format string used for printing out the column content, e.g.
            '%g', '%.2f', '%s', etc.
        val: None, float, int, string, etc.
            If not None, data value to be set as the first data element of the new column.
        """
        if self.addcol >= len(self.data):
            self.header.append([label])
            self.formats.append(formats)
            self.units.append(unit)
            self.hidden.append(False)
            self.data.append([])
        else:
            self.header[self.addcol] = [label] + self.header[self.addcol]
            self.units[self.addcol] = unit
            self.formats[self.addcol] = formats
        if value is not None:
            self.data[-1].append(value)
        self.addcol = len(self.data)
        self.shape = (self.rows(), self.columns())
        return self.addcol-1

    def section(self, column, level):
        """
        The section name of a specified column.

        Parameters
        ----------
        col: None, int, or string
            A specification of a column.
            See self.column_index() for more information on how to specify a column.
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
            raise IndexError
        column = self.column_index(column)
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
            See self.column_index() for more information on how to specify a column.
        level: int
            The level of the section to be set. The column label itself is level=0.
        """
        column = self.column_index(column)
        self.header[column][level] = label
        return column

    def label(self, column):
        """
        The column name.

        Parameters
        ----------
        col: None, int, or string
            A specification of a column.
            See self.column_index() for more information on how to specify a column.

        Returns
        -------
        name: string
            The column label.
        """
        column = self.column_index(column)
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
            See self.column_index() for more information on how to specify a column.
        """        
        column = self.column_index(column)
        self.header[column][0] = label
        return column

    def unit(self, column):
        """
        The unit of the column.

        Parameters
        ----------
        col: None, int, or string
            A specification of a column.
            See self.column_index() for more information on how to specify a column.

        Returns
        -------
        unit: string
            The unit.
        """
        column = self.column_index(column)
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
            See self.column_index() for more information on how to specify a column.
        """
        column = self.column_index(column)
        self.units[column] = unit
        return column

    def format(self, column):
        """
        The format string of the column.

        Parameters
        ----------
        col: None, int, or string
            A specification of a column.
            See self.column_index() for more information on how to specify a column.

        Returns
        -------
        format: string
            The format string.
        """
        column = self.column_index(column)
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
            See self.column_index() for more information on how to specify a column.
        """
        column = self.column_index(column)
        self.formats[column] = format
        return column

    def column_spec(self, column):
        """
        Full specification of a column.

        Parameters
        ----------
        column: int or string
            Specifies the column.
            See self.column_index() for more information on how to specify a column.

        Returns
        -------
        s: string
            Full specification of the column by all its section names and its header name.
        """
        c = self.column_index(column)
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
            See self.column_index() for more information on how to specify a column.

        Returns
        -------
        name: string
            The column label.
        unit: string
            The unit.
        format: string
            The format string.
        """
        column = self.column_index(column)
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
            data.add_column(*self.column_head(c))
            for l in range(self.nsecs):
                s, i = self.section(c, l+1)
                if i != sec_indices[l]:
                    data.header[-1].append(s)
                    sec_indices[l] = i
        data.nsecs = self.nsecs
        return data
        
    def __len__(self):
        """
        The number of rows.
        
        Returns
        -------
        rows: int
            The number of rows contained in the table.
        """
        return self.rows()

    def rows(self):
        """
        The number of rows.
        
        Returns
        -------
        rows: int
            The number of rows contained in the table.
        """
        return max(map(len, self.data))

    def columns(self):
        """
        The number of columns.
        
        Returns
        -------
        columns: int
            The number of columns contained in the table.
        """
        return len(self.header)

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
            data.add_column(*self.column_head(c))
            for l in range(self.nsecs):
                s, i = self.section(c, l+1)
                if i != sec_indices[l]:
                    data.header[-1].append(s)
                    sec_indices[l] = i
            data.data[-1] = [self.data[c][index]]
        data.nsecs = self.nsecs
        return data

    def col(self, column):
        """
        A single column of the table.

        Parameters
        ----------
        column: None, int, or string
            The column to be returned.
            See self.column_index() for more information on how to specify a column.

        Return
        ------
        table: DataFile
            A DataFile object with a single column.
        """
        data = DataFile()
        c = self.column_index(column)
        data.add_column(*self.column_head(c))
        data.data = [self.data[c]]
        data.nsecs = 0
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
            start = self.column_index(cols.start)
            stop = self.column_index(cols.stop)
            cols = slice(start, stop, cols.step)
            cols = range(self.columns())[cols]
        elif type(cols) is list or type(cols) is tuple or type(cols) is np.ndarray:
            cols = [self.column_index(inx) for inx in cols]
        else:
            cols = [self.column_index(cols)]
        return rows, cols

    def __getitem__(self, key):
        """
        Data elements specified by slice.

        Parameters
        -----------
        key:
            First key specifies row, (optional) second one the column.
            Columns can be specified by index or name, see column_index() for details.

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
                    data.add_column(*self.column_head(c))
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
            Columns can be specified by index or name, see column_index() for details.
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
            Columns can be specified by index or name, see column_index() for details.
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
            for c in cols:
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
            See self.column_index() for more information on how to specify a column.
        missing: string
            String indicating non-existing data elements.

        Returns
        -------
        s: string
            A string composed of the header label of the column, an '=' character,
            a textual representation of the data element according to the format
            of the column, followed by the unit of the column.
        """
        col = self.column_index(col)
        if col is None:
            return ''
        if isinstance(self.data[col][row], float) and m.isnan(self.data[col][row]):
            v = missing
        else:
            u = self.units[col] if self.units[col] != '1' else ''
            v = (self.formats[col] % self.data[col][row]) + u
        return self.header[col][0] + '=' + v

    def _find_col(self, ss, si, minns, maxns, c0, strict=True):
        """
        Helper function for finding column indices from textual column specifications.
        """
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

    def find_col(self, column):
        """
        Find the start and end index of a column specification.
        
        Parameters
        ----------
        column: None, int, or string
            A specification of a column.
            See self.column_index() for more information on how to specify a column.

        Returns
        -------
        c0: int or None
            A valid column index or None that is specified by `column`.
        c1: int or None
            A valid column index or None of the column following the range specified
            by `column`.
        """
        if column is None:
            return None, None
        if not isinstance(column, int) and column.isdigit():
            column = int(column)
        if isinstance(column, int):
            if column >= 0 and column < len(self.formats):
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
        c0, c1, ns, si = self._find_col(ss, si0, 0, maxns, 0, True)
        if c0 is None and c1 is not None:
            c0, c1, ns, si = self._find_col(ss, si, ns, maxns, c1, False)
        return c0, c1
    
    def column_index(self, column):
        """
        The index of a column.
        
        Parameters
        ----------
        column: None, int, or string
            A specification of a column.
            - None: no column is specified
            - int: the index of the column (first column is zero), e.g. `column_index(2)`.
            - a string representing an integer is converted into the column index,
              e.g. `column_index('2')`
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
            See self.column_index() for more information on how to specify a column.

        Returns
        -------
        contains: bool
            True if `column` specifies an existing column key.
        """
        return self.column_index(column) is not None

    def add_value(self, val, column=None):
        """
        Add a single data element to a specific column of the table.

        Parameters
        ----------
        val: float, int, string, etc.
            Data value to be appended to a column.
        column: None, int, or string
            The column to which the data element should be appended.
            If None, append to the current column.
            See self.column_index() for more information on how to specify a column.
        """
        column = self.column_index(column)
        if column is None:
            column = self.setcol
        self.data[column].append(val)
        self.setcol = column+1
        self.shape = (self.rows(), self.columns())

    def add_data(self, data, column=None):
        """
        Add a list of data elements to the table.

        The data values are appended to successive columns, starting at `column`.

        Parameters
        ----------
        data: list of float, int, string, etc.
            Data values to be added to the table.
        column: None, int, or string
            The column to which the first data element should be appended.
            If None, append to the current column.
            See self.column_index() for more information on how to specify a column.
            The remaining data elements will be added to the subsequent columns.
        """
        for val in data:
            self.add_value(val, column)
            column = None

    def set_column(self, column):
        """
        Set the column where to add data.

        Parameters
        ----------
        column: int or string
            The column to which data elements should be appended.
            See self.column_index() for more information on how to specify a column.

        Raises
        ------
        IndexError:
            If an invalid column was specified.
        """
        col = self.column_index(column)
        if col is None:
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
            ds.add_section('statistics')
            for l in range(1,self.nsecs):
                ds.add_section('-')
            ds.add_column('-', '-', '%-10s')
        else:
            ds.add_column('statistics', '-', '%-10s')
        ds.header.extend(self.header)
        ds.units.extend(self.units)
        ds.formats.extend(self.formats)
        ds.nsecs = self.nsecs
        ds.hidden = [False] * ds.columns()
        for c in range(self.columns()):
            ds.data.append([])
        ds.add_value('mean', 0)
        ds.add_value('std', 0)
        ds.add_value('min', 0)
        ds.add_value('quartile1', 0)
        ds.add_value('median', 0)
        ds.add_value('quartile3', 0)
        ds.add_value('max', 0)
        ds.add_value('count', 0)
        for c in range(self.columns()):
            ds.add_value(np.nanmean(self.data[c]), c+1)
            ds.add_value(np.nanstd(self.data[c]), c+1)
            ds.add_value(np.nanmin(self.data[c]), c+1)
            q1, m, q3 = np.percentile(self.data[c], [25., 50., 75.])
            ds.add_value(q1, c+1)
            ds.add_value(m, c+1)
            ds.add_value(q3, c+1)
            ds.add_value(np.nanmax(self.data[c]), c+1)
            ds.add_value(np.count_nonzero(~np.isnan(self.data[c])), c+1)
        ds.shape = (ds.rows(), ds.columns())
        return ds
                
    def sort(self, columns):
        """
        Sort the table rows.

        Generate an index list for the rows that is used by write() when writing the table.
        This only affects the output via the write() function, the data elements are
        not rearranged.

        Parameters
        ----------
        columns: int or string or list of int or string
            A column specifier or a list of column specifiers of the columns
            to be sorted.
        """
        if type(columns) is not list and type(columns) is not tuple:
            columns = [ columns ]
        if len(columns) == 0:
            return
        self.indices = range(len(self.data[0]))
        for col in reversed(columns):
            rev = False
            if len(col) > 0 and col[0] in '^!':
                rev = True
                col = col[1:]
            c = self.column_index(col)
            if c is None:
                print('sort column ' + col + ' not found')
                continue
            self.indices = sorted(self.indices, key=self.data[c].__getitem__, reverse=rev)

    def hide(self, column):
        """
        Hide a column or a range of columns.

        Hidden columns will not be printed out by the write() function.

        Parameters
        ----------
        column: int or string
            The column to be hidden.
            See self.column_index() for more information on how to specify a column.
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
            See self.column_index() for more information on how to specify a column.
        """
        c0, c1 = self.find_col(column)
        if c0 is not None:
            for c in range(c0, c1):
                self.hidden[c] = False

    def adjust_columns(self, missing='-'):
        """
        Adjust the format of each column to the maximum width of its data elements.

        Parameters
        ----------
        missing: string
            String indicating missing data.
        """
        for c, f in enumerate(self.formats):
            w = 0
            # extract width from format:
            i0 = 1
            if f[1] == '-' :
                i0 = 2
            i1 = f.find('.')
            if len(f[i0:i1]) > 0:
                w = int(f[i0:i1])
            # adapt width to header:
            if w < len(self.header[c][0]):
                w = len(self.header[c][0])
            # adapt width to data:
            if f[-1] == 's':
                for v in self.data[c]:
                    if w < len(v):
                        w = len(v)
            else:
                for v in self.data[c]:
                    if isinstance(v, float) and m.isnan(v):
                        s = missing
                    else:
                        s = f % v
                    if w < len(s):
                        w = len(s)
            # set width of format string:
            f = f[:i0] + str(w) + f[i1:]
            self.formats[c] = f

    def write(self, df=sys.stdout, table_format='dat',
              units="row", number_cols=None, missing='-'):
        """
        Write the table into a stream.

        Parameters
        ----------
        df: stream
            Stream to be used for writing.
        table_format: string
            The format to be used for output.
            One of "dat", "ascii", "rtai", "csv", "md", "html", "tex".
        units: string
            - "row": write an extra row to the table header specifying the units of the columns.
            - "header": add the units to the column headers.
            - "none": do not specify the units.
        number_cols: string or None
            If not None, add a row that specifies the colum indices ('index' starting with 0),
            column number ('num' starting with 1), or column letters ('aa' or 'AA').
        missing: string
            Indicate missing data by this string.
        """
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
        if table_format[0] == 'a':
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
        elif table_format[0] == 'c':
            # csv according to http://www.ietf.org/rfc/rfc4180.txt :
            number_cols=None
            if units == "row":
                units = "header"
            format_width = False
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
        elif table_format[0] == 'm':
            number_cols=None
            if units == "row":
                units = "header"
            format_width = True
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
        elif table_format[0] == 't':
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

        # begin table:
        df.write(begin_str)
        if table_format[0] == 't':
            df.write('{')
            for f in self.formats:
                if f[1] == '-':
                    df.write('l')
                else:
                    df.write('r')
            df.write('}\n')
        # retrieve column widths:
        widths = []
        for f in self.formats:
            i0 = 1
            if f[1] == '-' :
                i0 = 2
            i1 = f.find('.')
            if len(f[i0:i1]) > 0:
                widths.append(int(f[i0:i1]))
            else:
                widths.append(1)
        # top line:
        if top_line:
            if table_format[0] == 't':
                df.write('  \\hline\n')
            else:
                first = True
                df.write(header_start.replace(' ', '-'))
                for c in range(len(self.header)):
                    if self.hidden[c]:
                        continue
                    if not first:
                        df.write('-'*len(header_sep))
                    first = False
                    df.write(header_close)
                    w = widths[c]
                    df.write(w*'-')
                df.write(header_end.replace(' ', '-'))
        # section and column headers:
        nsec0 = 0
        if table_format[0] in 'cm':
            nsec0 = self.nsecs
        for ns in range(nsec0, self.nsecs+1):
            nsec = self.nsecs-ns
            first = True
            df.write(header_start)
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
                    if columns == 0:
                        continue
                    if not first:
                        df.write(header_sep)
                    first = False
                    if table_format[0] == 'h':
                        if columns>1:
                            df.write(' colspan="%d"' % columns)
                    elif table_format[0] == 't':
                        df.write('\\multicolumn{%d}{l}{' % columns)
                    df.write(header_close)
                    hs = self.header[c][nsec]
                    if nsec == 0 and units == "header":
                        if units and self.units[c] != '1':
                            hs += '/' + self.units[c]
                    if format_width:
                        f = '%%-%ds' % sw
                        df.write(f % hs)
                    else:
                        df.write(hs)
                    if table_format[0] == 't':
                        df.write('}')
            df.write(header_end)
        # units:
        if units == "row":
            first = True
            df.write(header_start)
            for c in range(len(self.header)):
                if self.hidden[c]:
                    continue
                if not first:
                    df.write(header_sep)
                first = False
                df.write(header_close)
                if table_format[0] == 't':
                    df.write('\\multicolumn{1}{l}{%s}' % self.units[c])
                else:
                    if format_width:
                        f = '%%-%ds' % widths[c]
                        df.write(f % self.units[c])
                    else:
                        df.write(self.units[c])
            df.write(header_end)
        # column numbers:
        if number_cols is not None:
            first = True
            df.write(header_start)
            for c in range(len(self.header)):
                if self.hidden[c]:
                    continue
                if not first:
                    df.write(header_sep)
                first = False
                df.write(header_close)
                i = c
                if number_cols == 'num':
                    i = c+1
                aa = index2aa(c, 'a')
                if number_cols == 'AA':
                    aa = index2aa(c, 'A')
                if table_format[0] == 't':
                    if number_cols == 'num' or number_cols == 'index':
                        df.write('\\multicolumn{1}{l}{%d}' % i)
                    else:
                        df.write('\\multicolumn{1}{l}{%s}' % aa)
                else:
                    if number_cols == 'num' or number_cols == 'index':
                        if format_width:
                            f = '%%%dd' % widths[c]
                            df.write(f % i)
                        else:
                            df.write("%d" % i)
                    else:
                        if format_width:
                            f = '%%%ds' % widths[c]
                            df.write(f % aa)
                        else:
                            df.write(aa)
            df.write(header_end)
        # header line:
        if header_line:
            if table_format[0] == 'm':
                df.write('|')
                for c in range(len(self.header)):
                    if self.hidden[c]:
                        continue
                    w = widths[c]+2
                    if self.formats[c][1] == '-':
                        df.write(w*'-' + '|')
                    else:
                        df.write((w-1)*'-' + ':|')
                df.write('\n')
            elif table_format[0] == 't':
                df.write('  \\hline\n')
            else:
                first = True
                df.write(header_start.replace(' ', '-'))
                for c in range(len(self.header)):
                    if self.hidden[c]:
                        continue
                    if not first:
                        df.write(header_sep.replace(' ', '-'))
                    first = False
                    df.write(header_close)
                    w = widths[c]
                    df.write(w*'-')
                df.write(header_end.replace(' ', '-'))
        # start table data:
        if table_format[0] == 'h':
            df.write('</thead>\n<tbody>\n')
        # data:
        if len(self.data) == 0:
            self.indices = []
        elif self.indices is None or len(self.indices) != len(self.data[0]):
            self.indices = range(len(self.data[0]))
        for i, k in enumerate(self.indices):
            first = True
            if table_format[0] == 'h':
                eo = "even" if i % 2 == 1 else "odd"
                df.write('  <tr class"%s">\n    <td' % eo)
            else:
                df.write(data_start)
            for c, f in enumerate(self.formats):
                if self.hidden[c]:
                    continue
                if not first:
                    df.write(data_sep)
                first = False
                if table_format[0] == 'h':
                    if f[1] == '-':
                        df.write(' align="left"')
                    else:
                        df.write(' align="right"')
                df.write(data_close)
                if isinstance(self.data[c][k], float) and m.isnan(self.data[c][k]):
                    if format_width:
                        if f[1] == '-':
                            fn = '%%-%ds' % widths[c]
                        else:
                            fn = '%%%ds' % widths[c]
                        df.write(fn % missing)
                    else:
                        df.write(missing)
                else:
                    ds = f % self.data[c][k]
                    if not format_width:
                        ds = ds.strip()
                    df.write(ds)
            df.write(data_end)
        # bottom line:
        if bottom_line:
            if table_format[0] == 't':
                df.write('  \\hline\n')
            else:
                first = True
                df.write(header_start.replace(' ', '-'))
                for c in range(len(self.header)):
                    if self.hidden[c]:
                        continue
                    if not first:
                        df.write('-'*len(header_sep))
                    first = False
                    df.write(header_close)
                    w = widths[c]
                    df.write(w*'-')
                df.write(header_end.replace(' ', '-'))
        # end table:
        df.write(end_str)

    def __str__(self):
        stream = StringIO()
        self.write(stream, table_format='dat')
        return stream.getvalue()

    def _read_line(self, line, sep):
        """
        Helper function for parsing a data line.
        """
        if sep is None:
            cols = [m.group(0) for m in re.finditer(r'\S+', line.strip())]
        else:
            seps = r'[^\s'+re.escape(sep)+']+'
            cols = [m.group(0) for m in re.finditer(seps, line.strip())]
        for k, c in enumerate(cols):
            cols[k] = c.strip()
        return cols

    def _read_key_line(self, line, sep, table_format):
        """
        Helper function for parsing a key line.
        """
        if sep is None:
            cols, indices = zip(*[(m.group(0), m.start()) for m in re.finditer(r'\S+', line.strip())])
        else:
            seps = r'[^\s'+re.escape(sep)+']+'
            cols, indices = zip(*[(m.group(0), m.start()) for m in re.finditer(seps, line.strip())])
        colss = []
        if table_format == 'tex':
            indices = []
            i = 0
            for c in cols:
                if 'multicolumn' in c:
                    fields = c.split('{')
                    n = int(fields[1].strip().rstrip('}').rstrip())
                    colss.append(fields[3].strip().rstrip('}').rstrip())
                    indices.append(i)
                    i += n
                else:
                    colss.append(c.strip())
                    indices.append(i)
                    i += 1
        else:
            for c in cols:
                colss.append(c.strip())
        return colss, indices

    def _col_format(self, line, sep, post, precd, alld, numc, exped, fixed, strf, missing):
        """
        Helper function for adding data and analysing format.
        """
        cols = self._read_line(line, sep)
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
            self.add_value(v, k)
                

    def load(self, sf, table_format='dat',
              units="row", number_cols=None, missing='-'):
        """
        Load data from file stream. 
        """
        # read inital lines of file:
        key = []
        data = []
        target = data
        comment = False
        table_format='dat'        
        for line in sf:
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
        # find column separator of data:
        col_seps = ['|', ',', '&', None]
        colstd = np.zeros(len(col_seps))
        colnum = np.zeros(len(col_seps), dtype=int)
        for k, sep in enumerate(col_seps):
            cols = []
            for line in data:
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
            columns = 1
        else:
            ci = np.where(np.array(colnum)>1.5)[0]
            ci = ci[np.argmin(colstd[ci])]
            sep = col_seps[ci]
            colnum = int(colnum[ci])
        # fix key:
        if sep == ',' and len(key) == 0:
            table_format == 'csv'
            key = [data.pop(0)]
        # read key:
        kr = len(key)-1
        cols, indices = self._read_key_line(key[kr], sep, table_format)
        # check for key with column indices:
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
            cols, indices = self._read_key_line(key[kr], sep, table_format)
        # check for unit line:
        units = None
        if kr > 0:
            kr -= 1
            cols0, indices0 = self._read_key_line(key[kr], sep, table_format)
            if len(cols0) == len(cols):
                units = cols
                cols = cols0
                indices = indices0
        # units may be part of the label:
        if units is None:
            labels = []
            units = []
            for c in cols:
                lu = c.split('/')
                if len(lu) >= 2:
                    labels.append(lu[0].strip())
                    units.append('/'.join(lu[1:]).strip())
                else:
                    labels.append(c)
                    units.append('')
        else:
            labels = cols
        for k in range(colnum):
            self.add_column(labels[k], units[k], '%g')
        # read in sections:
        while kr > 0:
            kr -= 1
            sec_cols, sec_indices = self._read_key_line(key[kr], sep, table_format)
            for sec_label, sec_inx in zip(sec_cols, sec_indices):
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
            self._col_format(line, sep, post, precd, alld, numc, exped, fixed, strf, missing)
        # read remaining data:
        for line in sf:
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
            self._col_format(line, sep, post, precd, alld, numc, exped, fixed, strf, missing)
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
    df.add_section("data")
    df.add_section("info")
    df.add_column("size", "m", "%6.2f", 2.34)
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
    
    # write out in all formats:
    for tf in DataFile.formats:
        print('    - `%s`: %s' % (tf, DataFile.descriptions[tf]))
        print('      ```')
        iout = IndentStream(sys.stdout, 4+2)
        df.write(iout, table_format=tf)
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
    print(df.keys())
    print('')

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
    
    # contains:
    print('jitter' in df)
    print('velocity' in df)
    print('reaction>size' in df)
