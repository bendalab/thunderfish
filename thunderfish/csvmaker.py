import numpy as np


def write_csv(filename, csv_header, data_matrix, format_str='%.3f'):

    """
    Write matrix to csv-file.

    Parameters
    ----------
    filename: string
        String with the path and filename of the csv-file to be produced.
    csv_header: list or array with strings
        List with the names of the columns of the csv-matrix.
    data_matrix: n-d array
        Matrix with the data to be converted into a csv-file.
    format_str: string
        Format string for writing data_matrix (decimals)
    """

    # Check if header has same row length as data_matrix
    if len(data_matrix) > 0 and len(csv_header) != len(data_matrix[0]):
        raise ValueError('The length of the header does not match the length of the data matrix!')

    with open(filename, 'wb') as fin:
        fin.write(str.encode(','.join(csv_header)))
        fin.write(str.encode('\n'))
        for row_id in range(len(data_matrix)):

            fin.write(str.encode(','.join([format_str % e for e in data_matrix[row_id]])))  # convert to strings!
            fin.write(str.encode('\n'))
    pass


if __name__ == '__main__':
    print("\nChecking csvmaker module ...")

    header = ['fundamental frequency', 'dB']
    data = [[1.2, 2.3], [3.4, 4.5], [5.6, 6.7]]
    filename = 'csvmaker_testfile.csv'
    write_csv(filename, header, data)
    
    print('\ncsv_file created in %s' % filename)
