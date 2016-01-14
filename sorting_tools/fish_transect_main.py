#!/usr/raab_code/fish_transect/sorting_tools/python
__author__ = 'raab'
import sys
import os
import glob
import numpy as np
from IPython import embed

def main():
    filepath = sys.argv[1]

    # load txt file; filenames in line
    if not os.path.exists('files.txt'):
        os.system('ls %s > files.txt' %filepath)

    txt_file =glob.glob('files.txt')
    f = open('%s' %txt_file[0], 'r')

    # temporary file untill we got a file with pulse fundamentals
    if not os.path.exists('fish_pulse.npy'):
        np.save('fish_pulse.npy', np.array([]))

    # for loop with doing till_juan_fishfinder.py with each file
    for file in f:
        os.system('python till_juan_fishfinder2.2.py %s%s' %(filepath, file))
    os.remove('files.txt')

    # create_plots.py
    os.system('python create_plots2.py')

    # create_tex.py
    os.system('python create_tex2.py')

if __name__ == '__main__':
    main()