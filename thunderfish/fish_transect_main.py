#!/usr/raab_code/fish_transect/sorting_tools/python
__author__ = 'raab'
import sys
import os
import glob
import numpy as np
from IPython import embed

def main():
    input = sys.argv[1]
    file_type = input.split('/')[-1].split('.')[-1]

    if file_type == '':
        print ''
        print 'Input is a directory !!!'
        print ''

        if not os.path.exists('files.txt'):
            os.system('ls %s > files.txt' %input)

        txt_file =glob.glob('files.txt')
        f = open('%s' %txt_file[0], 'r')

        # for loop with doing till_juan_fishfinder.py with each file
        for file in f:
            os.system('python till_juan_fishfinder2.2.py %s%s' %(input, file))
        os.remove('files.txt')

    else:
        print ''
        print 'Input is a ', file_type, 'file !!!'
        print ''

        os.system('python till_juan_fishfinder2.2.py %s' %input)

    # create_plots.py
    os.system('python create_plots2.py')

    # create_tex.py

    # create script with allgemeininfos xD

    # os.system('python create_tex2.py')

if __name__ == '__main__':
    main()