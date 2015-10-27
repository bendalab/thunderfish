__author__ = 'juan'

import numpy as np
import glob
from IPython import embed

if __name__ == '__main__':

    ## Parallel jobs for plotting best window in Colombia cano Rubiano recordings
    files = glob.glob('../../joint_fishfinder_data/recordings_Colombia/data/recordings_cano_rubiano_RAW/*')
    task_list = ['python test_best_window.py ' + e + ' colombia_cano_rubiano' for e in files]
    task_list = sorted(task_list)
    np.savetxt('parallel_test_best_window_colombia.txt', task_list, fmt="%s")  # Works fine so far...

    ## Parallel jobs for plotting best window in recordings_Panama2014
    files = glob.glob('../../joint_fishfinder_data/recordings_Panama2014/MP3_1/*/*MP3')
    task_list = ['python test_best_window.py ' + e + ' panama_2014' for e in files]
    task_list = sorted(task_list)
    np.savetxt('parallel_test_best_window_panama_2014.txt', task_list, fmt="%s")  # Works fine so far...