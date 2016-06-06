__author__ = 'juan'

import numpy as np
import glob
import sys

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print('\nError! You forgot to type the the path of the folder where your recordings are as first argument!\n')
        sys.exit(2)

    # Create parallel jobs for user specific folder!
    rec_dir = sys.argv[1]
    job_name = 'parallel_jobs_for_' + sys.argv[-1].split('/')[-2] + '_' + sys.argv[-1].split('/')[-1] + '.txt'
    if rec_dir[-1] != '/':
        rec_dir += '/'

    rec_files = glob.glob(rec_dir + '*')
    task_list = ['python thunderfish.py ' + e for e in rec_files]
    if len(task_list) > 0:
        np.savetxt(job_name, task_list, fmt="%s")

    print('\nTask-list terminated. Tasks stored in %s\n' % job_name)
