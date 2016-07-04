__author__ = 'juan'

import numpy as np
import glob
import sys

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print('Error! You forgot to type the the path of the folder where your recordings are as first argument!')
        sys.exit(2)

    # create parallel jobs for user specific folder:
    folders = sys.argv[-1].split(os.sep)
    job_name = 'parallel_jobs_for_' + folders[-2] + '_' + folders[-1] + '.txt'

    rec_dir = sys.argv[1]
    rec_files = glob.glob(os.path.join(rec_dir, '*')
    
    task_list = ['python thunderfish.py ' + e for e in rec_files]
    if len(task_list) > 0:
        np.savetxt(job_name, task_list, fmt="%s")

    print('Task-list terminated. Tasks stored in %s' % job_name)
