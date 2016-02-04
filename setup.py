from distutils.core import setup

setup(name='thunderfish',
      version='0.5',
      packages=['thunderfish'],
      description='Python scripts for analyzing recordings of e-fish electric fields.',
      author='Juan F. Sehuanes, Till Raab, Fabian Sinz, Jan Benda, Joerg Henninger, Jan Grewe',
      requires=['matplotlib', 'numpy', 'scipy', 'seaborn', 'IPython', 'audioread']
      )
