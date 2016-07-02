from distutils.core import setup
from setuptools import find_packages

setup(name='thunderfish',
      version='0.5.0',     # see http://semver.org/
      packages=find_packages(exclude=['contrib', 'doc', 'tests*']),
      description='Algorithms and scripts for analyzing recordings of e-fish electric fields.',
      author='Jan Benda, Juan F. Sehuanes, Till Raab, Joerg Henninger, Jan Grewe, Fabian Sinz',
      requires=['numpy', 'matplotlib', 'audioio']
      )
