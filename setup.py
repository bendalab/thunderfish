from setuptools import setup, find_packages

exec(open('thunderfish/version.py').read())

long_description = """
# ThunderFish

Algorithms and programs for analysing electric field recordings of
weakly electric fish.

[Documentation](https://bendalab.github.io/thunderfish) |
[API Reference](https://bendalab.github.io/thunderfish/api)

Weakly electric fish generate an electric organ discharge (EOD).  In
wave-type fish the EOD resembles a sinewave of a specific frequency
and with higher harmonics. In pulse-type fish EODs have a distinct
waveform and are separated in time. The thunderfish package provides
algorithms and tools for analysing both wavefish and pulsefish EODs.
"""

setup(
    name = 'thunderfish',
    version = __version__,
    author = 'Jan Benda, Juan F. Sehuanes, Till Raab, Jörg Henninger, Jan Grewe, Fabian Sinz, Liz Weerdmeester',
    author_email = "jan.benda@uni-tuebingen.de",
    description = 'Algorithms and scripts for analyzing recordings of electric fish waveforms.',
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/bendalab/thunderfish",
    license = "GPLv3",
    classifiers = [
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    packages = find_packages(exclude = ['contrib', 'docs', 'tests*']),
    entry_points = {
        'console_scripts': [
            'thunderfish = thunderfish.thunderfish:main',
            'thunderlogger = thunderfish.thunderlogger:main',
            'fishfinder = thunderfish.fishfinder:main',
            'collectfish = thunderfish.collectfish:main',
            'eodexplorer = thunderfish.eodexplorer:main',
        ]},
    python_requires = '>=3.4',
    install_requires = ['sklearn', 'scipy', 'numpy', 'matplotlib', 'audioio'],
)
