from setuptools import setup, find_packages

exec(open('thunderfish/version.py').read())

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='thunderfish',
    version=__version__,
    author='Jan Benda, Juan F. Sehuanes, Till Raab, Joerg Henninger, Jan Grewe, Fabian Sinz, Liz Weerdmeester',
    author_email="jan.benda@uni-tuebingen.de",
    description='Algorithms and scripts for analyzing recordings of electric fish waveforms.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bendalab/thunderfish",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    entry_points={
        'console_scripts': [
            'thunderfish = thunderfish.thunderfish:main',
            'fishfinder = thunderfish.fishfinder:main',
            'collectfish = thunderfish.collectfish:main',
            'eodexplorer = thunderfish.eodexplorer:main',
        ]},
    python_requires='>=3.4',
    install_requires=['sklearn', 'scipy', 'numpy', 'matplotlib', 'audioio'],
)
