from setuptools import setup, find_packages

exec(open('thunderfish/version.py').read())

setup(
    version = __version__,
    packages = ['thunderfish']
    #packages = find_packages(exclude = ['docs', 'tests', 'site'])
)
