#!/usr/bin/env python
import os
import subprocess as sp

try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension


def read(fname):
    try:
        return open(os.path.join(os.path.dirname(__file__), fname)).read()
    except:
        return ''


version = sp.run(
    'python mcs_prime/version.py',
    check=True,
    shell=True,
    stdout=sp.PIPE,
    stderr=sp.PIPE,
    encoding='utf8',
).stdout


setup(
    name='mcs_prime',
    version=version,
    description='COSMIC package containing tools and analysis',
    license='LICENSE',
    long_description=read('README.md'),
    author='Mark Muetzelfeldt',
    author_email='mark.muetzelfeldt@reading.ac.uk',
    maintainer='Mark Muetzelfeldt',
    maintainer_email='mark.muetzelfeldt@reading.ac.uk',
    packages=[
        'mcs_prime',
    ],
    scripts=[],
    python_requires='>=3.10',
    # These should all be met if you use the conda_env in envs.
    install_requires=[
        'cartopy',
        'cdsapi',
        'matplotlib',
        'numpy',
        'pandas',
        'shapely',
        'xarray',
    ],
    url='https://github.com/markmuetz/mcs_prime',
    classifiers=[
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Atmospheric Science',
    ],
    keywords=['MCS', 'parametrization', 'analysis'],
)
