from setuptools import setup
from Cython.Build import cythonize

import subprocess

try:  # Try to create an rst long_description from README.md
    args = "pandoc", "--to", "rst", "README.md"
    long_description = subprocess.check_output(args)
    long_description = long_description.decode()
except Exception as error:
    print("README.md conversion to reStructuredText failed. Error:\n",
          error, "Setting long_description to None.")
    long_description = None

setup(
    name='starburst',
    version='0.1.0',
    packages=['starburst'],
    url='https://github.com/Palpatineli/starburst',
    download_url='https://github.com/Palpatineli/starburst/archive/0.1.0.tar.gz',
    license='GPLv3',
    author='Keji Li',
    author_email='mail@keji.li',
    install_requires=["opencv_python>=3.0", "numpy", "scipy", "tiffreader", "tqdm", "uifunc"],
    tests_require=["pytest"],
    entry_points={
        "gui_scripts": ['starburst=starburst.main:main']
    },
    description='identify and measure pupil using starburst algorithm',
    long_description=long_description,
    classifiers=['Development Status :: 4 - Beta',
                 'Programming Language :: Python :: 3']
)
