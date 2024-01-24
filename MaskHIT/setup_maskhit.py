from setuptools import setup, find_packages

# Setup script for the MaskHIT package
# $ python setup_maskhit.py install
# To install this package in development mode, use:
# $ python setup_maskhit.py develop

setup(
   name='maskhit',
   version='1.0',
   description='MaskHIT',
   author='Shuai Jiang, Naofumi Tomita',
   packages=find_packages()
   # install_requires=[], #external packages as dependencies
)

# Notes for uninstallation:
# To uninstall the MaskHIT package, use the command:
# $ pip uninstall MaskHIT

# For earlier versions of the package named 'maskhit', use:
# $ pip uninstall maskhit

# For unisntalling develop version, use the commands above and
# remove a directory, called `maskhit.egg-info`