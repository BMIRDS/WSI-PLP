from setuptools import setup

# $ python setup.py develop

setup(
   name='popp_slide',
   version='1.0',
   description='Whole Slide Image based Patient Level Prediction',
   author='Shuai Jiang, Diana Song, Naofumi Tomita',
   packages=['data', 'model', 'options', 'utils'],  #same as name
   # install_requires=[], #external packages as dependencies
)
