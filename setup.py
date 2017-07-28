import os
from setuptools import setup

os.path.split(__file__)[0] + '/README'

with open('README') as readme:
    readme_contents = readme.read()

setup(
    name='quilted',
    version='0.1',
    long_description=readme_contents,
    packages=['quilted', 'quilted.tests'],
    include_package_data=True,
    zip_safe=False,
    #install_requires=['numpy, h5py']
)
