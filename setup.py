"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages, Extension
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
  name='datamodels',

  # Versions should comply with PEP440.  For a discussion on single-sourcing
  # the version across setup.py and the project code, see
  # https://packaging.python.org/en/latest/single_source_version.html
  version='0.0.1',

  description='Tools for Datamodel Estimation',
  long_description=long_description,
  long_description_content_type="text/markdown",

  # The project's main homepage.
  url='https://github.com/MadryLab/datamodeler',

  # Author details
  author='MadryLab',
  author_email='datamodels@mit.edu',

  # Choose your license
  license_files = ('LICENSE.txt',),


  # See https://pypi.python.org/pypi?%4Aaction=list_classifiers
  classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.8',
  ],

  # What does your project relate to?
  keywords='machine learning datamodels',

  # You can just specify the packages manually here if your project is
  # simple. Or you can use find_packages().
  packages=find_packages(),

  include_package_data=True,
  package_data={
            'certificate': ['client/server.crt']
  },


  #  Alternatively, if you want to distribute just a my_module.py, uncomment
  #  this:
  #    py_modules=["my_module"],
  # 
  #  List run-time dependencies here.  These will be installed by pip when
  #  your project is installed. For an analysis of "install_requires" vs pip's
  #  requirements files see:
  #  https://packaging.python.org/en/latest/requirements.html
  install_requires=['fastargs'],
  entry_points={}
)