"""
Kinematik Package
© A. & M. Mathis Labs
https://github.com/MMathisLab/Kinematik/

Please see AUTHORS for contributors.
https://github.com/MMathisLab/Kinematik/blob/master/AUTHORS
"""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="kinematik",
    version="0.0b",
    author="Tanmay Nath, Kai Sandbrink, Alexander Mathis, Mackenzie Mathis",
    author_email="mackenzie@post.harvard.edu",
    description="Library for kinematic analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MMathisLab/Kinematik/",
    install_requires=['h5py~=2.7','imageio==2.3.0','intel-openmp',
                      'ipython~=6.0.0','ipython-genutils==0.2.0',
                      'matplotlib==3.0.3','moviepy~=0.2.3.5','numpy==1.14.5','opencv-python~=3.4',
                      'pandas==0.21.0','patsy','python-dateutil==2.7.3','pyyaml>=5.1','requests',
                      'ruamel.yaml==0.15','setuptools','scikit-image~=0.14.0','scikit-learn~=0.19.2',
                      'scikit-kinematics','scipy~=1.1.0','six==1.11.0','statsmodels==0.9.0','tables',
                      'tqdm>4','wheel==0.31.1'],
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
))
