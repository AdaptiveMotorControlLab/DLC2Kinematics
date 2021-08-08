"""
DLC2Kinematics Package
© M. Mathis Labs
https://github.com/AdaptiveMotorControlLab/DLC2Kinematics/

Please see AUTHORS for contributors.
https://github.com/AdaptiveMotorControlLab/DLC2Kinematics/blob/master/AUTHORS
"""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dlc2kinematics",
    version="0.0b1",
    author="Mackenzie Mathis & Lab",
    author_email="mackenzie@post.harvard.edu",
    description="a post-DeepLabCut module for kinematic analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AdaptiveMotorControlLab/DLC2Kinematics/",
    install_requires=['h5py>=2.7','imageio>=2.3.0','intel-openmp',
                      'ipython~=6.0.0','ipython-genutils==0.2.0',
                      'matplotlib','moviepy','numpy','opencv-python',
                      'pandas','patsy','python-dateutil','pyyaml>=5.1','requests',
                      'ruamel.yaml>=0.15','setuptools','scikit-image','scikit-learn',
                      'scikit-kinematics','scipy','six','statsmodels','tables',
                      'tqdm>4','wheel'],
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
))
