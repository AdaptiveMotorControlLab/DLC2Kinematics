"""
dlc2kinematics
© M. Mathis Lab
https://github.com/AdaptiveMotorControlLab/dlc2kinematics/

"""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dlc2kinematics",
    version="0.0.4",
    author="Mackenzie Mathis Lab Members",
    author_email="mackenzie@post.harvard.edu",
    description="Library for kinematic analysis of DeepLabCut outputs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AdaptiveMotorControlLab/dlc2kinematics/",
    install_requires=[
        "h5py",
        "intel-openmp",
        "ipython",
        "ipython-genutils",
        "matplotlib>=3.0.3",
        "numpy<=1.21",
        "pandas>=1.0",
        "python-dateutil",
        "pyyaml",
        "requests",
        "setuptools",
        "scikit-image",
        "scikit-learn",
        "scikit-kinematics",
        "scipy",
        "tables",
        "umap-learn",
        "tqdm",
        "ruamel.yaml>=0.15.0",
        "wheel",
    ],
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ),
    entry_points="""[console_scripts]
            kin=kin:main""",
)
