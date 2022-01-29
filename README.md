[![PyPI version](https://badge.fury.io/py/kinematik.svg)](https://badge.fury.io/py/DLC2Kinematics)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/kinematik.svg?color=purple&label=PyPi)](https://pypistats.org/packages/DLC2Kinematics)

<img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1628452157953-RBVUGI7M3ABF9AOSUMMS/DLC2k.jpg?format=2500w" width="750" title="camera control" alt="cam cntrl" align="center" vspace = "80">


#### A post-deeplabcut module for kinematic analysis

This repo will continue to grow, but here are some helper functions to get you started. Note, the API is subject to change. You can run the functions on data files obtained from running inference with [DeepLabCut](http://deeplabcut.org/).


## Quick start

``` bash
pip install dlc2kinematics
```
## Useage

``` python
import dlc2kinematics
```

Load data:

``` python
df, bodyparts, scorer = dlc2kinematics.load_data(<path of the h5 file>)
```

### Basic Operations

Compute velocity:

  - For all bodyparts:
    ``` python
    df_vel = dlc2kinematics.compute_velocity(df,bodyparts=['all'])
    ```
  - For only few bodyparts:
    ``` python
    df_vel = dlc2kinematics.compute_velocity(df,bodyparts=['nose','joystick'])
    ```

Compute acceleration:

- For all bodyparts:
  ``` python
  df_acc = dlc2kinematics.compute_acceleration(df,bodyparts=['all'])
  ```
- For only few bodyparts:
  ``` python
  df_vel = dlc2kinematics.compute_acceleration(df,bodyparts=['nose','joystick'])
  ```

Compute speed:

``` python
df_speed = dlc2kinematics.compute_speed(df,bodyparts=['nose','joystick'])
```

### Computations in joint coordinates

To compute joint angles, we first create a dictionary where keys are the joint angles and the corresponding values are the set of bodyparts:

``` python
joint_dict= {}
joints_dict['R-Elbow']  = ['R_shoulder', 'Right_elbow', 'Right_wrist']
```

and compute the joint angles with

``` python
joint_angles = dlc2kinematics.compute_joint_angles(df,joints_dict)
```

Compute joint angular velocity with

``` python
joint_vel = dlc2kinematics.compute_joint_velocity(joint_angles)
```

Compute joint angular acceleration with

``` python
joint_acc = dlc2kinematics.compute_joint_acceleration(joint_angles)
```

Compute correlation of angular velocity

``` python
corr = dlc2kinematics.compute_correlation(joint_vel, plot=True)
```

Compute PCA of angular velocity with

``` python
pca = dlc2kinematics.compute_pca(joint_vel, plot=True)
```
### PCA-based reconstruction of postures

Compute and plot PCA based on posture reconstruction with: 

``` python
dlc2kinematics.plot_3d_pca_reconstruction(df_vel, n_components=10, framenumber=500,
                                     bodyparts2plot=bodyparts2plot, bp_to_connect=bp_to_connect)
```

### UMAP Embeddings
``` python
dlc2kinematics.compute_umap(df, key=['LeftForelimb', 'RightForelimb'], chunk_length=30, fit_transform=True, n_neighbors=30, n_components=3,metric="euclidean")
```

## Contributing

- If you spot an issue or have a question, please open an [issue](https://github.com/AdaptiveMotorControlLab/dlc2kinematics/issues) with a suitable tag.
- For [code contributions](https://github.com/AdaptiveMotorControlLab/dlc2kinematics/pulls):
  - please see the [contributing guide](docs/CONTRIBUTING.md).
  - Please reference all issues this PR addresses in the description text.
  - Before submitting your PR, ensure all code is formatted properly by running
    ``` bash
    black .
    ```
    in the root directory.
  - Assign a reviewer, typically [MMathisLab](https://github.com/MMathisLab).
  - sign CLA.

## Acknowledgements

This code is a collect of contributions from members of the Mathis Laboratory over the years. In particular: Tanmay Nath, Michael Beauzile, Sebastien Hausmann, Jessy Lauer, Steffen Schneider, Mackenzie Mathis
