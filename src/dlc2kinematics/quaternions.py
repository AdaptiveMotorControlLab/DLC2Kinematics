"""
dlc2kinematics
© M. Mathis Lab
https://github.com/AdaptiveMotorControlLab/dlc2kinematics/
"""

import pandas as pd
import numpy as np
from skinematics import quat, vector
import matplotlib.pyplot as plt
import os
import scipy as sc
from pathlib import Path
from sklearn.decomposition import PCA
from dlc2kinematics.utils import auxiliaryfunctions


def compute_joint_quaternions(
    df,
    joints_dict,
    save=True,
    destfolder=None,
    output_filename=None,
    dropnan=False,
    smooth=False,
    filter_window=3,
    order=1,
    use4d=True,
):
    """
    Computes the joint quaternions for the bodyparts.

    Parameters
    ----------
    df: Pandas multiindex dataframe. Assumes the the dataframe is already smoothed. If not, adjust the filter_window and order to smooth the dataframe.

    joints_dict: Dictionary
        Keys of the dictionary specifies the joint angle and the corresponding values specify the bodyparts. e.g.
        joint_dict = {'R-Elbow': ['R_shoulder', 'Right_elbow', 'Right_wrist']

    save: Bool
        Optional. Saves the joint angles as a pandas dataframe if set to True.

    destfolder: string
        Optional. Saves the joint angles in the specfied destination folder. If it is set to None, the joint angles are saved in the current working directory.

    output_filename: string
        Optional. Name of the output file. If it is set to None, the file is saved as joint_angles_<scorer_name>.h5, <scorer_name> is the name of the scorer in the input df.

    dropnan: boolean
        Optional. If you want to drop any NaN values, this is useful for some downstream analysis (like PCA).

    smooth: boolean
        Optional. If you want to smooth the data with a svagol filter, you can set this to true, and then also add filter_window and order.

    filter_window: int
        Optional. If smooth=True,  window is set here, which needs to be a positive odd integer.

    order: int
        Optional. Only used if the optional argument `smooth` is set to True. Order of the polynomial to fit the data. The order must be less than the filter_window

    use4d: boolean
        Optional. Determines whether all 4 components of the quaternion are returned or just the quaternion vector (which uniquely determines quaternion of rotation due to the constraing mag = 1)

    Outputs
    -------
    joint_quaternions: dataframe of joint angles.
        Rows are time points, columns are multiindex with joint names ['R-Elbow', ...] and quaternion components ['a', 'b', 'c', 'd']

    Example
    -------
    >>> joint_quaternions = dlc2kinematics.compute_joint_quaternions(df,joint_dict)

    """

    scorer = df.columns.get_level_values(0)[0]

    if use4d:
        comps = ["a", "b", "c", "d"]
    else:
        comps = ["b", "c", "d"]

    joints = list(joints_dict.keys())

    quaternion_columns = pd.MultiIndex.from_product(
        (joints, comps), names=["joint name", "comp"]
    )

    quaternions = pd.DataFrame(index=df.index, columns=quaternion_columns)

    destfolder, output_filename = _get_filenames(destfolder, output_filename, scorer)

    if os.path.isfile(os.path.join(destfolder, output_filename + ".h5")):
        return _load_quaternions(destfolder, output_filename)
    else:
        for keys, vals in joints_dict.items():
            a, b, c = vals[0], vals[1], vals[2]
            jointname = keys
            """
            if not use4d:
                quatcompnames = [jointname + '-quat' + compname for compname in ['b', 'c', 'd']]
            else:
                quatcompnames = [jointname + '-quat' + compname for compname in ['a', 'b', 'c', 'd']]
            """
            print("Computing joint quaternions for %s" % jointname)
            # print(scorer, a, b, c)
            # print(df[scorer].shape)
            # print(df[scorer][[a,b,c]].shape)
            tmpquat = np.squeeze(
                np.stack(
                    df[scorer][[a, b, c]]
                    .apply(
                        auxiliaryfunctions.jointquat_calc,
                        axis=1,
                        result_type="reduce",
                        args=tuple([use4d]),
                    )
                    .values
                )
            )

            quaternions[jointname] = tmpquat

            """
            for compname, comp in zip(quatcompnames, tmpquat.transpose()):
                print(comp)
                quaternions[compname] = comp
            """
            if smooth:
                for col in list(quaternions.columns):
                    quaternions[col] = auxiliaryfunctions.smoothen_angles(
                        quaternions, col, filter_window, order
                    )

    if dropnan:
        quaternions = quaternions.dropna()

    if save:
        print(
            "Saving the joint quaternions as a pandas array in %s "
            % os.path.join(destfolder, output_filename + ".h5")
        )
        quaternions.to_hdf(
            os.path.join(destfolder, output_filename + ".h5"),
            "df_with_missing",
            format="table",
            mode="w",
        )
        print("saved")
    return quaternions


def compute_joint_doubleangles(
    df,
    joints_dict,
    save=True,
    destfolder=None,
    output_filename=None,
    dropnan=False,
    smooth=False,
    filter_window=3,
    order=1,
    use4d=True,
):
    """
    Computes the joint double angles for the bodyparts.
    https://stackoverflow.com/questions/15101103/euler-angles-between-two-3d-vectors

    Parameters
    ----------
    df: Pandas multiindex dataframe. Assumes the the dataframe is already smoothed. If not, adjust the filter_window and order to smooth the dataframe.

    joints_dict: Dictionary
        Keys of the dictionary specifies the joint angle and the corresponding values specify the bodyparts. e.g.
        joint_dict = {'R-Elbow': ['R_shoulder', 'Right_elbow', 'Right_wrist']

    save: Bool
        Optional. Saves the joint angles as a pandas dataframe if set to True.

    destfolder: string
        Optional. Saves the joint angles in the specfied destination folder. If it is set to None, the joint angles are saved in the current working directory.

    output_filename: string
        Optional. Name of the output file. If it is set to None, the file is saved as joint_angles_<scorer_name>.h5, <scorer_name> is the name of the scorer in the input df.

    dropnan: boolean
        Optional. If you want to drop any NaN values, this is useful for some downstream analysis (like PCA).

    smooth: boolean
        Optional. If you want to smooth the data with a svagol filter, you can set this to true, and then also add filter_window and order.

    filter_window: int
        Optional. If smooth=True,  window is set here, which needs to be a positive odd integer.

    order: int
        Optional. Only used if the optional argument `smooth` is set to True. Order of the polynomial to fit the data. The order must be less than the filter_window

    use4d: boolean
        Optional. Determines whether all 4 components of the quaternion are returned or just the quaternion vector (which uniquely determines quaternion of rotation due to the constraing mag = 1)


    Outputs
    -------
    doubleangles: dataframe of joint angles
        Rows are time points, columns are multiindex with joint names ['R-Elbow', ...] and double angle components ['pitch', 'yaw']


    Example
    -------
    >>> doubleangles = dlc2kinematics.compute_joint_angles(df,joint_dict)

    """
    scorer = df.columns.get_level_values(0)[0]

    comps = ["pitch", "yaw"]

    joints = list(joints_dict.keys())

    doubleangle_columns = pd.MultiIndex.from_product((joints, comps))

    doubleangles = pd.DataFrame(index=df.index, columns=doubleangle_columns)

    destfolder, output_filename = _get_filenames(
        destfolder, output_filename, scorer, datatype="doubleangles"
    )

    if os.path.isfile(os.path.join(destfolder, output_filename + ".h5")):
        return _load_quaternions(destfolder, output_filename)
    else:
        for keys, vals in joints_dict.items():
            a, b, c = vals[0], vals[1], vals[2]
            jointname = keys
            """
            if not use4d:
                quatcompnames = [jointname + '-quat' + compname for compname in ['b', 'c', 'd']]
            else:
                quatcompnames = [jointname + '-quat' + compname for compname in ['a', 'b', 'c', 'd']]
            """
            print("Computing joint doubleangles for %s" % jointname)
            # print(scorer, a, b, c)
            # print(df[scorer].shape)
            # print(df[scorer][[a,b,c]].shape)
            tmpda = np.squeeze(
                np.stack(
                    df[scorer][[a, b, c]]
                    .apply(
                        auxiliaryfunctions.doubleangle_calc,
                        axis=1,
                        result_type="reduce",
                    )
                    .values
                )
            )

            doubleangles[jointname] = tmpda

            """
            for compname, comp in zip(quatcompnames, tmpquat.transpose()):
                print(comp)
                quaternions[compname] = comp
            """
            if smooth:
                for col in list(doubleangles.columns):
                    doubleangles[col] = auxiliaryfunctions.smoothen_angles(
                        doubleangles, col, filter_window, order
                    )

    if dropnan:
        doubleangles = doubleangles.dropna()

    if save:
        print(
            "Saving the joint quaternions as a pandas array in %s "
            % os.path.join(destfolder, output_filename + ".h5")
        )
        doubleangles.to_hdf(
            os.path.join(destfolder, output_filename + ".h5"),
            "df_with_missing",
            format="table",
            mode="w",
        )
        print("saved")
    return doubleangles


def plot_joint_quaternions(joint_quaternion, quats=[None], start=None, end=None):
    """
    Plots the joint quaternions (or velocity, or acceleration)

    Parameters
    ----------
    joint_quaternion: Pandas dataframe of joint quaternions, matching output of compute_joint_quaternions ()
        Rows are time points, columns are multiindex with joint names ['R-Elbow', ...] and quaternion components ['a', 'b', 'c', 'd']

    quats: list
        Optional. List of quats to plot, e.g. ['R-Elbow a', 'R-Elbow b', ... ] containing both the name of the joint and the component

    start: int
        Optional. Integer specifying the start of frame index to select. Default is set to 0.

    end: int
        Optional. Integer specifying the end of frame index to select. Default is set to length of dataframe.


    Example
    -------
    >>> dlc2kinematics.plot_joint_quaternions(joint_quaternion)
    """

    """
    try:
        joint_quaternion = pd.read_hdf(joint_quaternion, "df_with_missing")
    except:
        pass
    if start == None:
        start = 0
    if end == None:
        end = len(joint_quaternion)

    if quats[0] == None:
        quats = list(joint_quaternion.columns.get_level_values(0))

    ax = joint_quaternion[quats][start:end].plot(kind="line")
    #    plt.tight_layout()
    plt.ylim([0, 180])
    plt.xlabel("Frame numbers")
    plt.ylabel("joint quaternions")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.title("Joint Quaternion", loc="left")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.show()
    """

    try:
        joint_quaternion = pd.read_hdf(joint_quaternion, "df_with_missing")
    except:
        pass

    joint_quaternion = joint_quaternion.copy()

    joint_quaternion.columns = [
        " ".join(col).strip() for col in joint_quaternion.columns.values
    ]

    if start == None:
        start = 0
    if end == None:
        end = len(joint_quaternion)

    if quats[0] == None:
        angles = list(joint_quaternion.columns.get_level_values(0))

    ax = joint_quaternion[angles][start:end].plot(kind="line")
    #    plt.tight_layout()
    plt.xlabel("Frame numbers")
    plt.ylabel("Quaternion Component Magnitude")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.title("Joint Quaternions", loc="left")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.show()


def compute_joint_quaternion_velocity(
    joint_quaternion,
    filter_window=3,
    order=1,
):
    """
    Computes the first derivative of the joint quaternions in each component

    Parameters
    ----------
    joint_quaternion: Pandas dataframe of joint quaternions, matching output of compute_joint_quaternions
        Rows are time points, columns are multiindex with joint names ['R-Elbow', ...] and quaternion components ['a', 'b', 'c', 'd']


    filter_window: int
        Optional. The length of filter window which needs to be a positive odd integer

    order: int
        Optional. Order of the polynomial to fit the data. The order must be less than the filter_window

    Outputs
    -------
    quaternion_vel: dataframe of joint angular velocity
        Rows are time points, columns are multiindex with joint names ['R-Elbow', ...] and quaternion components ['a', 'b', 'c', 'd']


    Example
    -------
    >>> joint_quaternion_vel = dlc2kinematics.compute_joint_quaternion_velocity(joint_quaternion)
    """
    try:
        joint_quaternion = pd.read_hdf(joint_quaternion, "df_with_missing")
    except:
        pass

    """
    numCols = joint_quaternion.shape[2]

    if numCols == 3 and use4d:
        joint_quaternion = quat.unit_q(joint_quaternion.to_array())
    else:
        assert use4d, "cannot convert to 3d (either change input manually or set use4d==True)"
    """

    quaternion_vel = pd.DataFrame(
        columns=joint_quaternion.columns, index=joint_quaternion.index
    )
    for i in list(joint_quaternion.columns.values):
        quaternion_vel[i] = sc.signal.savgol_filter(
            joint_quaternion[i],
            window_length=filter_window,
            polyorder=order,
            axis=0,
            deriv=1,
        )

    return quaternion_vel


def compute_joint_quaternion_acceleration(joint_quaternion, filter_window=3, order=2):
    """
    Computes the joint angular acceleration.

    Parameters
    ----------
    joint_quaternion: Pandas dataframe of joint quaternions, matching output of compute_joint_quaternions

    filter_window: int
        Optional. The length of filter window which needs to be a positive odd integer.

    order: int
        Optional. Order of the polynomial to fit the data. The order must be less than the filter_window.

    Outputs
    -------
    joint_acc: dataframe of joint angular acceleration.
        Rows are time points, columns are multiindex with joint names ['R-Elbow', ...] and quaternion components ['a', 'b', 'c', 'd']

    Example
    -------
    >>> joint_acc = dlc2kinematics.compute_joint_acceleration(joint_angle)
    """
    try:
        joint_quaternion = pd.read_hdf(joint_quaternion, "df_with_missing")
    except:
        pass

    """
    numCols = joint_quaternion.shape[2]

    if numCols == 3:
        joint_quaternion = quat.unit_q(joint_quaternion.to_array())
    """

    quaternion_acc = pd.DataFrame(
        columns=joint_quaternion.columns, index=joint_quaternion.index
    )
    for i in list(joint_quaternion.columns.values):
        quaternion_acc[i] = sc.signal.savgol_filter(
            joint_quaternion[i],
            window_length=filter_window,
            polyorder=order,
            axis=0,
            deriv=2,
        )

    return quaternion_acc


def _get_filenames(destfolder, output_filename, scorer, datatype="joint_quaternions"):
    """Get the (formatted and completed) destination folder and output filename,
        helper function for various computes

    Parameters
    ----------

    destfolder : str, either None or destination folder, passed into compute function

    output_filename : str, either None or output filename, passed into compute function

    scorer: str

    Outputs
    -------

    destfolder : str

    output_filename : str
    """

    if destfolder == None:
        destfolder = os.getcwd()

    if output_filename == None:
        output_filename = str(datatype + "_" + scorer)

    return destfolder, output_filename


def _load_quaternions(destfolder, output_filename):
    """Load the quaternions from previously saved Pandas h5 file

    Parameters
    ----------
    destfolder: str
    output_filename: str

    Outputs
    -------
    quaternions : pandas dataframe
    """

    print("File already present. Reading %s" % output_filename)
    quaternions = pd.read_hdf(os.path.join(destfolder, output_filename + ".h5"))
    return quaternions
