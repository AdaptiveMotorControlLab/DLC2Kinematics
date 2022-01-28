"""
dlc2kinematics
© M. Mathis Lab
https://github.com/AdaptiveMotorControlLab/dlc2kinematics/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from dlc2kinematics.utils import auxiliaryfunctions


def load_joint_angles(data):
    """
    Loads the joint angles which are computed by dlc2kinematics.compute_joint_angles() and stored as pandas dataframe

    Parameters
    ----------
    data: string
        Full path of the pandas array(.h5) file as a string.

    Outputs
    -------
    df: dataframe

    Example
    -------
    Linux/MacOs
    >>> joint_angle = dlc2kinematics.load_joint_angles('/home/data/joint_angle.h5')
    Windows
    >>> joint_angle = dlc2kinematics.load_joint_angles('C:\\yourusername\\rig-95\\joint_angle.h5')

    """
    joint_angle = pd.read_hdf(data)
    return joint_angle


def compute_joint_angles(
    df,
    joints_dict,
    save=True,
    destfolder=None,
    output_filename=None,
    dropnan=False,
    smooth=False,
    filter_window=3,
    order=1,
    pcutoff=0.4,
):
    """
    Computes the joint angles for the bodyparts.

    Parameters
    ----------
    df: Pandas multiindex dataframe which is the output of DeepLabCut. Assumes the the dataframe is already smoothed. If not, adjust the filter_window and order to smooth the dataframe.

    joints_dict: Dictionary
        Keys of the dictionary specifies the joint angle and the corresponding values specify the bodyparts. e.g.
        joint_dict = {'R-Elbow': ['R_shoulder', 'Right_elbow', 'Right_wrist']}

    save: boolean
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

    pcutoff: float
        Optional. Specifies the likelihood. All bodyparts with low `pcutoff` (i.e. < 0.4) are not used to compute the joint angles. It is only useful when computing joint angles from 2d data.

    Outputs
    -------
    joint_angles: dataframe of joint angles

    Example
    -------
    >>> joint_angles = dlc2kinematics.compute_joint_angles(df,joint_dict)

    """
    flag, _ = auxiliaryfunctions.check_2d_or_3d(df)
    is_multianimal = "individuals" in df.columns.names
    if flag == "2d" and pcutoff:

        def filter_low_prob(cols, prob):
            mask = cols.iloc[:, 2] < prob
            cols.loc[mask, :2] = np.nan
            return cols

        df = df.groupby("bodyparts", axis=1).apply(filter_low_prob, prob=pcutoff)

    angle_names = list(joints_dict)
    if not destfolder:
        destfolder = os.getcwd()
    if not output_filename:
        output_filename = (
            "joint_angles_" + df.columns.get_level_values("scorer").unique()[0]
        )
    filepath = os.path.join(destfolder, output_filename + ".h5")
    if os.path.isfile(filepath):
        print("File already present. Reading %s" % output_filename)
        angles = pd.read_hdf(filepath)
        if not all(angles.columns.isin(angle_names)):
            raise IOError(
                "The existing file has different joint angles than specified "
                "in the dictionary joints_dict. "
                "Please delete the existing file and try again!"
            )
        else:
            angles = angles.loc[:, angle_names]
    else:
        angles = dict()
        for joint, bpts in joints_dict.items():
            print(f"Computing joint angles for {joint}")
            mask = df.columns.get_level_values("bodyparts").isin(bpts)
            temp = df.loc[:, mask]
            if is_multianimal:
                for animal, frame in temp.groupby(level="individuals", axis=1):
                    angles[f"{joint}_{animal}"] = frame.apply(
                        auxiliaryfunctions.jointangle_calc, axis=1
                    ).values
            else:
                angles[joint] = temp.apply(
                    auxiliaryfunctions.jointangle_calc, axis=1
                ).values
        angles = pd.DataFrame.from_dict(angles)

    if dropnan:
        angles.dropna(inplace=True)

    if smooth:
        angles[:] = savgol_filter(angles, filter_window, order, deriv=0, axis=0)

    if save:
        print(f"Saving the joint angles as a pandas array in {destfolder}")
        angles.to_hdf(
            filepath,
            "df_with_missing",
            format="table",
            mode="w",
        )

    return angles


def compute_joint_velocity(
    joint_angle,
    filter_window=3,
    order=1,
    save=True,
    destfolder=None,
    output_filename=None,
    dropnan=False,
):
    """
    Computes the joint angular velocities.

    Parameters
    ----------
    joint_angle: Pandas dataframe of joint angles. You can also pass the full path of joint angle filename as a string.

    filter_window: int
        Optional. The length of filter window which needs to be a positive odd integer

    order: int
        Optional. Order of the polynomial to fit the data. The order must be less than the filter_window

    save: boolean
        Optional. Saves the joint velocity as a pandas dataframe if set to True.

    destfolder: string
        Optional. Saves the joint velocity in the specfied destination folder. If it is set to None, the joint velocity are saved in the current working directory.

    output_filename: string
        Optional. Name of the output file. If it is set to None, the file is saved as joint_angular_velocity.h5.

    dropnan: boolean
        Optional. If you want to drop any NaN values, this is useful for some downstream analysis (like PCA).

    Outputs
    -------
    joint_vel: dataframe of joint angular velocity

    Example
    -------
    >>> joint_vel = dlc2kinematics.compute_joint_velocity(joint_angle)
    """
    try:
        joint_angle = pd.read_hdf(joint_angle, "df_with_missing")
    except:
        pass
    temp = savgol_filter(joint_angle, filter_window, order, axis=0, deriv=1)
    angular_vel = pd.DataFrame(
        temp, columns=joint_angle.columns, index=joint_angle.index
    )

    if not destfolder:
        destfolder = os.getcwd()

    if not output_filename:
        output_filename = "joint_angular_velocity"

    if dropnan:
        print("Dropping the indices where joint angular velocity is nan")
        angular_vel.dropna(inplace=True)

    if save:
        print("Saving the joint angular velocity as a pandas array in %s " % destfolder)
        angular_vel.to_hdf(
            os.path.join(destfolder, output_filename + ".h5"),
            "df_with_missing",
            format="table",
            mode="w",
        )

    return angular_vel


def compute_joint_acceleration(
    joint_angle,
    filter_window=3,
    order=2,
    save=True,
    destfolder=None,
    output_filename=None,
    dropnan=False,
):
    """
    Computes the joint angular acceleration.

    Parameters
    ----------
    joint_angle: Pandas dataframe of joint angles. You can also pass the full path of joint angle filename as a string.

    filter_window: int
        Optional. The length of filter window which needs to be a positive odd integer.

    order: int
        Optional. Order of the polynomial to fit the data. The order must be less than the filter_window.

    save: boolean
        Optional. Saves the joint acceleration as a pandas dataframe if set to True.

    destfolder: string
        Optional. Saves the joint acceleration in the specfied destination folder. If it is set to None, the joint acceleration are saved in the current working directory.

    output_filename: string
        Optional. Name of the output file. If it is set to None, the file is saved as joint_angular_acceleration.h5

    dropnan: boolean
        Optional. If you want to drop any NaN values, this is useful for some downstream analysis (like PCA).

    Outputs
    -------
    joint_acc: dataframe of joint angular acceleration.


    Example
    -------
    >>> joint_acc = dlc2kinematics.compute_joint_acceleration(joint_angle)
    """
    try:
        joint_angle = pd.read_hdf(joint_angle, "df_with_missing")
    except:
        pass

    temp = savgol_filter(joint_angle, filter_window, order, axis=0, deriv=2)
    angular_acc = pd.DataFrame(
        temp, columns=joint_angle.columns, index=joint_angle.index
    )
    if not destfolder:
        destfolder = os.getcwd()

    if not output_filename:
        output_filename = "joint_angular_acceleration"
    if dropnan:
        print("Dropping the indices where joint angular acceleration is nan")
        angular_acc.dropna(inplace=True)

    if save:
        print(
            "Saving the joint angular acceleration as a pandas array in %s "
            % destfolder
        )
        angular_acc.to_hdf(
            os.path.join(destfolder, output_filename + ".h5"),
            "df_with_missing",
            format="table",
            mode="w",
        )

    return angular_acc


def compute_correlation(feature, plot=False, colormap="viridis"):
    """
    Computes the correlation between the joint angles.

    Parameters
    ----------
    feature: Pandas dataframe of joint anglular feature e.g. angular velocity. You can also pass the full path of joint anglular feature filename as a string.

    plot: Bool
        Optional. Plots the correlation.

    colormap: string
        Optional. The colormap associated with the matplotlib. Check here for range of colormap options https://matplotlib.org/3.1.1/tutorials/colors/colormaps.html

    Outputs
    -------
    corr: dataframe of correlation.


    Example
    -------
    >>> corr = dlc2kinematics.compute_correlation(joint_vel, plot=True)
    """

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    try:
        feature = pd.read_hdf(feature, "df_with_missing")
    except:
        pass
    keys = list(feature.columns.values)

    correlation = feature.corr()
    if plot:
        im = plt.matshow(correlation, cmap=colormap)
        ax = plt.gca()
        plt.title("Correlation")
        plt.xticks(np.arange(0, len(keys), 1.0))
        plt.yticks(np.arange(0, len(keys), 1.0))
        ticks_labels = keys

        ax.set_xticklabels(ticks_labels, rotation=90)
        ax.set_yticklabels(ticks_labels)
        plt.gca().xaxis.tick_bottom()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.10)

        plt.colorbar(im, cax=cax)
        plt.clim(0, 1)
        plt.show()

    return correlation


def compute_pca(feature, n_components=None, plot=True, alphaValue=0.7):
    """
    Computes dimentionality reduction using Principal Component Analysis.

    Parameters
    ----------
    feature: Pandas dataframe of joint anglular feature e.g. angular velocity. You can also pass the full path of joint anglular feature filename as a string.

    n_components: int
        Number of components to keep.

    plot: Bool
        Optional. Plots the correlation.

    alphaValue: float
        Optional. The colormap associated with the matplotlib. Check here for range of colormap options https://matplotlib.org/3.1.1/tutorials/colors/colormaps.html

    Outputs
    -------
    corr: dataframe of correlation.


    Example
    -------
    >>> corr = dlc2kinematics.compute_correlation(joint_vel)
    """
    try:
        feature = pd.read_hdf(feature, "df_with_missing")
    except:
        pass
    # Remove the nans
    feature = feature.dropna()
    pca = PCA(n_components)
    pca.fit(feature)
    if plot:
        fig, ax = plt.subplots(figsize=(8, 5))
        plt.xlabel("Principal components")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.ylabel("Cumulative Variance Explained")
        plt.plot(np.cumsum(pca.explained_variance_ratio_), alpha=alphaValue)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        plt.title("Dimensionality Reduction", loc="left")
        plt.show()
    return pca
