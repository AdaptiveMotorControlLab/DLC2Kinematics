"""
dlc2kinematics
© M. Mathis Lab
https://github.com/AdaptiveMotorControlLab/dlc2kinematics/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from dlc2kinematics.utils import auxiliaryfunctions


def plot_velocity(df, df_velocity, start=None, end=None):
    """
    Plots the computed velocities with the original X, Y data

    Parameters
    ----------
    df: the .h5 file 2d or 3d from DeepLabCut that you loaded during dlc2kinematics.load_data

    df_velocity: Pandas dataframe of computed velocities. Computed with dlc2kinematics.compute_velocity

    start: int
        Optional. Integer specifying the start of frame index to select. Default is set to 0.

    end: int
        Optional. Integer specifying the end of frame index to select. Default is set to length of dataframe.

    Example
    -------
    >>> dlc2kinematics.plot_velocity(df, df_velocity, start=1,end=500)
    """

    try:
        df_velocity = pd.read_hdf(df_velocity, "df_with_missing")
    except:
        pass
    if start == None:
        start = 0
    if end == None:
        end = len(df_velocity)

    ax = df_velocity[start:end].plot(kind="line")
    plt.xlabel("Frame numbers")
    plt.ylabel("velocity (AU)")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.title("Computed Velocity", loc="left")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.show()

    ax1 = df[start:end].plot(kind="line")
    plt.xlabel("Frame numbers")
    plt.ylabel("position")
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    plt.title("Loaded Position Data", loc="left")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.show()


def plot_joint_angles(joint_angle, angles=[None], start=None, end=None):
    """
    Plots the joint angles

    Parameters
    ----------
    joint_angle: Pandas dataframe of joint angles.

    angles: list
        Optional. List of angles to plot

    start: int
        Optional. Integer specifying the start of frame index to select. Default is set to 0.

    end: int
        Optional. Integer specifying the end of frame index to select. Default is set to length of dataframe.


    Example
    -------
    >>> dlc2kinematics.plot_joint_angles(joint_angle)
    """
    try:
        joint_angle = pd.read_hdf(joint_angle, "df_with_missing")
    except:
        pass
    if start == None:
        start = 0
    if end == None:
        end = len(joint_angle)

    if angles[0] == None:
        angles = list(joint_angle.columns.get_level_values(0))

    ax = joint_angle[angles][start:end].plot(kind="line")
    #    plt.tight_layout()
    plt.ylim([0, 180])
    plt.xlabel("Frame numbers")
    plt.ylabel("Joint angle in degrees")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.title("Joint Angles", loc="left")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.show()


def visualize_synergies(data_reconstructed):
    ncols, _, nrows = data_reconstructed.shape
    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(10, 8))
    for row in range(nrows):
        for col in range(ncols):
            ax = axes[row, col]
            ax.plot(data_reconstructed[col, :, row])
            if row == 0:
                ax.set_title(f"Synergy {col + 1}")
            ax.tick_params(axis="both", which="both", bottom=False, left=False)


def pca_plot(
    num,
    v,
    fig,
    k,
    df_projected,
    variance_explained,
    n_comps,
    bodyparts2plot,
    bp_to_connect,
):
    """
    customized PCA plot.

    """
    gs = gridspec.GridSpec(1, num)
    cmap = "cividis_r"
    alphaValue = 0.7
    color = plt.cm.get_cmap(cmap, len(bodyparts2plot))
    scorer = df_projected.columns.get_level_values(0)[0]
    axes2 = fig.add_subplot(gs[0, v], projection="3d")  # row 0, col 1
    fig.tight_layout()
    plt.cla()
    axes2.cla()

    xdata_3d = []
    ydata_3d = []
    zdata_3d = []

    xdata_3d_pca = []
    ydata_3d_pca = []
    zdata_3d_pca = []

    xRight_3d = []
    yRight_3d = []
    zRight_3d = []

    xRight_3d_pca = []
    yRight_3d_pca = []
    zRight_3d_pca = []

    axes2.view_init(113, 270)
    axes2.set_xlim3d([4, 10])  # [4,9]
    axes2.set_ylim3d([1, -3])  # 1,-5
    axes2.set_zlim3d([14, 10])
    axes2.set_xticklabels([])
    axes2.set_yticklabels([])
    axes2.set_zticklabels([])
    axes2.xaxis.grid(False)

    # Need a loop for drawing skeleton
    for right_bp in bp_to_connect:
        xRight_3d_pca.append(df_projected.iloc[k][scorer][right_bp]["x"])
        yRight_3d_pca.append(df_projected.iloc[k][scorer][right_bp]["y"])
        zRight_3d_pca.append(df_projected.iloc[k][scorer][right_bp]["z"])

    for bpindex, bp in enumerate(bodyparts2plot):
        xdata_3d_pca.append(df_projected.iloc[k][scorer][bp]["x"])
        ydata_3d_pca.append(df_projected.iloc[k][scorer][bp]["y"])
        zdata_3d_pca.append(df_projected.iloc[k][scorer][bp]["z"])
        p2 = axes2.scatter(
            df_projected[scorer][bp]["x"][k],
            df_projected[scorer][bp]["y"][k],
            df_projected[scorer][bp]["z"][k],
            marker="o",
        )  # c=color(bodyparts2plot.index(bp))

    axes2.plot(
        xRight_3d_pca, yRight_3d_pca, zRight_3d_pca, color="black", alpha=alphaValue
    )
    axes2.set_title(
        "Reconstructed with %s PCs and %s%% EV"
        % (n_comps, round(variance_explained * 100, 2))
    )

    return plt


def plot_3d_pca_reconstruction(
    df, n_components, framenumber, bodyparts2plot, bp_to_connect
):
    """
    Computes and plots a 3D reconstruction of various pc's.

    Parameters
    ----------
    df: Pandas dataframe of either original bodypart 3d file, or outputs from compute_joint_velocity, or compute_joint_acceleration.

    n_components: int
        how many principal components to use

    framenumer: int
        which frame do you want to display?

    Example
    -------
    >>> dlc2kinematics.plot_3d_pca_reconstruction(joint_velocity, n_components=10, frame=1)
    """
    # pca = PCA(n_components=n_components, svd_solver="full").fit(df)
    comp = [n_components, 1]
    num = len(comp)
    fig = plt.figure(figsize=(9, 5))
    fig.tight_layout()
    scorer = df.columns.get_level_values(0)[0]

    k = framenumber  # a specific frame the user wants to show.
    for v in range(len(comp)):
        p = PCA(comp[v]).fit(df)
        n_comps = p.n_components_  # n_components
        components = p.transform(df)
        projected = p.inverse_transform(components)
        variance_explained = np.cumsum(p.explained_variance_ratio_)[-1]
        df_projected = auxiliaryfunctions.create_empty_df(df)
        bodyparts = list(df.columns.get_level_values(1))[0::3]
        projected_reshape = projected.reshape(len(projected), len(bodyparts), 3)
        for idx, bp in enumerate(bodyparts):
            df_projected.loc(axis=1)[scorer, bp] = projected_reshape[:, idx]
        pca_plot(
            num,
            v,
            fig,
            k,
            df_projected,
            variance_explained,
            n_comps,
            bodyparts2plot,
            bp_to_connect,
        )
    plt.show()


def plot_umap(Y, size=5, alpha=1, color="indigo", figsize=(10, 6)):
    """
    Computes and plots a 3D reconstruction of various pc's.

    Parameters
    ----------
    Y: fitted data from dlc2kinematics.compute_umap. n_fitted_samples x number_components

    size: int
        size of each individual datapoint

    alpha: int
        alpha blending value, transparent - opaque

    color:
        Depending on what you want to color, the color input can be a matrix of size n_fitted_samples x number_components

    Example
    -------
    >>> c = np.random.rand(Y.shape[0],3)
    >>> dlc2kinematics.plot_umap(Y, 5, 1, c)
    """
    
    if Y.shape[1] < 2:
        print("Please pick at least 2 components for plotting.")
    else:
        fig, ax = plt.subplots(figsize=figsize)

        if Y.shape[1] > 2:
            ax = Axes3D(fig)
            ax.view_init(0, 20)
            ax.dist = 8
            ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], alpha=alpha, s=size, c=color)
        else:
            ax.scatter(Y[:, 0], Y[:, 1], alpha=alpha, s=size, c=color)
    

        
