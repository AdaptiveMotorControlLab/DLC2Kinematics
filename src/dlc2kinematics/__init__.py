"""
dlc2kinematics
© M. Mathis Lab
https://github.com/AdaptiveMotorControlLab/dlc2kinematics/
"""

from dlc2kinematics.version import __version__, VERSION
from dlc2kinematics.preprocess import load_data, smooth_trajectory
from dlc2kinematics.mainfxns import (
    compute_velocity,
    compute_acceleration,
    compute_speed,
    extract_kinematic_synergies,
    compute_umap,
)
from dlc2kinematics.plotting import (
    plot_joint_angles,
    plot_velocity,
    pca_plot,
    plot_3d_pca_reconstruction,
    visualize_synergies,
    plot_umap,
)
from dlc2kinematics.utils import auxiliaryfunctions

from dlc2kinematics.joint_analysis import (
    load_joint_angles,
    compute_joint_angles,
    compute_joint_velocity,
    compute_joint_acceleration,
    compute_correlation,
    compute_pca,
)
from dlc2kinematics.quaternions import (
    compute_joint_quaternions,
    compute_joint_doubleangles,
    plot_joint_quaternions,
    compute_joint_quaternion_velocity,
    compute_joint_quaternion_acceleration,
    _load_quaternions,
)

try:
    from dlc2kinematics.plotting import (
        plot_joint_angles,
        plot_velocity,
        pca_plot,
        plot_3d_pca_reconstruction,
        visualize_synergies,
    )
    from dlc2kinematics.visualization import (
        Visualizer3D,
        MinimalVisualizer3D,
        MultiVisualizer,
        Visualizer2D,
    )
except:
    print("Could not import plotting and visualization functions.")
