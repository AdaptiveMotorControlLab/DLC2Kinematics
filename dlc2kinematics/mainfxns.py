"""
dlc2kinematics
© M. Mathis Lab
https://github.com/AdaptiveMotorControlLab/dlc2kinematics/
"""

import pandas as pd
import numpy as np
from dlc2kinematics.preprocess import smooth_trajectory
from sklearn.decomposition import PCA
import umap


def compute_velocity(df, bodyparts, filter_window=3, order=1):
    """
    Computes the velocity of bodyparts in the input dataframe.

    Parameters
    ----------
    df: Pandas multiindex dataframe. Assumes the the dataframe is already smoothed. If not, adjust the filter_window and order to smooth the dataframe.

    bodyparts: List
        List of bodyparts to smooth. To smooth all the bodyparts use bodyparts=['all']

    filter_window: int
        Optional. The length of filter window which needs to be a positive odd integer

    order: int
        Optional. Order of the polynomial to fit the data. The order must be less than the filter_window

    Outputs
    -------
    vel: dataframe of velocity for the bodypart

    Example
    -------
    >>> df_smooth = dlc2kinematics.velocity(df,bodyparts=['nose','shoulder'])

    To smooth all the bodyparts in the dataframe, use
    >>> df_smooth = dlc2kinematics.velocity(df,bodyparts=['all'])

    """
    return smooth_trajectory(df, bodyparts, filter_window, order, deriv=1)


def compute_acceleration(df, bodyparts, filter_window=3, order=2):
    """
    Computes the acceleration of bodyparts in the input dataframe.

    Parameters
    ----------
    df: Pandas multiindex dataframe. Assumes the the dataframe is already smoothed. If not, adjust the filter_window and order to smooth the dataframe.

    bodyparts: List
        List of bodyparts to smooth. To smooth all the bodyparts use bodyparts=['all']

    filter_window: int
        Optional. The length of filter window which needs to be a positive odd integer

    order: int
        Optional. Order of the polynomial to fit the data. The order must be less than the filter_window

    Outputs
    -------
    vel: dataframe of velocity for the bodypart

    Example
    -------
    >>> df_smooth = dlc2kinematics.acceleration(df,bodyparts=['nose','shoulder'])

    To smooth all the bodyparts in the dataframe, use
    >>> df_smooth = dlc2kinematics.acceleration(df,bodyparts=['all'])

    """
    return smooth_trajectory(df, bodyparts, filter_window, order, deriv=2)


def compute_speed(df, bodyparts, filter_window=3, order=1):
    """
    Computes the speed of bodyparts in the input dataframe.

    Parameters
    ----------
    df: Pandas multiindex dataframe. Assumes the the dataframe is already smoothed. If not, adjust the filter_window and order to smooth the dataframe.

    bodyparts: List
        List of bodyparts to smooth. To smooth all the bodyparts use bodyparts=['all']

    filter_window: int
        Optional. The length of filter window which needs to be a positive odd integer

    order: int
        Optional. Order of the polynomial to fit the data. The order must be less than the filter_window

    Outputs
    -------
    speed: dataframe of speed for the bodyparts

    Example
    -------
    >>> speed = dlc2kinematics.compute_speed(df,bodyparts=['nose','shoulder'])

    To smooth all the bodyparts in the dataframe, use
    >>> speed = dlc2kinematics.compute_speed(df, bodyparts=['all'])
    """
    traj = smooth_trajectory(df, bodyparts, filter_window, order, deriv=1)
    coords = traj.columns.get_level_values("coords") != "likelihood"
    prob = traj.loc[:, ~coords]

    def _calc_norm(cols):
        return np.sqrt(np.sum(cols ** 2, axis=1))

    groups = (
        ["individuals", "bodyparts"]
        if "individuals" in df.columns.names
        else "bodyparts"
    )
    vel = traj.loc[:, coords].groupby(level=groups, axis=1).apply(_calc_norm)
    scorer = df.columns.get_level_values("scorer").unique().to_list()
    try:
        levels = vel.columns.levels
    except AttributeError:
        levels = [vel.columns.values]
    vel.columns = pd.MultiIndex.from_product(
        [scorer] + levels + [["speed"]],
        names=["scorer"] + vel.columns.names + ["coords"],
    )
    return vel.join(prob)


def extract_kinematic_synergies(
    data, tol=0.95, num_syn=None, standardize=False, ampl=1
):
    """
    Decompose kinematic data into synergies.

    Parameters
    ----------
    data: 2D Numpy array to be decomposed

    tol: 0 < float < 1, optional (default=0.95)
        Relative amount of variance to be explained.
        The number of components is automatically selected so that
        the explained variance is greater than the specified percentage.

    num_syn: list, optional (default=None)
        List of synergies to use for reconstruction.
        Note that Python indexing is zero based; synergy 1 must therefore be [0].
        By default, all synergies explaining up to *tol* % variance will be retained.

    standardize: bool, optional (default=False)
        If true, standardize data to unit variance before PCA.

    ampl: float > 1
        Amplification factor.
        Typically used for visualization to magnify the importance of synergies.

    Returns
    -------
    tuple(
        reconstructed data with shape (len(num_syn), data.shape[0], data.shape[1]),
        variance explained by each individual synergy,
        the actual synergies (or principal components)
    )

    Examples
    --------

    To extract all synergies explaining up to 80% variance:
    >>> dlc2kinematics.extract_kinematic_synergies(data, tol=0.8)

    To retain only synergy #2 and 4 and amplify their effect on reconstruction by a factor of 3:
    >>> dlc2kinematics.extract_kinematic_synergies(data, num_syn=[1, 3], ampl=3)
    """
    pca = PCA()
    if standardize:
        mean = data.mean(axis=0)
        sd = data.std(axis=0)
        data = (data - mean) / sd
    # Vector are projected onto the new PCA space
    scores = pca.fit_transform(data)
    vaf = np.cumsum(pca.explained_variance_ratio_)
    if not num_syn:
        max_syn = np.searchsorted(vaf, tol) + 1
        num_syn = np.arange(max_syn)
    # Projection back onto the original vector space for reconstruction
    data_recons = np.empty(shape=(len(num_syn), *data.shape))
    for n, syn in enumerate(num_syn):
        recons = scores[:, syn].reshape(-1, 1) @ pca.components_[syn].reshape(1, -1)
        data_recons[n] = pca.mean_ + ampl * recons
    return data_recons.squeeze(), vaf[num_syn], scores[:, num_syn]


def compute_umap(
    df,
    keypoints=None,
    pcutoff=0.6,
    chunk_length=30,
    fit_transform=True,
    n_neighbors=30,
    n_components=3,
    metric="euclidean",
    min_dist=0,
    metric_kwds=None,
    output_metric="euclidean",
    output_metric_kwds=None,
    n_epochs=None,
    learning_rate=1.0,
    init="spectral",
    spread=1.0,
    low_memory=False,
    set_op_mix_ratio=1.0,
    local_connectivity=1.0,
    repulsion_strength=1.0,
    negative_sample_rate=5,
    transform_queue_size=4.0,
    a=None,
    b=None,
    random_state=None,
    angular_rp_forest=False,
    target_n_neighbors=-1,
    target_metric="categorical",
    target_metric_kwds=None,
    target_weight=0.5,
    transform_seed=42,
    force_approximation_algorithm=False,
    verbose=False,
    unique=False,
):

    """
    Compute and fit UMAP embedding to data. Creates a matrix X used as input for UMAP. (n_samples x n_features)

    Main Parameters
    ----------
    df: original dataframe df, _, _ = dlc2kinematics.load_data(DLC_2D_file)

    keypoints: list of limbs of interests ex: key = ['LeftForelimb', 'RightForelimb'], if None, all bodyparts are taken into account

    pcutoff: likelihood at which the keypoints are kept for analysis, if under pcutoff, set coord to 0

    chunk_length: Number of frames per segment. #TODO: add the posibility of a sliding window?

    fit_transform: if False, returns the mapper, otherwise returns the transformed data.

    n_neighbors: float (optional, default 15)
        The size of local neighborhood (in terms of number of neighboring
        sample points) used for manifold approximation. Larger values
        result in more global views of the manifold, while smaller
        values result in more local data being preserved. In general
        values should be in the range 2 to 100.

    n_components: int (optional, default 2)
        The dimension of the space to embed into. This defaults to 2 to
        provide easy visualization, but can reasonably be set to any
        integer value in the range 2 to 100.

    min_dist: float (optional, default 0.1)
        The effective minimum distance between embedded points. Smaller values
        will result in a more clustered/clumped embedding where nearby points
        on the manifold are drawn closer together, while larger values will
        result on a more even dispersal of points. The value should be set
        relative to the ``spread`` value, which determines the scale at which
        embedded points will be spread out.

    metric: string or function (optional, default 'euclidean')
        The metric to use to compute distances in high dimensional space.
        If a string is passed it must match a valid predefined metric. If
        a general metric is required a function that takes two 1d arrays and
        returns a float can be provided. For performance purposes it is
        required that this be a numba jit'd function. Valid string metrics
        include:
            * euclidean
            * manhattan
            * chebyshev

        To have a full list of the available metrics, visit: https://umap-learn.readthedocs.io/en/latest/parameters.html
        Metrics that take arguments (such as minkowski, mahalanobis etc.)
        can have arguments passed via the metric_kwds dictionary. At this
        time care must be taken and dictionary elements must be ordered
        appropriately; this will hopefully be fixed in the future.

    other_param: Other parameters can be added as **kwargs
        For a complete list of possible umap paramters: https://umap-learn.readthedocs.io/en/latest/api.html#umap.umap_.UMAP
        See example.

    Returns
    -------

    Y: Fits the sample x feature matrix into an embedded space and returns that transformed output.

    Examples
    --------

    To compute and fit an embedding to your set of dataframe df, _, _ = dlc2kinematics.load_data(DLC_2D_file):
    >>> other_param = {"n_components": 3, "min_dist":0}
    >>> Y = dlc2kinematics.compute_umap(df, key, chunk_length = 30 ,n_neighbors=15, **other_param)

    """

    # TODO: In case user gives a numpy array or something else than the expected df pandas table.

    # pandas dataframe managing
    df_clean = df.copy()

    if keypoints is None: # If no keypoints specified, use all
        keypoints = df_clean.columns.get_level_values('bodyparts').unique().to_list()

    df_limbs = df_clean.loc[:, pd.IndexSlice[:, keypoints]]

    temp = df_limbs.stack(level=['scorer', 'bodyparts']) # Stack with likelihood, x, y
    temp.loc[temp['likelihood'] < pcutoff, ['x','y']] = 0.0 # Set values under pcutoff to 0.0 to exclude
    unstacked_temp = temp.unstack(level=['scorer', 'bodyparts']) # Unstack again
    unstacked_temp.reorder_levels(['scorer','bodyparts','coords'], axis=1).reindex_like(df_limbs) # Re-index like original df

    n_frames, n_bodyparts = df_limbs.shape
    n_chunks = n_frames // chunk_length

    # Reshape for UMAP use
    poses = df_limbs.values[: n_chunks * chunk_length].reshape(
        (n_chunks, chunk_length, df_limbs.shape[1])
    )
    X = poses.reshape((n_chunks, -1))

    # Create embedding
    embedding = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        metric_kwds=None,
        output_metric="euclidean",
        output_metric_kwds=output_metric_kwds,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        init=init,
        spread=spread,
        low_memory=low_memory,
        set_op_mix_ratio=set_op_mix_ratio,
        local_connectivity=local_connectivity,
        repulsion_strength=repulsion_strength,
        negative_sample_rate=negative_sample_rate,
        transform_queue_size=transform_queue_size,
        a=a,
        b=b,
        random_state=random_state,
        angular_rp_forest=angular_rp_forest,
        target_n_neighbors=target_n_neighbors,
        target_metric=target_metric,
        target_metric_kwds=target_metric_kwds,
        target_weight=target_weight,
        transform_seed=transform_seed,
        force_approximation_algorithm=force_approximation_algorithm,
        verbose=verbose,
        unique=unique,
    )
    embedding.fit(X)
    transformed_data = embedding.transform(X) if fit_transform else None

    print(
        'Fitted data can be used with dlc2kinematics.plot_umap or if fit_transform was set to "False" with UMAP integrated functions (umap.plot.points(mapper))'
    )
    print(
        "Note that umap.plot.points plotting is currently only implemented for 2D embeddings, i.e., n_components = 2"
    )

    return (embedding, transformed_data)
