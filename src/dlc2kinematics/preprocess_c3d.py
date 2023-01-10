import c3d
import numpy as np
import pandas as pd
from dlc2kinematics.preprocess import smooth_trajectory


def load_c3d_data(filename, scorer="scorer", smooth=False, filter_window=3, order=1):
    """
    Reads the input datafile which is a c3d file and reshapes the data to be compatible with dlc2kinematics

    Parameters
    ----------
    filename: string
        Full path of the .c3d file as a string.

    scorer: string
        Optional. Name of the scorer.

    smooth: Bool
        Optional. Smooths coordinates of all bodyparts in the dataframe.

    filter_window: int
        Optional. Only used if the optional argument `smooth` is set to True. The length of filter window which needs to be a positive odd integer

    order: int
        Optional. Only used if the optional argument `smooth` is set to True. Order of the polynomial to fit the data. The order must be less than the filter_window

    Outputs
    -------
    df: dataframe; smoothed in case the optional argument `smooth` is set to True.
    bodyparts: List of unique bodyparts in the dataframe.
    scorer: Scorer name as string.
    first_frame: first frame of the exported region of interest from the video
    last_frame: last frame of the exported region of interest from the video
    sample_rate: acquisition rate of the kinematics data

    Example
    -------
    Linux/MacOS
    >>> df, bodyparts, scorer, first_frame, last_frame, sample_rate = preprocess_c3d.load_c3d_data('Users/user/Documents/c3d_dlc/runway03.c3d')
    Windows
    >>> df, bodyparts, scorer, first_frame, last_frame, sample_rate = dlc2kinematics.preprocess_c3d.load_c3d_data('C:\\yourusername\\runway03.c3d')


    """
    axis = ["x", "y", "z"]
    run, bodyparts, first_frame, last_frame, sample_rate = get_data_from_c3d_file(
        filename
    )
    df = create_empty_df(scorer, bodyparts, np.shape(run)[0])
    count = 0
    for bp in bodyparts:
        for a in axis:
            df[scorer, bp, a] = run[:, count]
            count += 1

    if smooth:
        df = smooth_trajectory(
            df,
            bodyparts,
            filter_window,
            order,
            deriv=0,
            save=False,
            output_filename=None,
            destfolder=None,
        )

    return df, bodyparts, scorer, first_frame, last_frame, sample_rate


def get_data_from_c3d_file(filename):
    """
    Loads the c3d file and returns the data and metadata in it

    Parameters
    ----------
    filename: string
        Full path to the .c3d file

    Outputs
    -------
    data: Array of the coordinates of each bodypart at each sample
    bodyparts: List of the bodyparts in the c3d file
    first_frame: int indicating the first frame of the region of interest of the recording
    last_frame: int indicating the last frame of the region of interest of the recording
    sample_rate: float sampling rate of the kinematics data
    """
    with open(filename, "rb") as f:
        c3d_reader = c3d.Reader(f)
        first_frame = c3d_reader._header.first_frame
        last_frame = c3d_reader._header.last_frame
        sample_rate = c3d_reader._header.frame_rate
        bodyparts = get_c3d_bodyparts(f)

        data = []
        for frame_no, points, analog in c3d_reader.read_frames(copy=False):
            fields = []
            for x, y, z, err, cam in points:
                fields.append(x)
                fields.append(y)
                fields.append(z)
            data.append(fields)
        data = np.asarray(data)

    return data, bodyparts, first_frame, last_frame, sample_rate


def create_empty_df(scorer, bodyparts, frames_no):
    """
    Creates empty dataframe to receive 3d data from c3d file
    Parameters
    ----------
    scorer: string
        mock data scorer
    bodyparts: list
        bodyparts that will be in the dataframe
    frames_no: int
        number of frames of the recording
    Outputs
    -------
    df: empty dataframe with shape compatible with dlc2kinematics
    """

    df = None
    a = np.full((frames_no, 3), np.nan)
    for bodypart in bodyparts:
        pdindex = pd.MultiIndex.from_product(
            [[scorer], [bodypart], ["x", "y", "z"]],
            names=["scorer", "bodyparts", "coords"],
        )

        frame = pd.DataFrame(a, columns=pdindex, index=range(0, frames_no))
        df = pd.concat([frame, df], axis=1)
    return df


def get_c3d_bodyparts(handle):
    """
    Reads in the labels from a .c3d handle
    Parameters
    ----------
    handle: BufferedReader of the .c3d file

    Outputs
    -------
    bodyparts_labels: list of the bodyparts labels
    """
    reader = c3d.Reader(handle)
    a = reader._groups["POINT"]._params["LABELS"]
    C, R = a.dimensions
    bodyparts_labels = [
        a.bytes[r * C : (r + 1) * C].strip().decode().lower() for r in range(R)
    ]
    return bodyparts_labels
