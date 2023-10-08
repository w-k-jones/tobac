"""Provide tracking methods.

The individual features and associated area/volumes identified in
each timestep have to be linked into trajectories to analyse
the time evolution of their properties for a better understanding of
the underlying physical processes.
The implementations are structured in a way that allows for the future
addition of more complex tracking methods recording a more complex
network of relationships between features at different points in
time.

References
----------
.. Heikenfeld, M., Marinescu, P. J., Christensen, M.,
   Watson-Parris, D., Senf, F., van den Heever, S. C.
   & Stier, P. (2019). tobac 1.2: towards a flexible
   framework for tracking and analysis of clouds in
   diverse datasets. Geoscientific Model Development,
   12(11), 4551-4570.
"""

import logging
from operator import is_
import numpy as np
import pandas as pd
import xarray as xr
import warnings
import math
from . import utils as tb_utils
from .utils import periodic_boundaries as pbc_utils
from .utils import internal as internal_utils

from packaging import version as pkgvsn
import trackpy as tp
from copy import deepcopy


def linking_trackpy(
    features,
    field_in,
    dt,
    dxy,
    dz=None,
    v_max=None,
    d_max=None,
    d_min=None,
    subnetwork_size=None,
    memory=0,
    stubs=1,
    time_cell_min=None,
    order=1,
    extrapolate=0,
    method_linking="random",
    adaptive_step=None,
    adaptive_stop=None,
    cell_number_start=1,
    cell_number_unassigned=-1,
    vertical_coord="auto",
    min_h1=None,
    max_h1=None,
    min_h2=None,
    max_h2=None,
    PBC_flag="none",
):
    """Perform Linking of features in trajectories.

    The linking determines which of the features detected in a specific
    timestep is most likely identical to an existing feature in the
    previous timestep. For each existing feature, the movement within
    a time step is extrapolated based on the velocities in a number
    previous time steps. The algorithm then breaks the search process
    down to a few candidate features by restricting the search to a
    circular search region centered around the predicted position of
    the feature in the next time step. For newly initialized trajectories,
    where no velocity from previous time steps is available, the
    algorithm resorts to the average velocity of the nearest tracked
    objects. v_max and d_min are given as physical quantities and then
    converted into pixel-based values used in trackpy. This allows for
    tracking that is controlled by physically-based parameters that are
    independent of the temporal and spatial resolution of the input
    data. The algorithm creates a continuous track for the feature
    that is the most probable based on the previous cell path.

    Parameters
    ----------
    features : pandas.DataFrame
        Detected features to be linked.

    field_in : xarray.DataArray
        Input field to perform the watershedding on (2D or 3D for one
        specific point in time).

    dt : float
        Time resolution of tracked features.

    dxy : float
        Horizontal grid spacing of the input data.

    dz : float
        Constant vertical grid spacing (m), optional. If not specified
        and the input is 3D, this function requires that `vertical_coord` is available
        in the `features` input. If you specify a value here, this function assumes
        that it is the constant z spacing between points, even if ```vertical_coord```
        is specified.

    d_max : float, optional
        Maximum search range
        Default is None.

    d_min : float, optional
        Variations in the shape of the regions used to determine the
        positions of the features can lead to quasi-instantaneous shifts
        of the position of the feature by one or two grid cells even for
        a very high temporal resolution of the input data, potentially
        jeopardising the tracking procedure. To prevent this, tobac uses
        an additional minimum radius of the search range.
        Default is None.

    subnetwork_size : int, optional
        Maximum size of subnetwork for linking. This parameter should be
        adjusted when using adaptive search. Usually a lower value is desired
        in that case. For a more in depth explanation have look
        `here <https://soft-matter.github.io/trackpy/v0.5.0/tutorial/adaptive-search.html>`_
        If None, 30 is used for regular search and 15 for adaptive search.
        Default is None.

    v_max : float, optional
        Speed at which features are allowed to move. Default is None.

    memory : int, optional
        Number of output timesteps features allowed to vanish for to
        be still considered tracked. Default is 0.
        .. warning :: This parameter should be used with caution, as it
                     can lead to erroneous trajectory linking,
                     espacially for data with low time resolution.

    stubs : int, optional
        Minimum number of timesteps of a tracked cell to be reported
        Default is 1

    time_cell_min : float, optional
        Minimum length in time of tracked cell to be reported in minutes
        Default is None.

    order : int, optional
        Order of polynomial used to extrapolate trajectory into gaps and
        ond start and end point.
        Default is 1.

    extrapolate : int, optional
        Number or timesteps to extrapolate trajectories.
        Default is 0.

    method_linking : {'random', 'predict'}, optional
        Flag choosing method used for trajectory linking.
        Default is 'random'.

    adaptive_step : float, optional
        Reduce search range by multiplying it by this factor. Needs to be
        used in combination with adaptive_stop. Default is None.

    adaptive_stop : float, optional
        If not None, when encountering an oversize subnet, retry by progressively
        reducing search_range by multiplying with adaptive_step until the subnet
        is solvable. If search_range becomes <= adaptive_stop, give up and raise
        a SubnetOversizeException. Needs to be used in combination with
        adaptive_step. Default is None.

    cell_number_start : int, optional
        Cell number for first tracked cell.
        Default is 1

    cell_number_unassigned: int
        Number to set the unassigned/non-tracked cells to. Note that if you set this
        to `np.nan`, the data type of 'cell' will change to float.
        Default is -1

    vertical_coord: str
        Name of the vertical coordinate. The vertical coordinate used
        must be meters. If None, tries to auto-detect.
        It looks for the coordinate or the dimension name corresponding
        to the string. To use `dz`, set this to `None`.

    min_h1: int
        Minimum hdim_1 value, required when PBC_flag is 'hdim_1' or 'both'

    max_h1: int
        Maximum hdim_1 value, required when PBC_flag is 'hdim_1' or 'both'

    min_h2: int
        Minimum hdim_2 value, required when PBC_flag is 'hdim_2' or 'both'

    max_h2: int
        Maximum hdim_2 value, required when PBC_flag is 'hdim_2' or 'both'

    PBC_flag : str('none', 'hdim_1', 'hdim_2', 'both')
        Sets whether to use periodic boundaries, and if so in which directions.
        'none' means that we do not have periodic boundaries
        'hdim_1' means that we are periodic along hdim1
        'hdim_2' means that we are periodic along hdim2
        'both' means that we are periodic along both horizontal dimensions

    Returns
    -------
    trajectories_final : pandas.DataFrame
        Dataframe of the linked features, containing the variable 'cell',
        with integers indicating the affiliation of a feature to a specific
        track, and the variable 'time_cell' with the time the cell has
        already existed.

    Raises
    ------
    ValueError
        If method_linking is neither 'random' nor 'predict'.
    """

    if extrapolate != 0:
        raise NotImplementedError(
            "Extrapolation is not yet implemented. Set this parameter to 0 to continue."
        )

    #    from trackpy import link_df
    #    from trackpy import link_df

    #    from trackpy import filter_stubs
    #    from .utils import add_coordinates

    if (v_max is None) and (d_min is None) and (d_max is None):
        raise ValueError(
            "Neither d_max nor v_max has been provided. Either one of these arguments must be specified."
        )

    # calculate search range based on timestep and grid spacing
    if v_max is not None:
        search_range = dt * v_max / dxy

    # calculate search range based on timestep and grid spacing
    if d_max is not None:
        if v_max is not None:
            raise ValueError(
                "Multiple parameter inputs for v_max, d_max or d_min have been provided. Only use one of these parameters as they supercede each other leading to unexpected behaviour"
            )
        search_range = d_max / dxy

    # calculate search range based on timestep and grid spacing
    if d_min is not None:
        if (v_max is not None) or (d_max is not None):
            raise ValueError(
                "Multiple parameter inputs for v_max, d_max or d_min have been provided. Only use one of these parameters as they supercede each other leading to unexpected behaviour"
            )
        search_range = d_min / dxy
        warnings.warn(
            "d_min parameter will be deprecated in a future version of tobac. Please use d_max instead",
            FutureWarning,
        )
    # Check if we are 3D.
    if "vdim" in features:
        is_3D = True
        if dz is not None and vertical_coord is not None:
            raise ValueError(
                "dz and vertical_coord both set, vertical"
                " spacing is ambiguous. Set one to None."
            )
        if dz is None and vertical_coord is None:
            raise ValueError(
                "Neither dz nor vertical_coord are set. One" " must be set."
            )
        if vertical_coord is not None:
            found_vertical_coord = internal_utils.find_dataframe_vertical_coord(
                variable_dataframe=features, vertical_coord=vertical_coord
            )
    else:
        is_3D = False

    # make sure that we have min and max for h1 and h2 if we are PBC
    if PBC_flag in ["hdim_1", "both"] and (min_h1 is None or max_h1 is None):
        raise ValueError("For PBC tracking, must set min and max coordinates.")

    if PBC_flag in ["hdim_2", "both"] and (min_h2 is None or max_h2 is None):
        raise ValueError("For PBC tracking, must set min and max coordinates.")

    # in case of adaptive search, check wether both parameters are specified
    if adaptive_stop is not None:
        if adaptive_step is None:
            raise ValueError(
                "Adaptive search requires values for adaptive_step and adaptive_stop. Please specify adaptive_step."
            )

    if adaptive_step is not None:
        if adaptive_stop is None:
            raise ValueError(
                "Adaptive search requires values for adaptive_step and adaptive_stop. Please specify adaptive_stop."
            )

    if time_cell_min:
        stubs = np.floor(time_cell_min / dt) + 1

    logging.debug("stubs: " + str(stubs))

    logging.debug("start linking features into trajectories")

    # If subnetwork size given, set maximum subnet size
    if subnetwork_size is not None:
        # Choose the right parameter depending on the use of adaptive search, save previously set values
        if adaptive_step is None and adaptive_stop is None:
            size_cache = tp.linking.Linker.MAX_SUB_NET_SIZE
            tp.linking.Linker.MAX_SUB_NET_SIZE = subnetwork_size
        else:
            size_cache = tp.linking.Linker.MAX_SUB_NET_SIZE_ADAPTIVE
            tp.linking.Linker.MAX_SUB_NET_SIZE_ADAPTIVE = subnetwork_size

    # deep copy to preserve features field:
    features_linking = deepcopy(features)

    # check if we are 3D or not
    if is_3D:
        # If we are 3D, we need to convert the vertical
        # coordinates so that 1 unit is equal to dxy.

        if dz is not None:
            features_linking["vdim_adj"] = features_linking["vdim"] * dz / dxy
        else:
            features_linking["vdim_adj"] = features_linking[found_vertical_coord] / dxy

        pos_columns_tp = ["vdim_adj", "hdim_1", "hdim_2"]

    else:
        pos_columns_tp = ["hdim_1", "hdim_2"]

    # Check if we have PBCs.
    if PBC_flag in ["hdim_1", "hdim_2", "both"]:
        # Per the trackpy docs, to specify a custom distance function
        # which we need for PBCs, neighbor_strategy must be 'BTree'.
        # I think this shouldn't change results, but it will degrade performance.
        neighbor_strategy = "BTree"
        dist_func = build_distance_function(min_h1, max_h1, min_h2, max_h2, PBC_flag)

    else:
        neighbor_strategy = "KDTree"
        dist_func = None

    if method_linking == "random":
        #     link features into trajectories:
        trajectories_unfiltered = tp.link(
            features_linking,
            search_range=search_range,
            memory=memory,
            t_column="frame",
            pos_columns=pos_columns_tp,
            adaptive_step=adaptive_step,
            adaptive_stop=adaptive_stop,
            neighbor_strategy=neighbor_strategy,
            link_strategy="auto",
            dist_func=dist_func,
        )
    elif method_linking == "predict":
        if is_3D and pkgvsn.parse(tp.__version__) < pkgvsn.parse("0.6.0"):
            raise ValueError(
                "3D Predictive Tracking Only Supported with trackpy versions newer than 0.6.0."
            )

        # avoid setting pos_columns by renaming to default values to avoid trackpy bug
        features_linking.rename(
            columns={
                "y": "__temp_y_coord",
                "x": "__temp_x_coord",
                "z": "__temp_z_coord",
            },
            inplace=True,
        )

        features_linking.rename(
            columns={"hdim_1": "y", "hdim_2": "x", "vdim_adj": "z"}, inplace=True
        )

        # generate list of features as input for df_link_iter to avoid bug in df_link
        features_linking_list = [
            frame for i, frame in features_linking.groupby("frame", sort=True)
        ]

        pred = tp.predict.NearestVelocityPredict(span=1)
        trajectories_unfiltered = pred.link_df_iter(
            features_linking_list,
            search_range=search_range,
            memory=memory,
            # pos_columns=["hdim_1", "hdim_2"], # not working atm
            t_column="frame",
            neighbor_strategy=neighbor_strategy,
            link_strategy="auto",
            adaptive_step=adaptive_step,
            adaptive_stop=adaptive_stop,
            # dist_func=dist_func
            #                                 copy_features=False, diagnostics=False,
            #                                 hash_size=None, box_size=None, verify_integrity=True,
            #                                 retain_index=False
        )
        # recreate a single dataframe from the list

        trajectories_unfiltered = pd.concat(trajectories_unfiltered)

        # change to column names back
        trajectories_unfiltered.rename(
            columns={"y": "hdim_1", "x": "hdim_2", "z": "vdim_adj"}, inplace=True
        )
        trajectories_unfiltered.rename(
            columns={
                "__temp_y_coord": "y",
                "__temp_x_coord": "x",
                "__temp_z_coord": "z",
            },
            inplace=True,
        )

    else:
        raise ValueError("method_linking unknown")

    # Reset trackpy parameters to previously set values
    if subnetwork_size is not None:
        if adaptive_step is None and adaptive_stop is None:
            tp.linking.Linker.MAX_SUB_NET_SIZE = size_cache
        else:
            tp.linking.Linker.MAX_SUB_NET_SIZE_ADAPTIVE = size_cache

    # Filter trajectories to exclude short trajectories that are likely to be spurious
    #    trajectories_filtered = filter_stubs(trajectories_unfiltered,threshold=stubs)
    #    trajectories_filtered=trajectories_filtered.reset_index(drop=True)

    # clean up our temporary filters
    if is_3D:
        trajectories_unfiltered = trajectories_unfiltered.drop("vdim_adj", axis=1)

    # Reset particle numbers from the arbitray numbers at the end of the feature detection and linking to consecutive cell numbers
    # keep 'particle' for reference to the feature detection step.
    trajectories_unfiltered["cell"] = None
    particle_num_to_cell_num = dict()
    for i_particle, particle in enumerate(
        pd.Series.unique(trajectories_unfiltered["particle"])
    ):
        cell = int(i_particle + cell_number_start)
        particle_num_to_cell_num[particle] = int(cell)
    remap_particle_to_cell_vec = np.vectorize(remap_particle_to_cell_nv)
    trajectories_unfiltered["cell"] = remap_particle_to_cell_vec(
        particle_num_to_cell_num, trajectories_unfiltered["particle"]
    )
    trajectories_unfiltered["cell"] = trajectories_unfiltered["cell"].astype(int)
    trajectories_unfiltered.drop(columns=["particle"], inplace=True)

    trajectories_bycell = trajectories_unfiltered.groupby("cell")
    stub_cell_nums = list()
    for cell, trajectories_cell in trajectories_bycell:
        # logging.debug("cell: "+str(cell))
        # logging.debug("feature: "+str(trajectories_cell['feature'].values))
        # logging.debug("trajectories_cell.shape[0]: "+ str(trajectories_cell.shape[0]))

        if trajectories_cell.shape[0] < stubs:
            logging.debug(
                "cell"
                + str(cell)
                + "  is a stub ("
                + str(trajectories_cell.shape[0])
                + "), setting cell number to "
                + str(cell_number_unassigned)
            )
            stub_cell_nums.append(cell)

    trajectories_unfiltered.loc[
        trajectories_unfiltered["cell"].isin(stub_cell_nums), "cell"
    ] = cell_number_unassigned

    trajectories_filtered = trajectories_unfiltered

    # Interpolate to fill the gaps in the trajectories (left from allowing memory in the linking)
    trajectories_filtered_unfilled = deepcopy(trajectories_filtered)

    #    trajectories_filtered_filled=fill_gaps(trajectories_filtered_unfilled,order=order,
    #                                extrapolate=extrapolate,frame_max=field_in.shape[0]-1,
    #                                hdim_1_max=field_in.shape[1],hdim_2_max=field_in.shape[2])
    #     add coorinates from input fields to output trajectories (time,dimensions)
    #    logging.debug('start adding coordinates to trajectories')
    #    trajectories_filtered_filled=add_coordinates(trajectories_filtered_filled,field_in)
    #     add time coordinate relative to cell initiation:
    #    logging.debug('start adding cell time to trajectories')
    trajectories_filtered_filled = trajectories_filtered_unfilled
    trajectories_final = add_cell_time(
        trajectories_filtered_filled, cell_number_unassigned=cell_number_unassigned
    )
    # Add metadata
    trajectories_final.attrs["cell_number_unassigned"] = cell_number_unassigned

    # add coordinate to raw features identified:
    logging.debug("start adding coordinates to detected features")
    logging.debug("feature linking completed")

    return trajectories_final


def fill_gaps(
    t, order=1, extrapolate=0, frame_max=None, hdim_1_max=None, hdim_2_max=None
):
    """Add cell time as time since the initiation of each cell.

    Parameters
    ----------
    t : pandas.DataFrame
        Trajectories from trackpy.

    order : int, optional
        Order of polynomial used to extrapolate trajectory into
        gaps and beyond start and end point. Default is 1.

    extrapolate : int, optional
        Number or timesteps to extrapolate trajectories. Default is 0.

    frame_max : int, optional
        Size of input data along time axis. Default is None.

    hdim_1_max, hdim2_max : int, optional
        Size of input data along first and second horizontal axis.
        Default is None.

    Returns
    -------
    t : pandas.DataFrame
        Trajectories from trackpy with with filled gaps and potentially
        extrapolated.
    """

    from scipy.interpolate import InterpolatedUnivariateSpline

    logging.debug("start filling gaps")

    t_list = []  # empty list to store interpolated DataFrames

    # group by cell number and perform process for each cell individually:
    t_grouped = t.groupby("cell")
    for cell, track in t_grouped:
        # Setup interpolator from existing points (of order given as keyword)
        frame_in = track["frame"].values
        hdim_1_in = track["hdim_1"].values
        hdim_2_in = track["hdim_2"].values
        s_x = InterpolatedUnivariateSpline(frame_in, hdim_1_in, k=order)
        s_y = InterpolatedUnivariateSpline(frame_in, hdim_2_in, k=order)

        # Create new index filling in gaps and possibly extrapolating:
        index_min = min(frame_in) - extrapolate
        index_min = max(index_min, 0)
        index_max = max(frame_in) + extrapolate
        index_max = min(index_max, frame_max)
        new_index = range(index_min, index_max + 1)  # +1 here to include last value
        track = track.reindex(new_index)

        # Interpolate to extended index:
        frame_out = new_index
        hdim_1_out = s_x(frame_out)
        hdim_2_out = s_y(frame_out)

        # Replace fields in data frame with
        track["frame"] = new_index
        track["hdim_1"] = hdim_1_out
        track["hdim_2"] = hdim_2_out
        track["cell"] = cell

        # Append DataFrame to list of DataFrames
        t_list.append(track)
    # Concatenate interpolated trajectories into one DataFrame:
    t_out = pd.concat(t_list)
    # Restrict output trajectories to input data in time and space:
    t_out = t_out.loc[
        (t_out["hdim_1"] < hdim_1_max)
        & (t_out["hdim_2"] < hdim_2_max)
        & (t_out["hdim_1"] > 0)
        & (t_out["hdim_2"] > 0)
    ]
    t_out = t_out.reset_index(drop=True)
    return t_out


def add_cell_time(t: pd.DataFrame, cell_number_unassigned: int):
    """add cell time as time since the initiation of each cell

    Parameters
    ----------
    t : pandas.DataFrame
        trajectories with added coordinates
    cell_number_unassigned: int
        unassigned cell value

    Returns
    -------
    t : pandas.Dataframe
        trajectories with added cell time
    """

    # logging.debug('start adding time relative to cell initiation')
    t_grouped = t.groupby("cell")

    t["time_cell"] = t["time"] - t.groupby("cell")["time"].transform("min")
    t["time_cell"] = pd.to_timedelta(t["time_cell"])
    t.loc[t["cell"] == cell_number_unassigned, "time_cell"] = pd.Timedelta("nat")
    return t


def remap_particle_to_cell_nv(particle_cell_map, input_particle):
    """Remaps the particles to new cells given an input map and the current particle.
    Helper function that is designed to be vectorized with np.vectorize

    Parameters
    ----------
    particle_cell_map: dict-like
        The dictionary mapping particle number to cell number
    input_particle: key for particle_cell_map
        The particle number to remap

    """
    return particle_cell_map[input_particle]


def build_distance_function(min_h1, max_h1, min_h2, max_h2, PBC_flag):
    """Function to build a partial ```calc_distance_coords_pbc``` function
    suitable for use with trackpy

    Parameters
    ----------
    min_h1: int
        Minimum point in hdim_1
    max_h1: int
        Maximum point in hdim_1
    min_h2: int
        Minimum point in hdim_2
    max_h2: int
        Maximum point in hdim_2
    PBC_flag : str('none', 'hdim_1', 'hdim_2', 'both')
        Sets whether to use periodic boundaries, and if so in which directions.
        'none' means that we do not have periodic boundaries
        'hdim_1' means that we are periodic along hdim1
        'hdim_2' means that we are periodic along hdim2
        'both' means that we are periodic along both horizontal dimensions

    Returns
    -------
    function object
        A version of calc_distance_coords_pbc suitable to be called by
        just f(coords_1, coords_2)

    """
    import functools

    return functools.partial(
        pbc_utils.calc_distance_coords_pbc,
        min_h1=min_h1,
        max_h1=max_h1,
        min_h2=min_h2,
        max_h2=max_h2,
        PBC_flag=PBC_flag,
    )


def linking_overlap(
    features: pd.DataFrame,
    segmentation_mask: xr.DataArray,
    dt: float,
    dxy: float,
    dz: float = None,
    stubs: int = 1,
    v_max: float = None,
    d_max: float = None,
    cell_number_start: int = 1,
    cell_number_unassigned: int = -1,
    vertical_coord: str = "auto",
    PBC_flag: str = "none",
    min_absolute_overlap: int = 1,
    min_relative_overlap: float = 0.0,
) -> pd.DataFrame:
    """Perform linking of features using the overlap of the segmented areas

    Parameters
    ----------
    features : pd.DataFrame
        _description_
    segmentation_mask : xr.DataArray
        _description_
    dt : float
        _description_
    dxy : float
        _description_
    dz : float, optional
        _description_, by default None
    stubs : int, optional
        _description_, by default 1
    v_max : float, optional
        _description_, by default None
    d_max : float, optional
        _description_, by default None
    d_min : float, optional
        _description_, by default None
    cell_number_start : int, optional
        _description_, by default 1
    cell_number_unassigned : int, optional
        _description_, by default -1
    vertical_coord : str, optional
        _description_, by default "auto"
    PBC_flag : str, optional
        _description_, by default "none"
    min_absolute_overlap : int, optional
        minimum number of pixels in overlapping labels, by default 1
    min_relative_overlap : float, optional
        minimum proportion of labels to overlap, by default 0

    Returns
    -------
    pd.DataFrame
        _description_
    """
    # Initial values
    current_cell = int(cell_number_start)
    features_out = features.copy()
    features_out["cell"] = np.full([len(features)], cell_number_unassigned, dtype=int)

    max_dist = np.inf
    if d_max is not None:
        max_dist = d_max / dxy
    if v_max is not None:
        max_dist = v_max * dt / dxy

    # Run initial link
    current_step = segmentation_mask.isel(time=0)
    next_step = segmentation_mask.isel(time=1)
    features_out, current_cell = linking_overlap_timestep(
        features_out,
        current_step,
        next_step,
        current_cell,
        cell_number_unassigned,
        min_relative_overlap=min_relative_overlap,
        min_absolute_overlap=min_absolute_overlap,
        max_dist=max_dist,
    )

    # Repeat for subsequent time steps
    for time_step in range(1, segmentation_mask.time.size - 1):
        current_step, next_step = next_step, segmentation_mask.isel(time=time_step + 1)
        features_out, current_cell = linking_overlap_timestep(
            features_out,
            current_step,
            next_step,
            current_cell,
            cell_number_unassigned,
            min_relative_overlap=min_relative_overlap,
            min_absolute_overlap=min_absolute_overlap,
            max_dist=max_dist,
        )

    # Now remove stub cells
    features_out = remove_stubs(features_out, stubs, cell_number_unassigned)

    return features_out


def remove_stubs(
    features: pd.DataFrame, stubs: int, cell_number_unassigned: int
) -> pd.DataFrame:
    """Remove cells which have fewer than a given number of features

    Parameters
    ----------
    features : pd.DataFrame
        DataFrame of tracked features
    stubs : int
        the minimum number of features for a valid tracked object
    cell_number_unassigned : int
        the cell value used to indicate that a feature is not part of trackd
        cell

    Returns
    -------
    pd.DataFrame
        DataFrame of tracked features with stub cells removed
    """
    if stubs > 1:
        cell_length = (
            features[features.cell != cell_number_unassigned]
            .groupby(features[features.cell != cell_number_unassigned].cell)
            .apply(len)
        )
        features.loc[
            np.isin(features.cell, cell_length.index[cell_length < stubs]),
            "cell",
        ] = cell_number_unassigned
    return features


def linking_overlap_timestep(
    features: pd.DataFrame,
    current_step: xr.DataArray,
    next_step: xr.DataArray,
    new_cell_value: int,
    stub_cell: int,
    ranking_method: str = "absolute",
    min_relative_overlap: float = 0,
    min_absolute_overlap: int = 1,
    max_dist: float = np.inf,
) -> tuple[pd.DataFrame, int]:
    """Link overlapping features between two consecutive segment masks

    Parameters
    ----------
    features : pd.DataFrame
        Features dataframe
    current_step : xr.DataArray
        segment mask for initial time step
    next_step : xr.DataArray
        segment mask for next time step
    new_cell_value : int
        value for the next cell label
    stub_cell : int
        value given to untracked features
    ranking_method : str, optional
        method used to rank overlap links, by default "absolute"
    min_relative_overlap : float, optional
        minimum proportion of labels to overlap, by default 0
    min_absolute_overlap : int, optional
        minimum number of pixels in overlapping labels, by default 1
    max_dist : float, optional
        maximum distance (in pixels) between the centroids of linked objects, by
        default np.inf

    Returns
    -------
    tuple[pd.DataFrame, int]
        features dataframe with linked cells and updated current_cell value

    Raises
    ------
    ValueError
        if ranking_method is not one of 'absolute', 'relative'
    """

    current_bins = np.bincount(np.maximum(current_step.data.ravel(), 0))
    cumulative_bins = np.cumsum(current_bins)
    args = np.argsort(np.maximum(current_step.data.ravel(), 0))

    next_bins = np.bincount(np.maximum(next_step.data.ravel(), 0))

    link_candidates = np.concatenate(
        list(
            filter(
                None,
                [
                    find_overlapping_labels(
                        label,
                        args[cumulative_bins[label - 1] : cumulative_bins[label]],
                        next_step.data,
                        next_bins,
                        min_relative_overlap=min_relative_overlap,
                        min_absolute_overlap=min_absolute_overlap,
                    )
                    for label in np.intersect1d(features.feature, current_step)
                ],
            )
        )
    )

    # Filter by max distance
    if np.isfinite(max_dist):
        if max_dist <= 0:
            raise ValueError("max_dist must be a positive value")

        features.set_index("feature", drop=False, inplace=True)

        # Need to consider lat/lon distance and PBCs
        wh_too_far = (
            (
                features.loc[link_candidates[:, 0], "hdim_1"]
                - features.loc[link_candidates[:, 1], "hdim_2"]
            )
            ** 2
            + (
                features.loc[link_candidates[:, 0], "hdim_1"].hm1
                - features.loc[link_candidates[:, 1], "hdim_2"]
            )
            ** 2
        ) > max_dist**2

        features.reset_index(drop=True, inplace=True)

        link_candidates = link_candidates[np.logical_not(wh_too_far)]

    if ranking_method == "absolute":
        rank = np.argsort(link_candidates[:, -1])[::-1]
    elif ranking_method == "relative":
        overlap_proportion = calc_proportional_overlap(
            link_candidates[:, -1],
            current_bins[link_candidates[:, 0]],
            next_bins[link_candidates[:, 1]],
        )
        rank = np.argsort(overlap_proportion)[::-1]
    else:
        raise ValueError("ranking method must be one of 'absolute', 'relative")

    current_step_labels = np.intersect1d(features.feature, current_step)
    current_is_linked = dict(
        zip(current_step_labels, np.full(current_step_labels.size, False))
    )

    next_step_labels = np.intersect1d(features.feature, next_step)
    next_is_linked = dict(zip(next_step_labels, np.full(next_step_labels.size, False)))

    for link in rank:
        current_label = link_candidates[link, 0]
        next_label = link_candidates[link, 1]
        if not current_is_linked[current_label] and not next_is_linked[next_label]:
            current_is_linked[current_label] = True
            next_is_linked[next_label] = True

            current_cell = features.loc[
                features.feature == current_label, "cell"
            ].item()
            if current_cell == stub_cell:
                features.loc[features.feature == current_label, "cell"] = new_cell_value
                features.loc[features.feature == next_label, "cell"] = new_cell_value
                new_cell_value += 1
            else:
                features.loc[features.feature == next_label, "cell"] = current_cell

    return features, new_cell_value


def find_overlapping_labels(
    current_label: int,
    locs: np.ndarray[int],
    next_labels: np.ndarray[int],
    next_bins: np.ndarray[int],
    min_relative_overlap: float = 0,
    min_absolute_overlap: int = 1,
) -> list[list[int]]:
    """Find which labels overlap at the locations given by locs, accounting for
    (proportional) overlap and absolute overlap requirements

    Parameters
    ----------
    current_label : int
        the value of the label for which we are finding overlaps
    locs : np.ndarray[int]
        array of array locations (ravelled indexes) in which to search for
        neighbouring labels
    next_labels : np.ndarray[int]
        array of labels in which to find neighbours
    next_bins : np.ndarray[int]
        array of bin locations for each label in next_labels
    min_relative_overlap : float, optional
        minimum proportion of labels to overlap, by default 0
    min_absolute_overlap : int, optional
        minimum number of pixels in overlapping labels, by default 1

    Returns
    -------
    dict[str, int]
        dictionary of neighbouring labels with the number of overlapping pixels
    """

    n_locs = len(locs)
    if n_locs > 0:
        overlap_labels = next_labels.ravel()[locs]
        overlap_bins = np.bincount(np.maximum(overlap_labels, 0))
        return [
            [current_label, new_label, overlap_bins[new_label]]
            for new_label in np.unique(overlap_labels)
            if new_label != 0
            and overlap_bins[new_label] >= min_absolute_overlap
            and calc_proportional_overlap(
                overlap_bins[new_label], n_locs, next_bins[new_label]
            )
            >= min_relative_overlap
        ]
    else:
        return []


def calc_proportional_overlap(
    n_overlapping: int,
    n_feature1: int,
    n_feature2: int,
) -> float:
    """Calculate the proportional overlap between two labels

    Parameters
    ----------
    n_overlapping : int
        number of overlapping pixels
    n_feature1 : int
        number of pixels in feature 1
    n_feature2 : int
        number of pixels in feature number 2

    Returns
    -------
    float
        fraction of total pixels which overlap
    """
    overlap = 2 * n_overlapping / (n_feature1 + n_feature2)

    return overlap
