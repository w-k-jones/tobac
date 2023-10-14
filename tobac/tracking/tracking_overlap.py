"""Overlap tracking methods
"""

import numpy as np
import pandas as pd
import xarray as xr


def linking_overlap(
    features: pd.DataFrame,
    segmentation_mask: xr.DataArray,
    dt: float,
    dxy: float,
    dz: float = None,
    v_max: float = None,
    d_max: float = None,
    stubs: int = 1,
    cell_number_start: int = 1,
    cell_number_unassigned: int = -1,
    vertical_coord: str = "auto",
    PBC_flag: str = "none",
    min_absolute_overlap: int = 1,
    min_relative_overlap: float = 0.0,
    projection_method: None | str = None,
) -> pd.DataFrame:
    """Perform linking of features using the overlap of the segmented areas

    Parameters
    ----------
    features : pd.DataFrame
        Detected features to be linked.
    segmentation_mask : xr.DataArray
        Segmentationb mask of the features to be tracked
    dt : float
        Time resolution of tracked features in seconds.
    dxy : float
        Horizontal grid spacing of the input data in meters.
    dz : float, optional
        Vertical grid spacing (m), by default None
    v_max : float, optional
        Maximum speed of linked features in m/s, by default None
    d_max : float, optional
        Maximum distance moved by linked features in one time step in meters, by
        default None
    stubs : int, optional
        Minimum number of time steps that a cell must be tracked over for it to
        be considered valid, by default 1
    cell_number_start : int, optional
        Cell number for first tracked cell, by default 1
    cell_number_unassigned : int, optional
        Value to set the unassigned/non-tracked cells, by default -1
    vertical_coord : str, optional
        Name of the vertical coordinate, by default "auto"
    PBC_flag : str, optional
        Sets whether to use periodic boundaries, and if so in which directions,
        by default "none"
    min_absolute_overlap : int, optional
        minimum number of pixels in overlapping labels, by default 1
    min_relative_overlap : float, optional
        minimum proportion of labels to overlap, by default 0
    projection_method : None | str, optional
        Method to project the segment locations of features into the next time-
        step. If linear, will use the motion of the cell in the last tracked
        step. By default None

    Returns
    -------
    pd.DataFrame
        Dataframe of the linked features, containing the variable 'cell'
    """

    if PBC_flag in ["hdim_1", "hdim_2", "both"] and projection_method is not None:
        raise ValueError("PBC not yet supported with motion projection")

    if "vdim" in features and projection_method is not None:
        raise ValueError("3D tracking not yet supported with motion projection")

    # Initial values
    current_cell = int(cell_number_start)
    features_out = features.copy()
    features_out["cell"] = np.full([len(features)], cell_number_unassigned, dtype=int)

    max_dist = np.inf
    if d_max is not None:
        max_dist = d_max / dxy
    if v_max is not None:
        max_dist = v_max * dt / dxy

    # Run initial link with no projection method
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

    # TODO: If using motion projection we should repeat the first step so that
    # we get initial estimates for cell motion

    if projection_method is not None:
        start_step = 0
    else:
        start_step = 1

    # Repeat for subsequent time steps
    for time_step in range(start_step, segmentation_mask.time.size - 1):
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
            projection_method=projection_method,
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
    projection_method: None | str = None,
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

    link_candidates = list(
        filter(
            None,
            [
                find_overlapping_labels(
                    label,
                    project_feature_locs(
                        args[cumulative_bins[label - 1] : cumulative_bins[label]],
                        current_step.shape,
                        features,
                        label,
                        stub_cell=stub_cell,
                        projection_method=projection_method,
                    ),
                    next_step.data,
                    next_bins,
                    min_relative_overlap=min_relative_overlap,
                    min_absolute_overlap=min_absolute_overlap,
                )
                for label in np.intersect1d(features.feature, current_step)
            ],
        )
    )

    if link_candidates:
        link_candidates = np.concatenate(link_candidates)

        # Filter by max distance
        if np.isfinite(max_dist):
            if max_dist <= 0:
                raise ValueError("max_dist must be a positive value")

            features.set_index("feature", drop=False, inplace=True)

            # Need to consider lat/lon distance and PBCs
            wh_within_max_dist = (
                (
                    features.loc[link_candidates[:, 0], "hdim_1"].to_numpy()
                    - features.loc[link_candidates[:, 1], "hdim_1"].to_numpy()
                )
                ** 2
                + (
                    features.loc[link_candidates[:, 0], "hdim_2"].to_numpy()
                    - features.loc[link_candidates[:, 1], "hdim_2"].to_numpy()
                )
                ** 2
            ) <= max_dist**2

            features.reset_index(drop=True, inplace=True)

            link_candidates = link_candidates[wh_within_max_dist]

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
        next_is_linked = dict(
            zip(next_step_labels, np.full(next_step_labels.size, False))
        )

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
                    features.loc[
                        features.feature == current_label, "cell"
                    ] = new_cell_value
                    features.loc[
                        features.feature == next_label, "cell"
                    ] = new_cell_value
                    new_cell_value += 1
                else:
                    features.loc[features.feature == next_label, "cell"] = current_cell

    return features, new_cell_value


def project_feature_locs(
    locs: np.ndarray[int],
    shape: tuple[int],
    features: pd.DataFrame,
    label: int,
    stub_cell: int = -1,
    projection_method: None | str = None,
) -> np.ndarray[int]:
    """project the locations of a features segmentation mask according to the
    velocity of the feature

    Parameters
    ----------
    locs : np.ndarray[int]
        the array of (ravelled) array locations of the segment
    shape : tuple[int]
        the shape of the dataset
    features : pd.DataFrame
        the features dataframe, including the feature to project
    label : int
        the label of the feature/segment to be projected
    stub_cell : int, optional
        the value used to represent a stub cell, by default -1
    projection_method : None | str, optional
        the method use to calculate the projection, out of None, 'linear', by
        default None

    Returns
    -------
    np.ndarray[int]
        the array of projected (ravelled) array locations of the segment

    Raises
    ------
    ValueError
        if 'projection_method' is not a valid option
    """
    # TODO: Add 3D support
    if projection_method is None:
        projected_locs = locs
    elif projection_method == "linear":
        offset = calc_cell_velocity(
            features, label, stub_cell=stub_cell, initiation_method=None
        ).astype(int)
        unravelled_inds = np.unravel_index(locs, shape)
        for axis, inds in enumerate(unravelled_inds):
            inds += offset[axis]
        # TODO: add PBC support by wrapping -- need to do this separately for hdim_1, hdim_2
        wh_overhanging = np.logical_or.reduce(
            [
                unravelled_inds[0] < 0,
                unravelled_inds[0] >= shape[0],
                unravelled_inds[1] < 0,
                unravelled_inds[1] >= shape[1],
            ]
        )
        projected_locs = np.ravel_multi_index(unravelled_inds, shape, mode="clip")[
            np.logical_not(wh_overhanging)
        ]
    else:
        raise ValueError("invalid projection method")
    return projected_locs


def calc_cell_velocity(
    features: pd.DataFrame,
    label: int,
    stub_cell: int = -1,
    initiation_method: None | str = None,
):
    # TODO: Add 3D support
    label_cell = features.loc[label, "cell"].item()
    if label_cell == stub_cell:
        if initiation_method is None:
            cell_velocity = np.array([0, 0])
        else:
            raise ValueError("invalid initiation method for cell velocity")
    else:
        wh_cell = features.cell == label_cell
        cell_velocity = np.array(
            [
                np.diff(features.loc[wh_cell, "hdim_1"][-2:]),
                np.diff(features.loc[wh_cell, "hdim_2"][-2:]),
            ]
        )
    return cell_velocity


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
