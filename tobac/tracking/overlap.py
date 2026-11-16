"""Provide overlap tracking methods"""

from functools import partial
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
from scipy.sparse import coo_array
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

from tobac.utils.datetime import to_timestamp
from tobac.utils.generators import field_and_features_over_time
from tobac.utils.overlap import (
    _assign_velocities,
    _find_overlaps,
    _get_coords_and_centroids,
    _transpose_mask,
)


def _maximise_matching_overlaps(
    overlaps: dict[int, tuple[np.ndarray, np.ndarray]],
) -> dict[int, int]:
    """Find the optimum one-to-one feature matching using max weight matching.

    Uses the scipy min_weight_full_bipartite_matching maximum weight matching algorithm to optimise the one-to-one mapping
    of features to maximise total overlap area between timesteps.

    Parameters
    ----------
    overlaps : dict[int, tuple[np.ndarray, np.ndarray]]
        Dictionary of overlap candidates for each feature. Keys are feature IDs,
        values are tuples of (labels, counts) arrays.

    Returns
    -------
    dict[int, int]
        One-to-one mapping of origin features to destination features.

    """
    filtered_overlaps = {k: v for k, v in overlaps.items() if len(v[0]) > 0}
    # if no overlap candidates, return an empty dict
    if not len(filtered_overlaps):
        return {}
    origin_nodes = np.repeat(
        list(filtered_overlaps.keys()), [len(v[0]) for v in filtered_overlaps.values()]
    )
    i_map, i_nodes = np.unique(origin_nodes, return_inverse=True)
    destination_nodes, weights = np.concatenate(
        list(filtered_overlaps.values()), axis=1
    )
    j_map, j_nodes = np.unique(destination_nodes, return_inverse=True)

    total_nodes = j_nodes.max() + 1

    # need to add null "padding" nodes to ensure maximum matching is possible
    padding_nodes = np.arange(total_nodes, total_nodes + i_nodes.max() + 1, dtype=int)
    size = total_nodes + padding_nodes.size

    weights = np.concatenate([weights, np.full((padding_nodes.size,), -1e-15)])
    i = np.concatenate([i_nodes, np.unique(i_nodes)])
    j = np.concatenate([j_nodes, padding_nodes])

    sparse_graph = coo_array(
        (weights, (i, j)),
        shape=(i_nodes.max() + 1, size),
    )

    # maximal_matching = maximum_bipartite_matching(sparse_graph.T)

    i_ind, j_ind = min_weight_full_bipartite_matching(sparse_graph, maximize=True)

    if i_ind.size:
        # remove any null nodes
        wh_null = j_ind >= total_nodes
        i_ind = i_ind[~wh_null]
        j_ind = j_ind[~wh_null]
        return dict(zip(i_map[i_ind], j_map[j_ind]))

    # if no matches return empty dict
    return {}


def _assign_cells_to_matches(tracks: pd.DataFrame, matches: dict[int, int]) -> None:
    """Assign cell identities to matched features.

    Parameters
    ----------
    tracks : pd.DataFrame
        Tracks dataframe with 'cell' column to update.
    matches : dict[int, int]
        Mapping of origin feature IDs to destination feature IDs.

    Returns
    -------
    None
        Modifies tracks DataFrame in place.

    """
    prior_cells = tracks.loc[matches.keys(), "cell"]
    wh_unassigned = prior_cells == 0
    prior_cells[wh_unassigned] = (
        np.arange(wh_unassigned.sum(), dtype=int) + tracks.cell.max() + 1
    )
    tracks.loc[matches.keys(), "cell"] = prior_cells.values
    tracks.loc[matches.values(), "cell"] = prior_cells.values


def _bootstrap_velocities(
    origin_features: pd.DataFrame,
    destination_features: pd.DataFrame,
    destination_labels: xr.DataArray,
    timestep: pd.Timestamp,
    min_count: int = 1,
    relative_count: float = 0,
    PBC_flag: Optional[Literal["none", "hdim_1", "hdim_2", "both"]] = None,
) -> None:
    """Estimate initial feature velocities from overlap matches.

    Parameters
    ----------
    origin_features : pd.DataFrame
        Features in the current timestep.
    destination_features : pd.DataFrame
        Features in the destination timestep.
    destination_labels : xr.DataArray
        Label mask for the destination timestep.
    timestep : pd.Timestamp
        Timestamp associated with the destination timestep.
    min_count : int, optional
        Minimum pixel overlap required for a match. Default is 1.
    relative_count : float, optional
        Minimum relative overlap fraction required for a match. Default is 0.
    PBC_flag : {'none', 'hdim_1', 'hdim_2', 'both'}, optional
        Periodic boundary condition specification.

    Returns
    -------
    None
        Updates the ``_track_velocity`` column of the features dataframes in-place.

    """
    matched_overlaps = _maximise_matching_overlaps(
        _find_overlaps(
            origin_features,
            destination_features,
            destination_labels,
            timestep,
            min_count=min_count,
            relative_count=relative_count,
        )
    )
    _assign_velocities(
        origin_features,
        destination_features,
        matched_overlaps,
        destination_labels.shape,
        PBC_flag=PBC_flag,
        prior=True,
    )


def _filter_stub_cells(
    tracks: pd.DataFrame,
    stubs: int,
    cell_number_start: int,
    cell_number_unassigned: int,
    time_cell_min: Optional[pd.Timedelta] = None,
) -> pd.DataFrame:
    """Remove cells with fewer than minimum number of time steps.

    Parameters
    ----------
    tracks : pd.DataFrame
        Tracks dataframe with 'cell' column.
    stubs : int
        Minimum number of time steps for a cell to be retained.
    cell_number_start : int
        Starting value for cell ID numbering.
    cell_number_unassigned : int
        Value to assign to unassigned/filtered cells.
    time_cell_min : pd.Timedelta, optional
        Minimum time span of a cell (latest minus earliest timestep of the
        cell, `time.max() - time.min()`) for the cell to be retained. Cells
        spanning less time are marked as unassigned. Default is None (no
        minimum time span).

    Returns
    -------
    pd.DataFrame
        Tracks dataframe with stub cells removed and cell IDs renumbered.

    """
    # Ensure all cell values are zero or greater (they should be already)
    tracks["cell"] = np.maximum(tracks.cell.values, 0)
    if stubs > 1:
        cell_count = tracks.groupby("cell").cell.count()
        stub_cells = cell_count.index[cell_count < stubs].values
        tracks.loc[np.isin(tracks.cell, stub_cells), "cell"] = 0

    if time_cell_min is not None:
        cell_time = (
            tracks.groupby("cell").time.max() - tracks.groupby("cell").time.min()
        )
        short_cells = cell_time.index[cell_time < time_cell_min].values
        tracks.loc[np.isin(tracks.cell, short_cells), "cell"] = 0

    new_cells = np.unique(tracks.cell, return_inverse=True)[1]
    if (tracks.cell == 0).any():
        new_cells = np.where(
            new_cells > 0, new_cells + cell_number_start - 1, cell_number_unassigned
        )
    else:
        new_cells = new_cells + cell_number_start

    tracks["cell"] = new_cells

    return tracks


def _assign_cell_times(
    tracks: pd.DataFrame, cell_number_unassigned: int
) -> pd.DataFrame:
    """Calculate time since start of each cell.

    Parameters
    ----------
    tracks : pd.DataFrame
        Tracks dataframe with 'cell' and 'time' columns.
    cell_number_unassigned : int
        Value assigned to unassigned cells.

    Returns
    -------
    pd.DataFrame
        Tracks dataframe with new 'time_cell' column containing time
        relative to cell start.

    """
    tracks["time_cell"] = (
        tracks.time - tracks.groupby("cell").time.min()[tracks.cell.values].values
    )
    tracks.loc[tracks.cell == cell_number_unassigned, "time_cell"] = pd.Timedelta("nat")
    return tracks


def linking_overlap(
    features: pd.DataFrame,
    mask: xr.DataArray,
    stubs: int = 1,
    cell_number_start: int = 1,
    cell_number_unassigned: int = -1,
    minimum_overlap: int = 1,
    minimum_relative_overlap: float = 0,
    translate_method: Optional[Literal["constant", "drift", "predict"]] = None,
    velocity_method: Optional[Literal["constant", "mean", "nearest"]] = None,
    velocity_constant: Optional[Union[float, np.ndarray]] = None,
    PBC_flag: Optional[Literal["none", "hdim_1", "hdim_2", "both"]] = None,
    vertical_axis: Optional[int] = None,
    vertical_coord: Optional[str] = None,
    memory: int = 0,
    max_time_gap: Optional[Union[pd.Timedelta, float]] = None,
    time_cell_min: Optional[Union[pd.Timedelta, float]] = None,
) -> pd.DataFrame:
    """Link features through time using spatial overlap tracking.

    Tracks features across consecutive time steps by finding the best
    one-to-one matches based on spatial overlap, optionally translating
    features based on velocity predictions. Unlike `linking_trackpy`,
    variable time gaps between timesteps are supported.

    Parameters
    ----------
    features : pd.DataFrame
        Features dataframe with 'feature' index and position columns
        ('hdim_1', 'hdim_2', and optionally 'vdim').
    mask : xr.DataArray
        Time series of labeled feature masks with coordinates matching features.
    stubs : int, optional
        Minimum number of timesteps for a cell to be retained. Default is 1.
    cell_number_start : int, optional
        Starting value for cell numbering. Default is 1.
    cell_number_unassigned : int, optional
        Value for unassigned features. Default is -1.
    minimum_overlap : int, optional
        Minimum pixel overlap required for feature linking. Default is 1.
    minimum_relative_overlap : float, optional
        Minimum fraction of the smaller of the two features required to overlap
        for linking. Default is 0.
    translate_method : {'constant', 'drift', 'predict'}, optional
        Method for predicting feature positions at next time step. Default is None.
    velocity_method : {'constant', 'mean', 'nearest'}, optional
        Method for velocity estimation if translate_method is used.
    velocity_constant : float or np.ndarray, optional
        Constant velocity for translation. Used if translate_method is 'constant'.
    PBC_flag : {'none', 'hdim_1', 'hdim_2', 'both'}, optional
        Periodic boundary condition specification.
    vertical_coord : str, optional
        Name of the vertical coordinate. If None, tries to auto-detect.
        It looks for the coordinate or the dimension name corresponding
        to the string.
    vertical_axis : int, optional
        The vertical axis number of the data. If None, uses vertical_coord
        to determine axis. This must be >=0.
    memory : int, optional
        Number of output timesteps features allowed to vanish for to
        be still considered tracked. Default is 0.
        See `max_time_gap` for limiting time gaps between timesteps.

        .. warning::
            This parameter should be used with caution, as it
            can lead to erroneous trajectory linking,
            especially for data with low time resolution.

    max_time_gap : pd.Timedelta or float, optional
        Maximum allowed time gap between a feature and the timestep it is
        matched to: before matching, origin features whose time difference to
        the current timestep exceeds `max_time_gap` are discarded, so tracks
        are not extended across time gaps larger than `max_time_gap`. This
        also applies to features kept as linking candidates via `memory`, so
        a cell cannot contain gaps between consecutive features that are
        larger than `max_time_gap`. If a float is given, it is interpreted as
        seconds. Default is None (no limit).
    time_cell_min : pd.Timedelta or float, optional
        Minimum time span of a cell (latest minus earliest timestep of the
        cell) for the cell to be retained; cells spanning less time are
        marked as unassigned. Like `stubs`, but based on the actual time span
        instead of the number of timesteps, so it also works with variable
        time gaps between timesteps (unlike `linking_trackpy`, which assumes
        a constant time resolution). If a float is given, it is interpreted
        as seconds. Default is None (no minimum time span).

    Returns
    -------
    pd.DataFrame
        Features dataframe with added columns:
        - 'cell': Cell identifier for tracked features
        - 'time_cell': Time relative to start of cell
        Features not linked to any cell are assigned cell_number_unassigned.

    """
    tracks = features.copy().set_index("feature")
    tracks["cell"] = 0

    if isinstance(max_time_gap, (int, float)):
        max_time_gap = pd.to_timedelta(max_time_gap, unit="s")
    if isinstance(time_cell_min, (int, float)):
        time_cell_min = pd.to_timedelta(time_cell_min, unit="s")

    mask = _transpose_mask(
        mask, vertical_coord=vertical_coord, vertical_axis=vertical_axis
    )

    field_and_features = iter(field_and_features_over_time(mask, tracks))

    _, _, labels, features_t = next(field_and_features)
    origin_features = _get_coords_and_centroids(features_t, labels, PBC_flag=PBC_flag)

    bootstrap = True if translate_method in ["drift", "predict"] else False
    matched_overlaps = {}
    for frame, timestep, labels, features_t in field_and_features:
        destination_features = _get_coords_and_centroids(
            features_t, labels, PBC_flag=PBC_flag
        )

        if max_time_gap is not None:
            origin_features = origin_features[
                (to_timestamp(timestep) - to_timestamp(origin_features.time))
                <= max_time_gap
            ]

        if len(origin_features) and len(destination_features):
            if bootstrap:
                _bootstrap_velocities(
                    origin_features,
                    destination_features,
                    labels,
                    timestep,
                    min_count=minimum_overlap,
                    relative_count=minimum_relative_overlap,
                    PBC_flag=PBC_flag,
                )
                bootstrap = False

            matched_overlaps = _maximise_matching_overlaps(
                _find_overlaps(
                    origin_features,
                    destination_features,
                    labels,
                    timestep,
                    min_count=minimum_overlap,
                    relative_count=minimum_relative_overlap,
                    translate_method=translate_method,
                    velocity_method=velocity_method,
                    velocity_constant=velocity_constant,
                    PBC_flag=PBC_flag,
                )
            )
            _assign_cells_to_matches(tracks, matched_overlaps)
            if translate_method in ["drift", "predict"]:
                _assign_velocities(
                    origin_features,
                    destination_features,
                    matched_overlaps,
                    labels.shape,
                    PBC_flag=PBC_flag,
                )
        else:
            if not len(origin_features):
                bootstrap = (
                    True if translate_method in ["drift", "predict"] else False
                )  # if no features, we need to bootstrap next time we have features

        if (memory > 0) and len(origin_features):
            # Merge untracked features from previous frames (that are within the memory limit) with current destination features for next iteration
            origin_features = pd.concat(
                [
                    origin_features[
                        np.logical_and(
                            (frame - origin_features.frame) <= memory,
                            np.logical_not(
                                np.isin(
                                    origin_features.index.values,
                                    list(matched_overlaps.keys()),
                                )
                            ),
                        )
                    ],
                    destination_features,
                ],
            )
        else:
            origin_features = destination_features

    tracks = _filter_stub_cells(
        tracks,
        stubs=stubs,
        cell_number_start=cell_number_start,
        cell_number_unassigned=cell_number_unassigned,
        time_cell_min=time_cell_min,
    )

    tracks = _assign_cell_times(tracks, cell_number_unassigned=cell_number_unassigned)

    # Reset index to match features input and replace features column in the correct location
    tracks = tracks.set_index(features.index)
    tracks.insert(features.columns.get_loc("feature"), "feature", features.feature)

    return tracks
