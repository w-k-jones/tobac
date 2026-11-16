"""Provide merge/split method based on spatial overlap of feature masks between time steps"""

from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
from scipy.sparse import coo_array, csgraph

from tobac.utils.overlap import (
    _assign_velocities,
    _find_overlaps,
    _get_coords_and_centroids,
    _transpose_mask,
)
from tobac.utils.generators import field_and_features_over_time


def _assign_track_times(
    tracks: pd.DataFrame, track_number_unassigned: int
) -> pd.DataFrame:
    """Calculate time since start of each track.

    Parameters
    ----------
    tracks : pd.DataFrame
        Tracks dataframe with 'track' and 'time' columns.
    cell_number_unassigned : int
        Value assigned to unassigned tracks.

    Returns
    -------
    pd.DataFrame
        Tracks dataframe with new 'time_cell' column containing time
        relative to cell start.

    """
    tracks["time_track"] = (
        tracks.time - tracks.groupby("track").time.min()[tracks.track.values].values
    )
    tracks.loc[tracks.track == track_number_unassigned, "time_track"] = pd.Timedelta(
        "nat"
    )
    return tracks


def merge_split_overlap(
    tracks: pd.DataFrame,
    mask: xr.DataArray,
    cell_number_unassigned: int = -1,
    track_number_start: int = 1,
    track_number_unassigned: int = -1,
    track_unassigned_cells: bool = False,
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
) -> pd.DataFrame:
    """Detect merges/splits of tracked cells based on spatial overlap of feature
    masks between time steps. Unlike `linking_overlap`, this function calculates
    a many-to-many mapping of features between time steps, and then groups the
    cells into tracks based on connected components.

    Parameters
    ----------
    tracks : pd.DataFrame
        Dataframe of features with columns 'feature', 'cell', and 'time', as
        produced by linking.
    mask : xr.DataArray
        The feature mask corresponding to the features in `tracks`, produced
        either by segmentation or from feature detection with `return_labels=True`.
    cell_number_unassigned : int, optional
        The cell number used for unassigned in linking, by default -1
    track_number_start : int, optional
        The track number given to the first detected track, by default 1
    track_number_unassigned : int, optional
        The track number for features with no track detected, by default -1
    track_unassigned_cells : bool, optional
        Whether to include features that are not assigned a cell in merges/splits
        if they overlap a tracked feature, by default False
    minimum_overlap : int, optional
        The minimum number of pixels that two features must overlap to be considered
        as a potential merge or split, by default 1
    minimum_relative_overlap : float, optional
        The minimum relative overlap between two features to be considered as a
        potential merge or split as a fraction of the area of the smaller feature,
        by default 0
    translate_method : Optional[Literal["constant", "drift", "predict"]], optional
        The method to use for translating features between time steps, by default None
    velocity_method : Optional[Literal["constant", "mean", "nearest"]], optional
        The method to use for calculating velocity for untracked features when
        translate method is "drift" or "predict", by default None
    velocity_constant : Optional[Union[float, np.ndarray]], optional
        The constant velocity to use when translate method is "constant" or when
        velocity_method is "constant" in grid spaces per second, by default None
    PBC_flag : Optional[Literal["none", "hdim_1", "hdim_2", "both"]], optional
        Whether to detect merges/splits on features that translate across periodic
        boundaries, by default None
    vertical_axis : Optional[int], optional
        The vertical axis in the input mask, by default None
    vertical_coord : Optional[str], optional
        The vertical coordinate name in the input mask, by default None
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

    Returns
    -------
    pd.DataFrame
        The input `tracks` dataframe with an additional column 'track' containing the
        track number assigned to each feature based on merges/splits. Features that
        are not assigned to a track will have the value `track_number_unassigned` in
        the 'track' column.
    """

    assert (
        "cell" in tracks.columns
    ), "tracks input is missing 'cell' column, please perform tracking before calculating merge/splits"
    tracks_in = tracks
    tracks = tracks.copy().set_index("feature")

    if isinstance(max_time_gap, (int, float)):
        max_time_gap = pd.to_timedelta(max_time_gap, unit="s")

    if track_unassigned_cells:
        original_max_cell = tracks.cell.max()
        wh_cell_unassigned = tracks.cell == cell_number_unassigned
        tracks.loc[wh_cell_unassigned, "cell"] = (
            np.arange(wh_cell_unassigned.sum(), dtype=tracks.cell.dtype)
            + original_max_cell
            + 1
        )
    max_cell = tracks.cell.max()

    tracks["track"] = track_number_unassigned

    # Return early if no tracks are present
    if not len(tracks):
        tracks = _assign_track_times(
            tracks, track_number_unassigned=track_number_unassigned
        )
        tracks = tracks.set_index(tracks_in.index)
        tracks.insert(
            tracks_in.columns.get_loc("feature"), "feature", tracks_in.feature
        )
        return tracks

    mask = _transpose_mask(
        mask, vertical_coord=vertical_coord, vertical_axis=vertical_axis
    )

    field_and_features = iter(field_and_features_over_time(mask, tracks))

    _, _, labels, features_t = next(field_and_features)
    features_t = features_t[features_t.cell != cell_number_unassigned]
    origin_features = _get_coords_and_centroids(features_t, labels, PBC_flag=PBC_flag)

    matched_overlaps = {}
    origin_cells = []
    destination_cells = []
    for frame, timestep, labels, features_t in field_and_features:
        features_t = features_t[features_t.cell != cell_number_unassigned]
        destination_features = _get_coords_and_centroids(
            features_t, labels, PBC_flag=PBC_flag
        )

        if max_time_gap is not None:
            origin_features = origin_features[
                (to_timestamp(timestep) - to_timestamp(origin_features.time))
                <= max_time_gap
            ]

        if len(origin_features) and len(destination_features):
            # Assign velocities from existing tracks
            cell_overlap = np.intersect1d(
                origin_features.cell[origin_features.cell != cell_number_unassigned],
                destination_features.cell[
                    destination_features.cell != cell_number_unassigned
                ],
            )

            matches = dict(
                zip(
                    origin_features.reset_index()
                    .set_index("cell")
                    .loc[cell_overlap, "feature"],
                    destination_features.reset_index()
                    .set_index("cell")
                    .loc[cell_overlap, "feature"],
                )
            )

            _assign_velocities(
                origin_features,
                destination_features,
                matches,
                labels.shape,
                PBC_flag=PBC_flag,
                prior=True,
            )

            matched_overlaps = _find_overlaps(
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

            # TODO: make this a function
            matched_overlaps = matched_overlaps.apply(lambda df: df[0])
            n_overlaps = matched_overlaps.apply(len)

            origin_cell_matches = origin_features.cell.loc[
                np.repeat(matched_overlaps.index.values, n_overlaps)
            ].values
            destination_cell_matches = destination_features.cell.loc[
                np.concatenate(matched_overlaps.values)
            ].values
            wh_different = origin_cell_matches != destination_cell_matches
            origin_cells.append(origin_cell_matches[wh_different])
            destination_cells.append(destination_cell_matches[wh_different])

        # construct new origin_features
        if (memory > 0) and len(origin_features):
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

    # Now perform connected components using cell_map to create tracks
    origin_cells = (
        np.concatenate(origin_cells) if len(origin_cells) else np.array([], dtype=int)
    )
    destination_cells = (
        np.concatenate(destination_cells)
        if len(destination_cells)
        else np.array([], dtype=int)
    )

    cell_graph = coo_array(
        (np.ones(origin_cells.size), (origin_cells, destination_cells)),
        shape=(max_cell + 1, max_cell + 1),
    )
    cell_to_track_map = csgraph.connected_components(cell_graph)[
        1
    ]  # + track_number_start - 1

    if track_unassigned_cells:
        # Remove tracks where there is no valid cell is involved
        cell_to_track_map[original_max_cell + 1 :][
            cell_to_track_map[original_max_cell + 1 :]
            > cell_to_track_map[original_max_cell]
        ] = track_number_unassigned

    # reorder to contiguous values
    wh_cell_valid = tracks.cell != cell_number_unassigned
    valid_cells = np.unique(tracks[wh_cell_valid].cell)
    valid_cells = valid_cells[cell_to_track_map[valid_cells] != track_number_unassigned]
    cell_to_track_map[valid_cells] = (
        np.unique(cell_to_track_map[valid_cells], return_inverse=True)[1]
        + track_number_start
    )

    tracks.loc[wh_cell_valid, "track"] = cell_to_track_map[
        tracks.loc[wh_cell_valid, "cell"]
    ]

    if track_unassigned_cells:
        tracks.loc[tracks.cell > original_max_cell, "cell"] = cell_number_unassigned

    tracks = _assign_track_times(
        tracks, track_number_unassigned=track_number_unassigned
    )

    # Reset index to match features input and replace features column in the correct location
    tracks = tracks.set_index(tracks_in.index)
    tracks.insert(tracks_in.columns.get_loc("feature"), "feature", tracks_in.feature)
    return tracks
