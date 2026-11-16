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
) -> pd.DataFrame:

    assert (
        "cell" in tracks.columns
    ), "tracks input is missing 'cell' column, please perform tracking before calculating merge/splits"
    tracks_in = tracks
    index_name = tracks.index.name or "index"
    tracks = tracks.copy().set_index("feature")

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
