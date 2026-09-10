"""Provide overlap tracking methods"""

import datetime
from typing import Generator, Literal, Optional, Union

import cftime
import numpy as np
import pandas as pd
import xarray as xr
import networkx as nx
import skimage.measure
from scipy.sparse import coo_array
from sklearn.neighbors import BallTree

from tobac.utils.datetime import to_timestamp
from tobac.utils.generators import field_and_features_over_time
from tobac.utils.periodic_boundaries import build_distance_function


def _get_paired_field_and_features_iterator(
    Mask: xr.DataArray, Features: pd.DataFrame
) -> Generator[
    tuple[
        tuple[
            int,
            Union[datetime.datetime, np.datetime64, cftime.datetime],
            xr.DataArray,
            pd.DataFrame,
        ],
        tuple[
            int,
            Union[datetime.datetime, np.datetime64, cftime.datetime],
            xr.DataArray,
            pd.DataFrame,
        ],
    ],
    None,
    None,
]:
    origin_iterator = field_and_features_over_time(Mask, Features)
    destination_iterator = field_and_features_over_time(Mask, Features)
    _ = next(destination_iterator)
    return zip(origin_iterator, destination_iterator)


def _get_indices_from_labels(
    labels: np.ndarray,
) -> tuple[dict[int, np.ndarray[int]], dict[int, int]]:
    """Function to get the x, y, and z indices (as well as point count) of all labeled regions.
    Slightly less deranged than the version in internal utils.

    Parameters
    ----------
    labels : np.ndarray
        The array of labels to get the indices for.
    Returns
    -------
    counts : dict
        The number of points in the label number (key: label number).
    coordinates : dict
        The coordinates in the label number. This is either a 2 or 3 | n array for each label
    """

    counts = {}
    coordinates = {}

    # loop through all skimage identified regions
    region_props = skimage.measure.regionprops(labels)
    for region_prop in region_props:
        coordinates[region_prop.label] = region_prop.coords.T
        counts[region_prop.label] = region_prop.coords.shape[0]

    return counts, coordinates


def _unique_nonzero(arr: np.ndarray, **kwargs) -> np.ndarray:
    return np.unique(arr[arr != 0], **kwargs)


def _find_overlaps_for_label(
    coords: np.ndarray[float],
    counts: int,
    destination_labels: np.ndarray[int],
    min_count: int = 1,
    relative_count: float = 0,
) -> tuple[np.ndarray[int], np.ndarray[int]]:
    matched_labels, matched_counts = _unique_nonzero(
        destination_labels.values[*coords], return_counts=True
    )

    wh = np.logical_and(
        matched_counts >= min_count, matched_counts / counts >= relative_count
    )

    return matched_labels[wh], matched_counts[wh]


def _maximise_matching_overlaps(
    overlaps: dict[int, tuple[np.ndarray, np.ndarray]],
) -> dict[int, int]:
    """_summary_

    Args:
        overlaps (dict[int, tuple[np.ndarray, np.ndarray]]): _description_

    Returns:
        dict[int, int]: _description_
    """
    filtered_overlaps = {k: v for k, v in overlaps.items() if len(v[0]) > 0}
    origin_nodes = np.repeat(
        list(overlaps.keys()), [len(v[0]) for v in overlaps.values()]
    )
    destination_nodes, weights = np.concatenate(
        list(filtered_overlaps.values()), axis=1
    )

    min_node = min(origin_nodes.min(), destination_nodes.min())
    max_node = max(origin_nodes.max(), destination_nodes.max())
    size = max_node - min_node + 1

    nx_graph = nx.from_scipy_sparse_array(
        coo_array(
            (weights, (origin_nodes - min_node, destination_nodes - min_node)),
            shape=(size, size),
        ),
    )

    matching = np.asarray(list(nx.max_weight_matching(nx_graph))).T + min_node

    matching = matching[:, np.isin(matching[0], origin_nodes)]
    matching = matching[:, np.argsort(matching[0])]

    return dict(zip(*matching))


class FeatureBallTree(BallTree):
    def __init__(
        self,
        features: pd.DataFrame,
        PBC_flag: Union[None, Literal["none", "hdim_1", "hdim_2", "both"]] = None,
        min_h1: int = 0,
        max_h1: int = 0,
        min_h2: int = 0,
        max_h2: int = 0,
        **kwargs,
    ) -> None:
        self.is_3D = "vdim" in features.columns
        self.index = features.index.values
        if PBC_flag in ["hdim_1", "hdim_2", "both"]:
            kwargs["metric"] = "pyfunc"
            kwargs["func"] = build_distance_function(
                min_h1, max_h1, min_h2, max_h2, PBC_flag, self.is_3D
            )
        super().__init__(self._get_feature_locations(features), **kwargs)

    def _get_feature_locations(self, features: pd.DataFrame) -> np.ndarray:
        assert (
            "vdim" in features.columns
        ) == self.is_3D, "Query features must match dimensionality of original features"
        return (
            features[["vdim", "hdim_1", "hdim_2"]].to_numpy()
            if self.is_3D
            else features[["hdim_1", "hdim_2"]].to_numpy()
        )

    def query(self, features: pd.DataFrame, *args, **kwargs) -> np.ndarray:
        query_result = super().query(
            self._get_feature_locations(features), *args, **kwargs
        )
        if isinstance(query_result, tuple):
            return (
                self.index[query_result[1]],
                query_result[0],
            )  # flip around results to match query_radius
        return self.index[query_result]

    def query_radius(self, features: pd.DataFrame, *args, **kwargs) -> np.ndarray:
        query_result = super().query_radius(
            self._get_feature_locations(features), *args, **kwargs
        )
        if isinstance(query_result, tuple):
            return (
                np.array([self.index[inds] for inds in query_result[0]], dtype=object),
                query_result[1],
            )
        return np.array([self.index[inds] for inds in query_result], dtype=object)


def _update_predicted_velocities(
    features: pd.DataFrame,
    velocity_method: Union[None, Literal["constant", "mean", "nearest"]] = "constant",
    velocity_constant: Union[None, float, np.ndarray] = 0,
) -> None:
    wh_missing_vels = features._track_velocity.isna()
    if wh_missing_vels.any():
        if velocity_method == "constant":
            features.loc[wh_missing_vels, "_track_velocity"] = (
                {k: velocity_constant for k in features[wh_missing_vels].index}
                if hasattr(velocity_constant, "__iter__")
                else velocity_constant
            )
        elif velocity_method is None or wh_missing_vels.all():
            features.loc[wh_missing_vels, "_track_velocity"] = 0
        elif velocity_method == "mean":
            features.loc[wh_missing_vels, "_track_velocity"] = (
                features._track_velocity.mean()
            )
        elif velocity_method == "nearest":
            # create BallTree to find nearest velocity accounting for PBCs
            btree = FeatureBallTree(features[~wh_missing_vels])
            features.loc[wh_missing_vels, "_track_velocity"] = features._track_velocity[
                ~wh_missing_vels
            ][
                btree.query(features[wh_missing_vels], return_distance=False).ravel()
            ].values


def _wrap_coords(
    coords: np.ndarray[int],
    hdim1_size: int,
    hdim2_size: int,
    vdim_size: Optional[int] = None,
    PBC_flag: Optional[Literal["none", "hdim_1", "hdim_2", "both"]] = None,
) -> np.ndarray[int]:
    if PBC_flag in ["hdim_1", "both"]:
        coords[-2] = coords[-2] % hdim1_size
    if PBC_flag in ["hdim_2", "both"]:
        coords[-1] = coords[-1] % hdim2_size
    filter = np.logical_and(coords[-2] < hdim1_size, coords[-1] < hdim2_size)
    if vdim_size is not None:
        filter = np.logical_and(filter, coords[0] < vdim_size)
    filter = np.logical_and(filter, np.any(coords >= 0, axis=0))
    return coords[:, filter]


def _translate_labels(
    tracks: pd.DataFrame,
    label_coords: dict[int, np.ndarray[int]],
    delta_t: float,
    translate_method: Optional[Literal["constant", "drift", "predict"]] = None,
    velocity_method: Optional[Literal["constant", "mean", "nearest"]] = None,
    velocity_constant: Optional[Union[float, np.ndarray]] = None,
    PBC_flag: Optional[Literal["none", "hdim_1", "hdim_2", "both"]] = None,
    hdim1_size: Optional[int] = None,
    hdim2_size: Optional[int] = None,
    vdim_size: Optional[int] = None,
) -> dict[int, np.ndarray[int]]:
    if translate_method == "predict":
        _update_predicted_velocities(
            tracks, velocity_method=velocity_method, velocity_constant=velocity_constant
        )
        return {
            k: _wrap_coords(
                (
                    label_coords[k].T
                    + np.round(tracks._track_velocity[k] * delta_t).astype(int)
                ).T,
                hdim1_size=hdim1_size,
                hdim2_size=hdim2_size,
                vdim_size=vdim_size,
                PBC_flag=PBC_flag,
            )
            for k in label_coords
        }
    if translate_method == "drift":
        translation = np.round(tracks._track_velocity.mean() * delta_t).astype(int)
        return {
            k: _wrap_coords(
                (label_coords[k].T + translation).T,
                hdim1_size=hdim1_size,
                hdim2_size=hdim2_size,
                vdim_size=vdim_size,
                PBC_flag=PBC_flag,
            )
            for k in label_coords
        }
    if translate_method == "constant":
        translation = np.round(velocity_constant * delta_t).astype(int)
        return {
            k: _wrap_coords(
                (label_coords[k].T + translation).T,
                hdim1_size=hdim1_size,
                hdim2_size=hdim2_size,
                vdim_size=vdim_size,
                PBC_flag=PBC_flag,
            )
            for k in label_coords
        }
    return label_coords


def _find_overlaps(
    tracks: pd.DataFrame,
    origin_labels: xr.DataArray,
    destination_labels: xr.DataArray,
    min_count: int = 1,
    relative_count: float = 0,
    delta_t: int = 0,
    translate_method: Optional[Literal["constant", "drift", "predict"]] = None,
    velocity_method: Optional[Literal["constant", "mean", "nearest"]] = None,
    velocity_constant: Optional[Union[float, np.ndarray]] = None,
    PBC_flag: Optional[Literal["none", "hdim_1", "hdim_2", "both"]] = None,
    hdim1_size: Optional[int] = None,
    hdim2_size: Optional[int] = None,
    vdim_size: Optional[int] = None,
) -> dict[int, int]:
    label_counts, label_coords = _get_indices_from_labels(origin_labels.values)
    if translate_method is not None:
        label_coords = _translate_labels(
            tracks,
            label_coords,
            delta_t,
            translate_method=translate_method,
            velocity_method=velocity_method,
            velocity_constant=velocity_constant,
            PBC_flag=PBC_flag,
            hdim1_size=hdim1_size,
            hdim2_size=hdim2_size,
            vdim_size=vdim_size,
        )
    overlap_candidates = {
        k: _find_overlaps_for_label(
            label_coords[k],
            label_counts[k],
            destination_labels,
            min_count=min_count,
            relative_count=relative_count,
        )
        for k in label_coords.keys()
    }
    return _maximise_matching_overlaps(overlap_candidates)


def _assign_cells_to_matches(tracks: pd.DataFrame, matches: dict[int, int]) -> None:
    prior_cells = tracks.loc[matches.keys(), "cell"]
    wh_unassigned = prior_cells == 0
    prior_cells[wh_unassigned] = (
        np.arange((wh_unassigned).sum()) + tracks.cell.max() + 1
    )
    tracks.loc[matches.keys(), "cell"] = prior_cells.values
    tracks.loc[matches.values(), "cell"] = prior_cells.values


def _calc_distances_pbcs(
    start_coords: np.ndarray,
    end_coords: np.ndarray,
    domain_size: tuple[int],
    PBC_flag: Optional[Literal["none", "hdim_1", "hdim_2", "both"]] = None,
) -> np.ndarray[float]:
    domain_size = np.array(domain_size) + 1
    pos_neg_offset = np.where(start_coords < end_coords, 1, -1)
    if len(domain_size) == 3:
        domain_size[0] = 0
    if PBC_flag in [None, "none", "hdim1"]:
        domain_size[-1] = 0
    if PBC_flag in [None, "none", "hdim2"]:
        domain_size[-2] = 0

    return pos_neg_offset * np.minimum(
        np.abs(end_coords - start_coords),
        np.abs(end_coords - pos_neg_offset * domain_size - start_coords),
    )


def _assign_velocities(
    tracks: pd.DataFrame,
    matches: dict[int, int],
    domain_size: tuple[int],
    PBC_flag: Optional[Literal["none", "hdim_1", "hdim_2", "both"]] = None,
    prior: bool = False,
) -> None:
    end_locations = tracks.loc[matches.values(), ["hdim_1", "hdim_2"]].to_numpy()
    start_locations = tracks.loc[matches.keys(), ["hdim_1", "hdim_2"]].to_numpy()
    velocities = (
        _calc_distances_pbcs(
            start_locations, end_locations, domain_size, PBC_flag=PBC_flag
        )
        / (
            tracks.loc[matches.values(), ["time"]]
            - tracks.loc[matches.keys(), ["time"]].values
        )
        .time.dt.total_seconds()
        .to_numpy()[:, None]
    )
    if prior:
        tracks.loc[matches.keys(), "_track_velocity"] = dict(
            zip(matches.keys(), velocities)
        )
    else:
        tracks.loc[matches.values(), "_track_velocity"] = dict(
            zip(matches.values(), velocities)
        )


def _bootstrap_velocities(
    tracks: pd.DataFrame,
    mask: xr.DataArray,
    min_count: int = 1,
    relative_count: float = 0,
    PBC_flag: Optional[Literal["none", "hdim_1", "hdim_2", "both"]] = None,
) -> None:
    [_, _, origin_labels, _], [_, _, destination_labels, _] = next(
        _get_paired_field_and_features_iterator(mask, tracks)
    )
    matched_overlaps = _find_overlaps(
        tracks,
        origin_labels,
        destination_labels,
        min_count=min_count,
        relative_count=relative_count,
        translate_method=None,
    )
    _assign_velocities(
        tracks, matched_overlaps, origin_labels.shape, PBC_flag=PBC_flag, prior=True
    )


def _filter_stub_cells(
    tracks: pd.DataFrame,
    stubs: int,
    cell_number_start: int,
    cell_number_unassigned: int,
) -> pd.DataFrame:
    cell_count = tracks.groupby("cell").cell.count()
    stub_cells = cell_count.index[cell_count < stubs].values
    tracks.loc[np.isin(tracks.cell, stub_cells), "cell"] = 0

    new_cells = np.unique(tracks.cell, return_inverse=True)[1]
    new_cells[new_cells > 0] += cell_number_start - 1
    new_cells = np.where(
        new_cells > 0, new_cells + cell_number_start - 1, cell_number_unassigned
    )

    tracks["cell"] = new_cells

    return tracks


def _assign_cell_times(
    tracks: pd.DataFrame, cell_number_unassigned: int
) -> pd.DataFrame:
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
    translate_method: None | Literal["constant", "drift", "predict"] = None,
    velocity_method: None | Literal["constant", "mean", "nearest"] = None,
    velocity_constant: None | float | np.ndarray = None,
    PBC_flag: Optional[Literal["none", "hdim_1", "hdim_2", "both"]] = None,
) -> pd.DataFrame:
    tracks = features.copy().set_index("feature")
    tracks["cell"] = 0

    hdim1_size = mask.shape[-2]
    hdim2_size = mask.shape[-1]
    vdim_size = mask.shape[-3] if "vdim" in features else None

    if translate_method in ["drift", "predict"]:
        _bootstrap_velocities(tracks, mask)

    paired_iterator = _get_paired_field_and_features_iterator(mask, tracks)

    for [_, origin_timestep, origin_labels, origin_features], [
        _,
        destination_timestep,
        destination_labels,
        _,
    ] in paired_iterator:
        delta_t = (
            to_timestamp(destination_timestep.values)
            - to_timestamp(origin_timestep.values)
        ).total_seconds()
        matches = _find_overlaps(
            origin_features,
            origin_labels,
            destination_labels,
            min_count=minimum_overlap,
            relative_count=minimum_relative_overlap,
            translate_method=translate_method,
            velocity_method=velocity_method,
            velocity_constant=velocity_constant,
            delta_t=delta_t,
            hdim1_size=hdim1_size,
            hdim2_size=hdim2_size,
            vdim_size=vdim_size,
        )
        _assign_cells_to_matches(tracks, matches)
        if translate_method in ["drift", "predict"]:
            _assign_velocities(tracks, matches, origin_labels.shape, pbc_flag=PBC_flag)

    if "_track_velocity" in tracks.columns:
        tracks = tracks.drop("_track_velocity", axis=1)

    tracks = _filter_stub_cells(
        tracks,
        stubs=stubs,
        cell_number_start=cell_number_start,
        cell_number_unassigned=cell_number_unassigned,
    )

    tracks = _assign_cell_times(tracks, cell_number_unassigned=cell_number_unassigned)

    # Reset index to match features input and replace features column in the correct location
    tracks = tracks.set_index(features.index)
    tracks.insert(features.columns.get_loc("feature"), "feature", features.feature)

    return tracks
