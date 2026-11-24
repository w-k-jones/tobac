"""Provide overlap tracking methods"""

from functools import partial
from typing import Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd
import xarray as xr
import skimage.measure
from scipy.sparse import coo_array
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from sklearn.neighbors import BallTree

import tobac.utils.internal as internal_utils
from tobac.utils.datetime import to_timestamp
from tobac.utils.generators import field_and_features_over_time
from tobac.utils.periodic_boundaries import build_distance_function

from tobac.feature_detection import feature_position


def _unique_nonzero(arr: np.ndarray, **kwargs) -> np.ndarray:
    """Return unique nonzero elements of an array.

    Equivalent to np.unique on nonzero elements. Output is always flattened,
    and inverse indices are positions within the nonzero array, not the original.

    Parameters
    ----------
    arr : np.ndarray
        Array to find unique nonzero values in.
    **kwargs
        Additional keyword arguments passed to np.unique.

    Returns
    -------
    np.ndarray
        Unique nonzero values in the input array.

    """
    return np.unique(arr[arr != 0], **kwargs)


def _find_overlaps_for_label(
    coords: np.ndarray[int],
    counts: int,
    destination_labels: np.ndarray[int],
    destination_counts: dict[int, int],
    min_count: int = 1,
    relative_count: float = 0,
) -> tuple[np.ndarray[int], np.ndarray[int]]:
    """Find unique, nonzero label overlaps for a given set of array indices.

    Parameters
    ----------
    coords : np.ndarray[int]
        Coordinate array of region locations. Must be 2xN or 3xN shaped
        for 2D or 3D data respectively.
    counts : int
        Number of pixels in the search region.
    destination_labels : np.ndarray[int]
        Array of label values to search for overlaps within.
    destination_counts : dict[int, int]
        Number of pixels in each label within destination_labels.
    min_count : int, optional
        Minimum number of pixels for a label to be returned as an overlap.
        Default is 1.
    relative_count : float, optional
        Minimum fraction of search region for a label to cover to be returned
        as an overlap. Default is 0.

    Returns
    -------
    matched_labels : np.ndarray[int]
        Array of labels matching the overlap criteria.
    matched_counts : np.ndarray[int]
        Array of pixel counts for each matched label.

    """
    matched_labels, matched_counts = _unique_nonzero(
        destination_labels.values[*coords], return_counts=True
    )

    min_counts = np.minimum(counts, [destination_counts[k] for k in matched_labels])

    wh = np.logical_and(
        matched_counts >= min_count, matched_counts / min_counts >= relative_count
    )

    return matched_labels[wh], matched_counts[wh]


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


class FeatureBallTree(BallTree):
    """A child class of scikit-learn's BallTree that handles feature dataframe input
    and periodic boundary conditions (PBC).
    """

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
        """Initialize a FeatureBallTree from a features dataframe.

        Parameters
        ----------
        features : pd.DataFrame
            Features dataframe with columns 'hdim_1', 'hdim_2', and optionally 'vdim'
            for 2D and 3D data respectively.
        PBC_flag : {'none', 'hdim_1', 'hdim_2', 'both'}, optional
            Specification of which dimensions have periodic boundary conditions.
            Default is None (no periodic boundaries).
        min_h1 : int, optional
            Minimum value of first horizontal dimension for PBC. Default is 0.
        max_h1 : int, optional
            Maximum value of first horizontal dimension for PBC. Default is 0.
        min_h2 : int, optional
            Minimum value of second horizontal dimension for PBC. Default is 0.
        max_h2 : int, optional
            Maximum value of second horizontal dimension for PBC. Default is 0.
        **kwargs
            Additional keyword arguments passed to BallTree constructor.

        """
        self.is_3D = "vdim" in features.columns
        self.index = features.index.values
        if PBC_flag in ["hdim_1", "hdim_2", "both"]:
            kwargs["metric"] = "pyfunc"
            kwargs["func"] = build_distance_function(
                min_h1, max_h1, min_h2, max_h2, PBC_flag, self.is_3D
            )
        super().__init__(self._get_feature_locations(features), **kwargs)

    def _get_feature_locations(self, features: pd.DataFrame) -> np.ndarray:
        """Extract coordinate array from features dataframe.

        Parameters
        ----------
        features : pd.DataFrame
            Features dataframe with position columns.

        Returns
        -------
        np.ndarray
            Coordinate array of shape (n_features, n_dims) containing either
            ['hdim_1', 'hdim_2'] for 2D or ['vdim', 'hdim_1', 'hdim_2'] for 3D data.

        Raises
        ------
        AssertionError
            If dimensionality of query features does not match the tree dimensionality.

        """
        assert (
            "vdim" in features.columns
        ) == self.is_3D, "Query features must match dimensionality of original features"
        return (
            features[["vdim", "hdim_1", "hdim_2"]].to_numpy()
            if self.is_3D
            else features[["hdim_1", "hdim_2"]].to_numpy()
        )

    def query(self, features: pd.DataFrame, *args, **kwargs) -> np.ndarray:
        """Query the tree for nearest neighbors.

        Wraps the parent BallTree.query method to handle feature dataframe input
        and return results mapped to original feature indices.

        Parameters
        ----------
        features : pd.DataFrame
            Query features dataframe with position columns matching tree dimensionality.
        *args
            Positional arguments passed to BallTree.query.
        **kwargs
            Keyword arguments passed to BallTree.query.

        Returns
        -------
        indices : np.ndarray
            Feature indices of nearest neighbors, shape (n_queries,) or
            (n_queries, k) depending on input.
        distances : np.ndarray, optional
            Distances to nearest neighbors if return_distance=True in kwargs.

        """
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
        """Query the tree for neighbors within a specified radius.

        Wraps the parent BallTree.query_radius method to handle feature dataframe
        input and return results mapped to original feature indices.

        Parameters
        ----------
        features : pd.DataFrame
            Query features dataframe with position columns matching tree dimensionality.
        *args
            Positional arguments passed to BallTree.query_radius.
        **kwargs
            Keyword arguments passed to BallTree.query_radius.

        Returns
        -------
        neighbors : np.ndarray
            Array of feature indices of neighbors within radius for each query point.
            Shape is (n_queries,) with dtype=object containing variable-length arrays.
        distances : np.ndarray, optional
            Distances to neighbors if return_distance=True in kwargs.

        """
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
    PBC_flag: Optional[Literal["none", "hdim_1", "hdim_2", "both"]] = None,
) -> None:
    """Update predicted velocities for features with missing velocity values.

    Parameters
    ----------
    features : pd.DataFrame
        Features dataframe with '_track_velocity' column to update.
    velocity_method : {None, 'constant', 'mean', 'nearest'}, optional
        Method for filling missing velocities. Default is 'constant'.
        - 'constant': Use velocity_constant value
        - 'mean': Use mean of existing velocities
        - 'nearest': Use nearest neighbor velocity (accounting for PBCs)
    velocity_constant : None, float, or np.ndarray, optional
        Constant velocity value or array of values. Default is 0.
    PBC_flag : {'none', 'hdim_1', 'hdim_2', 'both'}, optional
        Specification of which dimensions have periodic boundaries.
        Default is None.

    Returns
    -------
    None
        Modifies features DataFrame in place.

    """
    wh_missing_vels = features._track_velocity.isna()
    if wh_missing_vels.any():
        if velocity_method == "constant":
            features.loc[wh_missing_vels, "_track_velocity"] = (
                {k: velocity_constant for k in features[wh_missing_vels].index}
                if hasattr(velocity_constant, "__iter__")
                else velocity_constant
            )
        elif wh_missing_vels.all():
            features.loc[wh_missing_vels, "_track_velocity"] = 0
        elif velocity_method == "mean":
            mean_vel = (
                features._track_velocity[~features._track_velocity.isna()]
                .to_numpy()
                .mean()
            )
            features.loc[wh_missing_vels, "_track_velocity"] = {
                k: mean_vel for k in features[wh_missing_vels].index
            }
        elif velocity_method == "nearest":
            # create BallTree to find nearest velocity accounting for PBCs
            btree = FeatureBallTree(features[~wh_missing_vels], PBC_flag=PBC_flag)
            features.loc[wh_missing_vels, "_track_velocity"] = features._track_velocity[
                ~wh_missing_vels
            ][
                btree.query(features[wh_missing_vels], return_distance=False).ravel()
            ].values
        else:
            features.loc[wh_missing_vels, "_track_velocity"] = 0


def _wrap_coords(
    coords: np.ndarray[int],
    shape: np.ndarray[int],
    PBC_flag: Optional[Literal["none", "hdim_1", "hdim_2", "both"]] = None,
) -> np.ndarray[int]:
    """Wrap coordinates according to periodic boundary conditions. Coordinate locations which fall outside the size of the array are removed.

    Parameters
    ----------
    coords : np.ndarray[int]
        Coordinate array to wrap. Shape is (n_dims, n_points).
    shape : np.ndarray[int]
        Shape of the domain to wrap coordinates in. Translated points
        outside of the domain will be removed if they are not on a periodic boundary
    PBC_flag : {'none', 'hdim_1', 'hdim_2', 'both'}, optional
        Specification of which dimensions have periodic boundaries.
        Default is None.

    Returns
    -------
    np.ndarray[int]
        Wrapped coordinate array with invalid points filtered out.

    """
    if PBC_flag in ["hdim_1", "both"]:
        coords[-2] = coords[-2] % shape[-2]
    if PBC_flag in ["hdim_2", "both"]:
        coords[-1] = coords[-1] % shape[-1]
    return coords[
        :,
        np.logical_and(
            np.all(coords >= 0, axis=0), np.all(coords < shape[:, np.newaxis], axis=0)
        ),
    ]


def _translate_labels(
    features: pd.DataFrame,
    timestep: pd.Timestamp,
    translate_method: Optional[Literal["constant", "drift", "predict"]] = None,
    velocity_method: Optional[Literal["constant", "mean", "nearest"]] = None,
    velocity_constant: Optional[Union[float, np.ndarray]] = None,
    PBC_flag: Optional[Literal["none", "hdim_1", "hdim_2", "both"]] = None,
    hdim1_size: Optional[int] = None,
    hdim2_size: Optional[int] = None,
    vdim_size: Optional[int] = None,
) -> dict[int, np.ndarray[int]]:
    """Translate feature coordinates to a future timestep.

    Parameters
    ----------
    features : pd.DataFrame
        Features dataframe containing position and velocity information.
    timestep : pd.Timestamp
        Timestamp for the target timestep.
    translate_method : {'constant', 'drift', 'predict'}, optional
        Translation strategy to apply to coordinates. Defaults to None, which
        leaves the coordinates unchanged.
    velocity_method : {'constant', 'mean', 'nearest'}, optional
        Velocity estimation method used when ``translate_method='predict'``.
    velocity_constant : float or np.ndarray, optional
        Constant velocity used when ``translate_method='constant'``.
    PBC_flag : {'none', 'hdim_1', 'hdim_2', 'both'}, optional
        Periodic boundary condition specification.
    hdim1_size : int, optional
        Size of the first horizontal dimension.
    hdim2_size : int, optional
        Size of the second horizontal dimension.
    vdim_size : int, optional
        Size of the vertical dimension for 3D data.

    Returns
    -------
    dict[int, np.ndarray[int]]
        Mapping of feature indices to translated coordinate arrays. The arrays are
        stored in the ``_translated_coords`` column of ``features`` in-place.

    """

    wrap_coords = partial(
        _wrap_coords,
        shape=np.array(
            [s for s in [vdim_size, hdim1_size, hdim2_size] if s is not None], dtype=int
        ),
        PBC_flag=PBC_flag,
    )
    if translate_method == "predict":
        _update_predicted_velocities(
            features,
            velocity_method=velocity_method,
            velocity_constant=velocity_constant,
            PBC_flag=PBC_flag,
        )
        features["_translated_coords"] = (
            features._coords
            + features._track_velocity
            * (
                to_timestamp(timestep.values) - to_timestamp(features.time)
            ).dt.total_seconds()
        ).apply(lambda a: wrap_coords(a.round().astype(int).T))
    elif translate_method == "drift":
        if (~features._track_velocity.isna()).any():
            features["_translated_coords"] = (
                features._coords
                + pd.Series(
                    dict(
                        zip(
                            features.index,
                            features._track_velocity[~features._track_velocity.isna()]
                            .to_numpy()
                            .mean()
                            * (
                                to_timestamp(timestep.values)
                                - to_timestamp(features.time)
                            )
                            .dt.total_seconds()
                            .to_numpy()[:, np.newaxis],
                        )
                    )
                )
            ).apply(lambda a: wrap_coords(a.round().astype(int).T))
        else:  # If no valid velocities
            features["_translated_coords"] = features._coords.apply(
                lambda a: a.round().astype(int).T
            )
    elif translate_method == "constant":
        features["_translated_coords"] = (
            features._coords
            + pd.Series(
                dict(
                    zip(
                        features.index,
                        velocity_constant
                        * (to_timestamp(timestep.values) - to_timestamp(features.time))
                        .dt.total_seconds()
                        .to_numpy()[:, np.newaxis],
                    )
                )
            )
        ).apply(lambda a: wrap_coords(a.round().astype(int).T))
    else:
        features["_translated_coords"] = features._coords.apply(
            lambda a: a.round().astype(int).T
        )


def _find_overlaps(
    origin_features: pd.DataFrame,
    destination_features: pd.DataFrame,
    destination_labels: xr.DataArray,
    timestep: pd.Timestamp,
    min_count: int = 1,
    relative_count: float = 0,
    translate_method: Optional[Literal["constant", "drift", "predict"]] = None,
    velocity_method: Optional[Literal["constant", "mean", "nearest"]] = None,
    velocity_constant: Optional[Union[float, np.ndarray]] = None,
    PBC_flag: Optional[Literal["none", "hdim_1", "hdim_2", "both"]] = None,
) -> dict[int, int]:
    """Find overlap matches between origin and destination features.

    Parameters
    ----------
    origin_features : pd.DataFrame
        Features in the current timestep to be linked forward.
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
    translate_method : {'constant', 'drift', 'predict'}, optional
        Translation method used to predict the next location of origin features.
    velocity_method : {'constant', 'mean', 'nearest'}, optional
        Velocity estimation method for ``translate_method='predict'``.
    velocity_constant : float or np.ndarray, optional
        Constant velocity used when ``translate_method='constant'``.
    PBC_flag : {'none', 'hdim_1', 'hdim_2', 'both'}, optional
        Periodic boundary condition specification.

    Returns
    -------
    dict[int, int]
        Mapping of origin feature IDs to destination feature IDs for valid
        overlapping matches.

    """

    _translate_labels(
        origin_features,
        timestep,
        translate_method=translate_method,
        velocity_method=velocity_method,
        velocity_constant=velocity_constant,
        PBC_flag=PBC_flag,
        hdim1_size=destination_labels.shape[-2],
        hdim2_size=destination_labels.shape[-1],
        vdim_size=(
            destination_labels.shape[0] if len(destination_labels.shape) == 3 else None
        ),
    )
    overlap_candidates = origin_features.apply(
        lambda row: _find_overlaps_for_label(
            row._translated_coords,
            row._count,
            destination_labels,
            destination_features._count,
            min_count=min_count,
            relative_count=relative_count,
        ),
        axis=1,
    )
    return _maximise_matching_overlaps(overlap_candidates)


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


def _calc_distances_pbcs(
    start_coords: np.ndarray,
    end_coords: np.ndarray,
    domain_size: tuple[int],
    PBC_flag: Optional[Literal["none", "hdim_1", "hdim_2", "both"]] = None,
) -> np.ndarray[float]:
    """Calculate distances between coordinate pairs accounting for PBCs.

    Parameters
    ----------
    start_coords : np.ndarray
        Starting coordinates. Shape is (n_points, n_dims).
    end_coords : np.ndarray
        Ending coordinates. Shape is (n_points, n_dims).
    domain_size : tuple[int]
        Size of domain in each dimension.
    PBC_flag : {'none', 'hdim_1', 'hdim_2', 'both'}, optional
        Periodic boundary condition specification.

    Returns
    -------
    np.ndarray[float]
        Distance vectors accounting for periodic boundaries.
        Shape is (n_points, n_dims).

    #"""
    if PBC_flag in [None, "none"]:
        return end_coords - start_coords
    domain_size = np.array(domain_size) + 1
    pos_neg_offset = np.where(start_coords < end_coords, 1, -1)
    if len(domain_size) == 3:
        domain_size[0] = 0
    if PBC_flag == "hdim_1":
        domain_size[-1] = 0
    if PBC_flag == "hdim_2":
        domain_size[-2] = 0

    non_pbc_dist = np.abs(end_coords - start_coords)
    pbc_dist = np.abs(end_coords - pos_neg_offset * domain_size - start_coords)

    return pos_neg_offset * np.where(non_pbc_dist <= pbc_dist, non_pbc_dist, -pbc_dist)


def _assign_velocities(
    origin_features: pd.DataFrame,
    destination_features: pd.DataFrame,
    matches: dict[int, int],
    domain_size: tuple[int],
    PBC_flag: Optional[Literal["none", "hdim_1", "hdim_2", "both"]] = None,
    prior: bool = False,
) -> None:
    if len(matches):
        velocities = (
            _calc_distances_pbcs(
                np.stack(origin_features.loc[matches.keys(), "_centroid"]),
                np.stack(destination_features.loc[matches.values(), "_centroid"]),
                domain_size=domain_size,
                PBC_flag=PBC_flag,
            )
            / np.stack(
                (
                    destination_features.loc[matches.values(), "time"]
                    - origin_features.loc[matches.keys(), "time"].values
                ).dt.total_seconds()
            )[:, np.newaxis]
        )
        if prior:
            origin_features.loc[matches.keys(), "_track_velocity"] = dict(
                zip(matches.keys(), velocities)
            )
        else:
            destination_features.loc[matches.values(), "_track_velocity"] = dict(
                zip(matches.values(), velocities)
            )
    else:
        if prior:
            origin_features["_track_velocity"] = None
        else:
            destination_features["_track_velocity"] = None


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
    matched_overlaps = _find_overlaps(
        origin_features,
        destination_features,
        destination_labels,
        timestep,
        min_count=min_count,
        relative_count=relative_count,
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
    save_unfiltered_cell: bool = False,
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
    save_unfiltered_cell : bool, optional
        If True, save the cell numbers from before stub/time filtering in a
        'cell_unfiltered' column (placed directly after 'cell'). Every feature
        gets an unfiltered cell number, including features that were never
        matched (cell == 0), each of which is assigned its own unique number.
        Default is False.

    Returns
    -------
    pd.DataFrame
        Tracks dataframe with stub cells removed and cell IDs renumbered.

    """
    # Ensure all cell values are zero or greater (they should be already)
    tracks["cell"] = np.maximum(tracks.cell.values, 0)
    if save_unfiltered_cell:
        # Renumber the cells from before any stub filtering, giving features
        # that were never matched (cell == 0) their own unique cell numbers
        # first, so that every feature has an unfiltered cell identity.
        unfiltered_cell = tracks.cell.copy()
        wh_unmatched = unfiltered_cell == 0
        if wh_unmatched.any():
            unfiltered_cell.loc[wh_unmatched] = (
                np.arange(int(wh_unmatched.sum()), dtype=int)
                + unfiltered_cell.max()
                + 1
            )
        tracks.insert(
            tracks.columns.get_loc("cell") + 1,
            "cell_unfiltered",
            np.unique(unfiltered_cell, return_inverse=True)[1] + cell_number_start,
        )
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


def _get_coords_and_centroids(
    features: pd.DataFrame,
    mask: xr.DataArray,
    PBC_flag: Optional[Literal["none", "hdim_1", "hdim_2", "both"]] = None,
) -> pd.DataFrame:
    """Attach region counts and centroids to feature entries.

    Parameters
    ----------
    features : pd.DataFrame
        Features dataframe containing feature metadata.
    mask : xr.DataArray
        Labeled mask for the current timestep.
    PBC_flag : {'none', 'hdim_1', 'hdim_2', 'both'}, optional
        Periodic boundary condition specification.

    Returns
    -------
    pd.DataFrame
        Input features dataframe joined with ``_count``, ``_coords``, and
        ``_centroid`` columns derived from the mask.

    """
    if len(features):
        if len(mask.shape) == 3:  # is_3D
            feature_position_partial = partial(
                feature_position,
                hdim1_max=mask.shape[1],
                hdim2_max=mask.shape[2],
                PBC_flag=PBC_flag,
            )
            fp_lambda = lambda row: np.array(
                feature_position_partial(
                    row._coords[:, 1], row._coords[:, 2], vdim_indices=row._coords[:, 0]
                )
            )
        else:
            feature_position_partial = partial(
                feature_position,
                hdim1_max=mask.shape[0],
                hdim2_max=mask.shape[1],
                PBC_flag=PBC_flag,
            )
            fp_lambda = lambda row: np.array(feature_position_partial(*row._coords.T))

        props_df = (
            pd.DataFrame(
                skimage.measure.regionprops_table(
                    mask.values, properties=("label", "area", "coords")
                )
            )
            .rename(columns=dict(label="feature", area="_count", coords="_coords"))
            .set_index("feature")
        )

        props_df["_centroid"] = props_df.apply(fp_lambda, axis=1)
    else:
        props_df = pd.DataFrame(columns=["_count", "_coords", "_centroid"])

    return features.join(props_df)


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
    save_unfiltered_cell: bool = False,
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
    save_unfiltered_cell : bool, optional
        If True, additionally save the cell numbers from before stub filtering
        (see `stubs` and `time_cell_min`) in a 'cell_unfiltered' column. Every
        feature gets an unfiltered cell number, so cells that are filtered out
        as stubs keep their identity. This is required to append new features
        to the tracks with :func:`append_tracks_overlap` when `stubs` > 1 or
        `time_cell_min` is set. Default is False.

    Returns
    -------
    pd.DataFrame
        Features dataframe with added columns:
        - 'cell': Cell identifier for tracked features
        - 'time_cell': Time relative to start of cell
        - 'cell_unfiltered': (only if `save_unfiltered_cell` is True) cell
          numbers before stub filtering
        Features not linked to any cell are assigned cell_number_unassigned.

    """
    tracks = features.copy().set_index("feature")
    tracks["cell"] = 0

    if isinstance(max_time_gap, (int, float)):
        max_time_gap = pd.to_timedelta(max_time_gap, unit="s")
    if isinstance(time_cell_min, (int, float)):
        time_cell_min = pd.to_timedelta(time_cell_min, unit="s")

    mask = _prepare_mask_time_first(
        mask, vertical_axis=vertical_axis, vertical_coord=vertical_coord
    )

    field_and_features = iter(field_and_features_over_time(mask, tracks))

    _, _, labels, features_t = next(field_and_features)
    origin_features = _get_coords_and_centroids(features_t, labels, PBC_flag=PBC_flag)

    bootstrap = True if translate_method in ["drift", "predict"] else False
    tracks = _run_overlap_linking(
        field_and_features,
        tracks,
        origin_features,
        bootstrap,
        memory=memory,
        max_time_gap=max_time_gap,
        minimum_overlap=minimum_overlap,
        minimum_relative_overlap=minimum_relative_overlap,
        translate_method=translate_method,
        velocity_method=velocity_method,
        velocity_constant=velocity_constant,
        PBC_flag=PBC_flag,
    )

    tracks = _filter_stub_cells(
        tracks,
        stubs=stubs,
        cell_number_start=cell_number_start,
        cell_number_unassigned=cell_number_unassigned,
        time_cell_min=time_cell_min,
        save_unfiltered_cell=save_unfiltered_cell,
    )

    tracks = _assign_cell_times(tracks, cell_number_unassigned=cell_number_unassigned)

    # Reset index to match features input and replace features column in the correct location
    tracks = tracks.set_index(features.index)
    tracks.insert(features.columns.get_loc("feature"), "feature", features.feature)

    return tracks


def _prepare_mask_time_first(
    mask: xr.DataArray,
    vertical_axis: Optional[int] = None,
    vertical_coord: Optional[str] = None,
) -> xr.DataArray:
    """Transpose a 4D mask to (time, vdim, hdim_1, hdim_2) dimension order.

    3D masks (time, hdim_1, hdim_2) are returned unchanged.

    Parameters
    ----------
    mask : xr.DataArray
        Labeled mask with a 'time' coordinate; 3D (time, hdim_1, hdim_2) or
        4D (with a vertical dimension).
    vertical_axis : int, optional
        The vertical axis number of the data. If None, uses vertical_coord
        to determine axis. This must be >=0.
    vertical_coord : str, optional
        Name of the vertical coordinate. If None, tries to auto-detect.

    Returns
    -------
    xr.DataArray
        Mask with time as the first dimension, followed by the vertical
        dimension (for 4D masks), hdim_1, and hdim_2.

    """
    time_axis = internal_utils.find_axis_from_coord(mask, "time")
    if len(mask.shape) == 4:
        if vertical_axis is None:
            # We need to determine vertical axis.
            # first, find the name of the vertical axis
            vertical_axis_name = internal_utils.find_vertical_coord_name(
                mask, vertical_coord=vertical_coord
            )
            # then find our axis number.
            vertical_axis = internal_utils.find_axis_from_coord(
                mask, vertical_axis_name
            )

            if vertical_axis is None:
                raise ValueError("Cannot find vertical coordinate.")

        if vertical_axis < 0:
            raise ValueError("vertical_axis must be >=0.")

        hdim_1_axis, hdim_2_axis = internal_utils.find_hdim_axes_3D(
            mask, vertical_axis=vertical_axis
        )

        mask = mask.transpose(
            *[
                mask.dims[i]
                for i in [time_axis, vertical_axis, hdim_1_axis, hdim_2_axis]
            ]
        )
    return mask


def _run_overlap_linking(
    field_and_features,
    tracks: pd.DataFrame,
    origin_features: pd.DataFrame,
    bootstrap: bool,
    memory: int = 0,
    max_time_gap: Optional[pd.Timedelta] = None,
    minimum_overlap: int = 1,
    minimum_relative_overlap: float = 0,
    translate_method: Optional[Literal["constant", "drift", "predict"]] = None,
    velocity_method: Optional[Literal["constant", "mean", "nearest"]] = None,
    velocity_constant: Optional[Union[float, np.ndarray]] = None,
    PBC_flag: Optional[Literal["none", "hdim_1", "hdim_2", "both"]] = None,
) -> pd.DataFrame:
    """Run the overlap linking loop over a sequence of timesteps.

    Matches the features of each yielded timestep to the `origin_features`
    (features of the previous timestep, plus features kept via `memory`) and
    assigns cell numbers in `tracks` in-place.

    Parameters
    ----------
    field_and_features : iterator
        Iterator yielding (frame, timestep, labels, features_t) tuples, as
        produced by :func:`tobac.utils.generators.field_and_features_over_time`.
    tracks : pd.DataFrame
        Tracks dataframe indexed by 'feature' with a 'cell' column, modified
        in-place.
    origin_features : pd.DataFrame
        Features of the timestep preceding the first yielded timestep, with
        geometry columns ('_count', '_coords', '_centroid') attached, used as
        the initial linking candidates.
    bootstrap : bool
        Whether to estimate velocities from the first pair of timesteps with
        both origin and destination features before matching.
    memory : int, optional
        Number of output timesteps features allowed to vanish for to still be
        considered tracked. Default is 0.
    max_time_gap : pd.Timedelta, optional
        Maximum allowed time gap between an origin feature and the timestep it
        is matched to. Default is None (no limit).
    minimum_overlap : int, optional
        Minimum pixel overlap required for feature linking. Default is 1.
    minimum_relative_overlap : float, optional
        Minimum fraction of the smaller of the two features required to overlap
        for linking. Default is 0.
    translate_method : {'constant', 'drift', 'predict'}, optional
        Method for predicting feature positions at the next timestep.
    velocity_method : {'constant', 'mean', 'nearest'}, optional
        Method for velocity estimation if translate_method is used.
    velocity_constant : float or np.ndarray, optional
        Constant velocity for translation. Used if translate_method is
        'constant'.
    PBC_flag : {'none', 'hdim_1', 'hdim_2', 'both'}, optional
        Periodic boundary condition specification.

    Returns
    -------
    pd.DataFrame
        The `tracks` dataframe with updated 'cell' column.

    """
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

    return tracks


def _recompute_track_velocities(
    tracks: pd.DataFrame,
    domain_size: tuple[int],
    PBC_flag: Optional[Literal["none", "hdim_1", "hdim_2", "both"]] = None,
    centroids: Optional[dict] = None,
) -> None:
    """Recompute per-feature track velocities from the tracked positions.

    For every cell, the velocity of each feature is the displacement from the
    previous feature of the same cell (accounting for PBCs) divided by the
    time difference, reproducing the velocities assigned during linking.
    Features that are the first (or only) feature of their cell get no
    velocity (None).

    Parameters
    ----------
    tracks : pd.DataFrame
        Tracks dataframe indexed by 'feature' with 'cell', 'time' and position
        columns. A '_track_velocity' column is added in-place.
    domain_size : tuple[int]
        Size of the domain in each (spatial) dimension, ordered as the
        position columns: (hdim_1, hdim_2) for 2D or (vdim, hdim_1, hdim_2)
        for 3D data.
    PBC_flag : {'none', 'hdim_1', 'hdim_2', 'both'}, optional
        Periodic boundary condition specification.
    centroids : dict, optional
        Mapping of feature index to mask-derived centroid array (ordered as
        the position columns). Where available, centroids are used instead of
        the position columns, matching the velocities assigned during linking
        (which are based on the mask-derived centroids).

    Returns
    -------
    None
        Modifies tracks in-place.

    """
    pos_cols = (
        ["vdim", "hdim_1", "hdim_2"]
        if "vdim" in tracks.columns
        else ["hdim_1", "hdim_2"]
    )
    tracks["_track_velocity"] = None
    if not len(tracks):
        return
    positions = np.array(tracks[pos_cols].to_numpy(dtype=float), copy=True)
    if centroids:
        for fid, centroid in centroids.items():
            if fid in tracks.index:
                positions[tracks.index.get_loc(fid)] = np.asarray(centroid, dtype=float)
    for cell, group in tracks.groupby("cell"):
        if cell == 0 or len(group) < 2:
            continue
        group = group.sort_values("time")
        group_pos = positions[[tracks.index.get_loc(i) for i in group.index]]
        dt = group.time.diff().dt.total_seconds().to_numpy()
        velocities = (
            _calc_distances_pbcs(
                group_pos[:-1],
                group_pos[1:],
                domain_size=domain_size,
                PBC_flag=PBC_flag,
            )
            / dt[1:, np.newaxis]
        )
        tracks.loc[group.index[1:], "_track_velocity"] = dict(
            zip(group.index[1:], velocities)
        )


def _combine_features_and_masks(
    features: Union[pd.DataFrame, Sequence[pd.DataFrame]],
    mask: Union[xr.DataArray, Sequence[xr.DataArray]],
) -> tuple[pd.DataFrame, xr.DataArray]:
    """Combine one or more features dataframes and masks into single objects.

    Parameters
    ----------
    features : pd.DataFrame or sequence of pd.DataFrame
        Features dataframe(s) for the new time period(s).
    mask : xr.DataArray or sequence of xr.DataArray
        Labeled mask(s) matching the features dataframe(s). Sequences of
        masks are concatenated along their 'time' dimension.

    Returns
    -------
    features : pd.DataFrame
        Combined features dataframe with a fresh RangeIndex.
    mask : xr.DataArray
        Combined mask.

    """
    if isinstance(features, pd.DataFrame):
        features_list = [features]
    else:
        features_list = list(features)
    if isinstance(mask, xr.DataArray):
        mask_list = [mask]
    else:
        mask_list = list(mask)
    if len(features_list) != len(mask_list):
        raise ValueError(
            "features and mask must be single objects or sequences of the same "
            "length (one features dataframe and one mask per time period)."
        )
    if len(features_list) == 0:
        raise ValueError("At least one features dataframe and one mask are required.")
    combined_features = pd.concat(features_list, ignore_index=True)
    if "feature" in combined_features.columns:
        # chunks may overlap at their boundaries; keep the first occurrence
        combined_features = combined_features.drop_duplicates(
            subset="feature", keep="first"
        ).reset_index(drop=True)
    if len(mask_list) == 1:
        combined_mask = mask_list[0]
    else:
        combined_mask = xr.concat(mask_list, dim="time")
        # chunks may overlap at their boundaries; keep the first occurrence
        mask_time_index = combined_mask.coords["time"].to_index()
        combined_mask = combined_mask.isel(
            time=~mask_time_index.duplicated(keep="first")
        )
    return combined_features, combined_mask


def _combine_masks(
    mask: Union[xr.DataArray, Sequence[xr.DataArray]],
) -> xr.DataArray:
    """Combine one or more masks into a single mask along the time dimension.

    Parameters
    ----------
    mask : xr.DataArray or sequence of xr.DataArray
        Mask(s) with a 'time' coordinate.

    Returns
    -------
    xr.DataArray
        Combined mask with duplicated timesteps removed (first occurrence
        kept).

    """
    if isinstance(mask, xr.DataArray):
        mask_list = [mask]
    else:
        mask_list = list(mask)
    if len(mask_list) == 0:
        raise ValueError("At least one mask is required.")
    if len(mask_list) == 1:
        combined_mask = mask_list[0]
    else:
        combined_mask = xr.concat(mask_list, dim="time")
    mask_time_index = pd.DatetimeIndex(
        to_timestamp(combined_mask.coords["time"].values)
    )
    combined_mask = combined_mask.isel(time=~mask_time_index.duplicated(keep="first"))
    return combined_mask


def _datetime_index_union(*indexes: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return the sorted union of datetime indexes, without duplicates."""
    result = indexes[0]
    for idx in indexes[1:]:
        result = result.union(idx)
    return result[~result.duplicated()]


def append_tracks_overlap(
    tracks: pd.DataFrame,
    tracks_mask: Union[xr.DataArray, Sequence[xr.DataArray]],
    features: Union[pd.DataFrame, Sequence[pd.DataFrame]],
    mask: Union[xr.DataArray, Sequence[xr.DataArray]],
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
    """Continue overlap tracking of existing tracks with new features and masks.

    Takes the output of :func:`linking_overlap` for one time period and
    continues tracking for one or more sets of features dataframes and
    segmentation masks corresponding to future time periods, using the same
    logic as :func:`linking_overlap`. This enables incremental tracking
    without re-running the linking on the whole dataset.

    The `features` and `mask` arguments describe only the new (future) time
    periods and may each be a single object or a sequence of objects (one per
    period); sequences are combined before tracking. Rows of `features` at
    timesteps that are already present in `tracks` are allowed and ignored (the
    tracked rows are kept as they are); only timesteps that are not present in
    `tracks` are newly tracked.

    The labeled regions of the features at the end of the original period are
    needed to compute the overlaps (and the initial centroids and velocities)
    into the first new timestep. These are taken from the separate
    `tracks_mask` argument, which must include at least the last
    `max(stubs, memory)` timesteps present in `tracks`; earlier timesteps of
    the original period are not needed.

    Parameters
    ----------
    tracks : pd.DataFrame
        Output of :func:`linking_overlap` for an earlier time period. Must
        contain 'feature' and 'cell' columns. If it also contains a
        'cell_unfiltered' column (see the `save_unfiltered_cell` option of
        :func:`linking_overlap`), the unfiltered cell numbers are used to
        link the new features, so that cells filtered out as stubs in `tracks`
        keep their identity; the output then also contains 'cell_unfiltered'.
    tracks_mask : xr.DataArray or sequence of xr.DataArray
        Labeled mask(s) for the (end of the) original time period of `tracks`,
        used to recalculate the initial centroids and velocities of the
        boundary features. Must include at least the last
        `max(stubs, memory)` timesteps present in `tracks`. The labels at
        those timesteps must match the 'feature' values of `tracks`.
    features : pd.DataFrame or sequence of pd.DataFrame
        Features dataframe(s) for the future time period(s), with a 'feature'
        column and position columns ('hdim_1', 'hdim_2', and optionally
        'vdim'), as for :func:`linking_overlap`. May include rows at timesteps
        already present in `tracks`; those are ignored.
    mask : xr.DataArray or sequence of xr.DataArray
        Labeled mask(s) for the future time period(s), matching `features`.
        Must contain all timesteps of the new `features` that are not already
        in `tracks`.
    stubs : int, optional
        Minimum number of timesteps for a cell to be retained, applied to the
        combined tracks. Default is 1.
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
        Method for predicting feature positions at the next timestep. Default
        is None.
    velocity_method : {'constant', 'mean', 'nearest'}, optional
        Method for velocity estimation if translate_method is used.
    velocity_constant : float or np.ndarray, optional
        Constant velocity for translation. Used if translate_method is
        'constant'.
    PBC_flag : {'none', 'hdim_1', 'hdim_2', 'both'}, optional
        Periodic boundary condition specification.
    vertical_axis : int, optional
        The vertical axis number of the data. If None, uses vertical_coord
        to determine axis. This must be >=0.
    vertical_coord : str, optional
        Name of the vertical coordinate. If None, tries to auto-detect.
    memory : int, optional
        Number of output timesteps features allowed to vanish for to still be
        considered tracked. Default is 0. `tracks_mask` must include the last
        `memory` + 1 timesteps of `tracks` so that features kept via `memory`
        can be matched.
    max_time_gap : pd.Timedelta or float, optional
        Maximum allowed time gap between a feature and the timestep it is
        matched to. If a float is given, it is interpreted as seconds. Default
        is None (no limit).
    time_cell_min : pd.Timedelta or float, optional
        Minimum time span of a cell for the cell to be retained, applied to
        the combined tracks. If a float is given, it is interpreted as
        seconds. Default is None (no minimum time span).

    Returns
    -------
    pd.DataFrame
        The combined tracks (original `tracks` plus the newly tracked
        features) with added columns 'cell' and 'time_cell' (and
        'cell_unfiltered' if `tracks` contained it), in the same format as
        :func:`linking_overlap`.

    Raises
    ------
    ValueError
        If `tracks` has no 'cell' or 'feature' column, if `tracks_mask` does
        not include the last `max(stubs, memory)` timesteps of `tracks`, if the
        new features contain timesteps that are not in `mask`, or if `stubs` >
        1 (or `time_cell_min` is set) and `tracks` does not contain
        'cell_unfiltered'.

    """
    if "cell" not in tracks.columns:
        raise ValueError("Need to have existing tracks with a 'cell' column.")
    if "feature" not in tracks.columns:
        raise ValueError("Need to have existing tracks with a 'feature' column.")
    if "time" not in tracks.columns:
        raise ValueError("Need to have existing tracks with a 'time' column.")

    if isinstance(max_time_gap, (int, float)):
        max_time_gap = pd.to_timedelta(max_time_gap, unit="s")
    if isinstance(time_cell_min, (int, float)):
        time_cell_min = pd.to_timedelta(time_cell_min, unit="s")

    has_unfiltered_cell = "cell_unfiltered" in tracks.columns
    if (stubs > 1 or time_cell_min is not None) and not has_unfiltered_cell:
        raise ValueError(
            "Append tracks with stubs > 1 (or time_cell_min set) requires the "
            "unfiltered cell numbers, as cells filtered out in the original "
            "tracks would otherwise lose their identity at the boundary. "
            "Track the original features with save_unfiltered_cell=True."
        )

    new_features, mask = _combine_features_and_masks(features, mask)
    mask = _prepare_mask_time_first(
        mask, vertical_axis=vertical_axis, vertical_coord=vertical_coord
    )
    tracks_mask = _prepare_mask_time_first(
        _combine_masks(tracks_mask),
        vertical_axis=vertical_axis,
        vertical_coord=vertical_coord,
    )

    mask_time_index = pd.DatetimeIndex(to_timestamp(mask.coords["time"].values))
    mask_time_index = mask_time_index[~mask_time_index.duplicated()]
    tracks_mask_time_index = pd.DatetimeIndex(
        to_timestamp(tracks_mask.coords["time"].values)
    )
    tracks_mask_time_index = tracks_mask_time_index[
        ~tracks_mask_time_index.duplicated()
    ]
    tracks_mask_local_frame = pd.Series(
        np.arange(len(tracks_mask_time_index)), index=tracks_mask_time_index
    )

    # Combined time axis (original boundary timesteps + new timesteps) used for
    # frame numbering, so that the memory window arithmetic is consistent
    # across the boundary
    all_time_index = _datetime_index_union(mask_time_index, tracks_mask_time_index)
    time_to_frame = pd.Series(np.arange(len(all_time_index)), index=all_time_index)

    tracks_times = pd.DatetimeIndex(to_timestamp(tracks.time.unique()))
    if len(tracks_times) == 0:
        raise ValueError("tracks must contain at least one timestep.")
    last_track_time = tracks_times.max()

    # tracks_mask must cover at least the last max(stubs, memory) timesteps of
    # tracks, so that the boundary features' regions (and centroids/velocities)
    # can be recalculated
    n_window = max(stubs, memory, 1)
    required_times = tracks_times[-n_window:]
    missing_tracks_times = required_times.difference(tracks_mask_time_index)
    if len(missing_tracks_times) > 0:
        raise ValueError(
            "tracks_mask must include at least the last "
            f"max(stubs, memory) = {n_window} timesteps of tracks; missing: "
            f"{list(missing_tracks_times)}."
        )

    new_feature_times = pd.DatetimeIndex(to_timestamp(new_features.time.unique()))
    missing_times = new_feature_times.difference(mask_time_index)
    if len(missing_times) > 0:
        raise ValueError(
            "The new features contain timesteps that are not in the mask: "
            f"{list(missing_times)}."
        )

    # Combine the original tracks with the new features (rows at timesteps
    # that are already tracked are duplicates and are filtered out)
    new_features_only = new_features[~new_features.time.isin(tracks_times)]
    combined_features = pd.concat([tracks, new_features_only], ignore_index=True)

    tracks_work = combined_features.copy()
    # Use the unfiltered cell numbers as the working cell column if available,
    # so that stub-filtered cells keep their identity; otherwise map the
    # unassigned value to 0 (features matched in the new period then get fresh
    # cell numbers, as in linking_overlap).
    if has_unfiltered_cell:
        tracks_work["cell"] = tracks_work["cell_unfiltered"]
        tracks_work = tracks_work.drop(columns=["cell_unfiltered"])
    tracks_work["cell"] = tracks_work["cell"].fillna(0)
    tracks_work.loc[tracks_work.cell == cell_number_unassigned, "cell"] = 0
    if pd.api.types.is_integer_dtype(tracks["cell"].dtype):
        tracks_work["cell"] = tracks_work["cell"].astype(tracks["cell"].dtype)
    tracks_work = tracks_work.set_index("feature")

    # Recalculate the region geometry (coords, counts, centroids) of the
    # original tracks at the timesteps covered by tracks_mask; used for the
    # seed origins and to recalculate the initial velocities
    domain_size = mask.shape[1:]
    enriched_at_time = {}
    centroids = {}
    for t in tracks_times:
        if t in tracks_mask_time_index:
            labels_t = tracks_mask.isel(time=int(tracks_mask_local_frame[t]))
            features_t = tracks_work.loc[tracks_work.time == t]
            if len(features_t):
                enriched = _get_coords_and_centroids(
                    features_t, labels_t, PBC_flag=PBC_flag
                )
                enriched_at_time[t] = enriched
                centroids.update(zip(enriched.index, enriched["_centroid"]))

    # Recompute the velocities of the original tracks, using the mask-derived
    # centroids where available (as during linking) and the feature positions
    # otherwise; needed for predictive translation at the boundary
    _recompute_track_velocities(
        tracks_work, domain_size, PBC_flag=PBC_flag, centroids=centroids
    )
    # attach the recomputed velocities to the enriched frames (used as seeds)
    for enriched in enriched_at_time.values():
        enriched["_track_velocity"] = tracks_work.loc[enriched.index, "_track_velocity"]

    # Determine the timesteps of tracks that can serve as linking candidates
    # (origins) for the first new timestep: the last `memory` + 1 timesteps of
    # tracks that are present in tracks_mask. Features whose cell continues at
    # a later timestep were already matched, so they are not candidates.
    last_track_frame = time_to_frame[last_track_time]
    seed_times = [
        t
        for t in tracks_times
        if t in tracks_mask_time_index
        and (last_track_frame - time_to_frame[t]) <= memory
    ]
    cell_last_time = tracks_work.groupby("cell").time.max()
    tracks_work["_cell_last_time"] = cell_last_time[tracks_work["cell"].values].values
    wh_seed = tracks_work.time.isin(seed_times) & ~(
        (tracks_work.cell != 0) & (tracks_work.time < tracks_work["_cell_last_time"])
    )

    seed_parts = []
    for t in seed_times:
        if t not in enriched_at_time:
            continue
        seed_idx = tracks_work.index[wh_seed & (tracks_work.time == t)]
        seed_part = enriched_at_time[t].loc[seed_idx]
        if len(seed_part):
            if seed_part["_count"].isna().any():
                raise ValueError(
                    "tracks_mask does not label all features of tracks at "
                    f"timestep {t}; the labels must match the 'feature' values "
                    "of tracks."
                )
            seed_parts.append(seed_part)
    if seed_parts:
        origin_features = pd.concat(seed_parts)
        origin_features["frame"] = (
            pd.DatetimeIndex(to_timestamp(origin_features.time))
            .map(time_to_frame)
            .to_numpy()
        )
    else:
        origin_features = pd.DataFrame(
            columns=tracks_work.columns.append(
                pd.Index(["_count", "_coords", "_centroid"])
            )
        )

    # Iterate over the timesteps of the new mask that are not already tracked
    loop_features = combined_features.set_index("feature")
    loop_features["frame"] = (
        pd.DatetimeIndex(to_timestamp(loop_features.time)).map(time_to_frame).to_numpy()
    )

    def new_timesteps_iter():
        for _, timestep, labels, features_t in field_and_features_over_time(
            mask, loop_features
        ):
            if to_timestamp(timestep) in tracks_times:
                continue
            yield int(
                time_to_frame[to_timestamp(timestep)]
            ), timestep, labels, features_t

    bootstrap = bool(
        translate_method in ["drift", "predict"] and not len(origin_features)
    )
    tracks_work = _run_overlap_linking(
        new_timesteps_iter(),
        tracks_work,
        origin_features,
        bootstrap,
        memory=memory,
        max_time_gap=max_time_gap,
        minimum_overlap=minimum_overlap,
        minimum_relative_overlap=minimum_relative_overlap,
        translate_method=translate_method,
        velocity_method=velocity_method,
        velocity_constant=velocity_constant,
        PBC_flag=PBC_flag,
    )

    tracks_work = _filter_stub_cells(
        tracks_work,
        stubs=stubs,
        cell_number_start=cell_number_start,
        cell_number_unassigned=cell_number_unassigned,
        time_cell_min=time_cell_min,
        save_unfiltered_cell=has_unfiltered_cell,
    )

    tracks_work = _assign_cell_times(
        tracks_work, cell_number_unassigned=cell_number_unassigned
    )

    # Drop internal helper columns and reset index to match the combined
    # features input, re-inserting the 'feature' column in the correct location
    tracks_work = tracks_work.drop(
        columns=[
            c
            for c in [
                "_track_velocity",
                "_cell_last_time",
                "_count",
                "_coords",
                "_centroid",
                "_translated_coords",
            ]
            if c in tracks_work.columns
        ]
    )
    tracks_work = tracks_work.set_index(combined_features.index)
    tracks_work.insert(
        combined_features.columns.get_loc("feature"),
        "feature",
        combined_features.feature,
    )

    return tracks_work
