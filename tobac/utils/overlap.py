from functools import partial
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
from skimage.measure import regionprops_table

import tobac.utils.internal as internal_utils
from tobac.feature_detection import feature_position
from tobac.utils.datetime import to_timestamp
from tobac.utils.neighbors import FeatureBallTree


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
                regionprops_table(mask.values, properties=("label", "area", "coords"))
            )
            .rename(columns=dict(label="feature", area="_count", coords="_coords"))
            .set_index("feature")
        )

        props_df["_centroid"] = props_df.apply(fp_lambda, axis=1)
    else:
        props_df = pd.DataFrame(columns=["_count", "_coords", "_centroid"])

    return features.join(props_df)


def _transpose_mask(
    mask: xr.DataArray,
    vertical_coord: Optional[str] = None,
    vertical_axis: Optional[int] = None,
) -> xr.DataArray:
    """Transpose the segmentation mask for 3D data to ensure the dimensions are in the
    expected order (time, vdim, hdim1, hdim2).

    Parameters
    ----------
    mask : xr.DataArray
        Input segmentation mask
    vertical_coord : Optional[str], optional
        vertical coordinate name, by default None
    vertical_axis : Optional[int], optional
        vertical axis number, by default None

    Returns
    -------
    xr.DataArray
        Input segmentation mask, with dimensions reordered if necessary.

    Raises
    ------
    ValueError
        If the vertical coord cannot be found
    ValueError
        If the vertical axis number is negative
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

        return mask.transpose(
            *[
                mask.dims[i]
                for i in [time_axis, vertical_axis, hdim_1_axis, hdim_2_axis]
            ]
        )
    return mask


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

    min_counts = np.minimum(
        counts,
        [
            destination_counts[k] if k in destination_counts else 0
            for k in matched_labels
        ],
    )

    wh_valid_label = min_counts > 0
    matched_labels = matched_labels[wh_valid_label]
    matched_counts = matched_counts[wh_valid_label]
    min_counts = min_counts[wh_valid_label]

    wh = np.logical_and(
        matched_counts >= min_count,
        matched_counts / min_counts >= relative_count,
    )

    return matched_labels[wh], matched_counts[wh]


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
            if "_track_velocity" not in origin_features:
                origin_features["_track_velocity"] = None
        else:
            if "_track_velocity" not in destination_features:
                destination_features["_track_velocity"] = None


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
            mean_vel = features._track_velocity.mean()
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
    hdim1_size: int,
    hdim2_size: int,
    vdim_size: Optional[int] = None,
    PBC_flag: Optional[Literal["none", "hdim_1", "hdim_2", "both"]] = None,
) -> np.ndarray[int]:
    """Wrap coordinates according to periodic boundary conditions. Coordinate locations which fall outside the size of the array are removed.

    Parameters
    ----------
    coords : np.ndarray[int]
        Coordinate array to wrap. Shape is (n_dims, n_points).
    hdim1_size : int
        Size of the first horizontal dimension.
    hdim2_size : int
        Size of the second horizontal dimension.
    vdim_size : int, optional
        Size of the vertical dimension. If None, no vertical wrapping applied.
    PBC_flag : {'none', 'hdim_1', 'hdim_2', 'both'}, optional
        Specification of which dimensions have periodic boundaries.
        Default is None.

    Returns
    -------
    np.ndarray[int]
        Wrapped coordinate array with invalid points filtered out.

    """
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
        hdim1_size=hdim1_size,
        hdim2_size=hdim2_size,
        vdim_size=vdim_size,
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
                            features._track_velocity.mean()
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
    return overlap_candidates
