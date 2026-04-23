"""Function for converting feature labels back into an artificial interest field."""

import xarray as xr
import numpy as np
from typing import Literal, Optional, Tuple, Union


def features_to_interest_field(
    features,
    template: xr.DataArray,  # field for correct geometrie
    *,
    position_mode: Literal["hdim", "xy"] = "hdim",
    position_cols: Optional[Tuple[str, ...]] = None,
    time_key: Literal["frame", "time"] = "frame",
    blob: Literal["gaussian", "tophat"] = "gaussian",
    mode: Literal["max", "add"] = "max",
    amp_from: Union[str, float] = "threshold_value",
    amp_factor: float = 2.0,
    size_from: Optional[str] = "area",
    sigma: float = 5.0,
    min_sigma: float = 1.0,
):
    """
    Internal function to reconstruct an artificial interest field from feature
    positions.

    Parameters
    ----------
    features : pandas.DataFrame
        Table containing feature positions, times, and optional amplitude and
        size information.
    template : xarray.DataArray
        Reference array defining the output shape, dimensions, and coordinates.
    position_mode : {"hdim", "xy"}, optional
        Convention used to interpret spatial positions.
    position_cols : sequence of str, optional
        Column names containing the spatial coordinates of each feature.
    time_key : {"frame", "time"}, optional
        Column used to assign features to time steps.
    blob : {"gaussian", "tophat"}, optional
        Blob shape used to reconstruct each feature.
    mode : {"max", "add"}, optional
        Method used to combine overlapping blobs.
    amp_from : str or float, optional
        Column name or constant value used to define blob amplitude.
    amp_factor : float, optional
        Factor applied to amplitudes derived from `amp_from`.
    size_from : str, optional
        Column name used to estimate blob size.
    sigma : float, optional
        Default blob width or radius.
    min_sigma : float, optional
        Minimum allowed blob size.

    Returns
    -------
    xarray.DataArray
        Output interest field with the same structure as `template`.
    """

    if not isinstance(template, xr.DataArray):
        raise TypeError("template must be an xarray.DataArray")

    if template.ndim < 3:
        raise ValueError("template must be at least 3D (time + spatial dims)")

    tdim = template.dims[0]
    spatial_dims = template.dims[1:]
    n_spatial = len(spatial_dims)

    out = xr.zeros_like(template, dtype=float)

    sizes = [template.sizes[d] for d in spatial_dims]

    # default position columns
    if position_cols is None:
        if position_mode == "hdim":
            if n_spatial == 2:
                position_cols = ("hdim_1", "hdim_2")
            elif n_spatial == 3:
                position_cols = ("vdim", "hdim_1", "hdim_2")
        elif position_mode == "xy":
            position_cols = spatial_dims
        else:
            raise ValueError("position_mode must be 'hdim' or 'xy'")

    # coordinate grids
    if position_mode == "hdim":
        coords = [np.arange(n) for n in sizes]
    else:
        coords = [template[d].values for d in spatial_dims]

    grids = np.meshgrid(*coords, indexing="ij")

    for _, row in features.iterrows():

        if time_key == "frame":
            tidx = int(row["frame"])
            selector = {tdim: out[tdim].values[tidx]}
        elif time_key == "time":
            selector = {tdim: np.datetime64(row["time"])}
        else:
            raise ValueError("time_key must be 'frame' or 'time'")

        pos = [float(row[c]) for c in position_cols]

        if isinstance(amp_from, (int, float)):
            amp = float(amp_from)
        else:
            amp = amp_factor * float(row[amp_from])

        if (
            size_from is not None
            and size_from in row.index
            and np.isfinite(row[size_from])
        ):
            area = float(row[size_from])
            if area > 0:
                r = np.sqrt(area / np.pi)
                sig = max(min_sigma, r / 2.0)
            else:
                sig = float(sigma)
        else:
            sig = float(sigma)

        r2 = 0
        for g, p in zip(grids, pos):
            r2 += (g - p) ** 2

        if blob == "gaussian":
            blob_nd = amp * np.exp(-0.5 * r2 / (sig**2))
        elif blob == "tophat":
            blob_nd = amp * (r2 <= (sig**2))
        else:
            raise ValueError("blob must be 'gaussian' or 'tophat'")

        current = out.sel(selector).values

        if mode == "add":
            out.loc[selector] = current + blob_nd
        elif mode == "max":
            out.loc[selector] = np.maximum(current, blob_nd)
        else:
            raise ValueError("mode must be 'add' or 'max'")

    return out
