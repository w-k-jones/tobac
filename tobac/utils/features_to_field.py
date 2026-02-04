"""Function for converting feature labels back into an artificial interest field."""

import xarray as xr
import numpy as np


def features_to_interest_field(
    features,
    template: xr.DataArray,
    *,
    position_mode="hdim",  # "hdim" or "xy"
    position_cols=None,  # override
    time_key="frame",  # or "time"
    blob="gaussian",  # "gaussian" or "tophat"
    mode="max",  # "max" or "add"
    amp_from="threshold_value",  # column name or constant float
    amp_factor=2.0,  # amp = amp_factor * amp_from (if column)
    size_from="area",  # column name; if missing uses sigma
    sigma=5.0,  # default sigma (grid points for hdim)
    min_sigma=1.0,
):
    if not isinstance(template, xr.DataArray):
        raise TypeError("template must be an xarray.DataArray")
    if template.ndim < 3:
        raise ValueError("template must be at least 3D (time + 2D space)")

    tdim = template.dims[0]
    d1, d2 = template.dims[-2], template.dims[-1]
    out = xr.zeros_like(template, dtype=float)

    n1, n2 = template.sizes[d1], template.sizes[d2]

    # choose position columns
    if position_cols is None:
        if position_mode == "hdim":
            position_cols = ("hdim_1", "hdim_2")
        elif position_mode == "xy":
            position_cols = ("x", "y")
        else:
            raise ValueError("position_mode must be 'hdim' or 'xy'")

    p1_col, p2_col = position_cols

    # coordinate grids
    if position_mode == "hdim":
        C1, C2 = np.meshgrid(np.arange(n1), np.arange(n2), indexing="ij")

        # sigma is in grid points
        def rr2(p1, p2):
            return (C1 - p1) ** 2 + (C2 - p2) ** 2

    else:  # "xy" physical coords
        c1 = template[d1].values
        c2 = template[d2].values
        C1, C2 = np.meshgrid(c1, c2, indexing="ij")

        # sigma is in same units as coords
        def rr2(p1, p2):
            return (C1 - p1) ** 2 + (C2 - p2) ** 2

    def amplitude(row):
        if isinstance(amp_from, (int, float)):
            return float(amp_from)
        return amp_factor * float(row[amp_from])

    def sigma_for_row(row):
        if (
            size_from is not None
            and size_from in row.index
            and np.isfinite(row[size_from])
        ):
            area = float(row[size_from])
            if area > 0:
                r = np.sqrt(area / np.pi)
                s = max(min_sigma, r / 2.0)
                return s
        return float(sigma)

    for _, row in features.iterrows():
        # time selection
        if time_key == "frame":
            tidx = int(row["frame"])
            selector = {tdim: out[tdim].values[tidx]}
        elif time_key == "time":
            selector = {tdim: np.datetime64(row["time"])}
        else:
            raise ValueError("time_key must be 'frame' or 'time'")

        p1 = float(row[p1_col])
        p2 = float(row[p2_col])
        amp = amplitude(row)
        sig = sigma_for_row(row)

        r2 = rr2(p1, p2)

        if blob == "gaussian":
            blob2d = amp * np.exp(-0.5 * r2 / (sig**2))
        elif blob == "tophat":
            blob2d = amp * (r2 <= (sig**2))
        else:
            raise ValueError("blob must be 'gaussian' or 'tophat'")

        current = out.sel(selector).values
        if mode == "add":
            out.loc[selector] = current + blob2d
        elif mode == "max":
            out.loc[selector] = np.maximum(current, blob2d)
        else:
            raise ValueError("mode must be 'add' or 'max'")

    return out
