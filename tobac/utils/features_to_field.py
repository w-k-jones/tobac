"""Function for converting feature labels back into an artificial interest field."""

import xarray as xr
import numpy as np


def features_to_field(
    features,
    template,
    *,
    blob="gaussian",
    sigma=2500.0,  # meters
    radius=2500.0,  # meters
    amplitude="threshold_value",
    position=("x", "y"),
    time_key="frame",
    mode="max",
    dtype=None,
):
    if not isinstance(template, xr.DataArray):
        raise TypeError("template must be an xarray.DataArray")
    if template.ndim < 3:
        raise ValueError("template must be at least 3D (time + 2D space)")

    tdim = template.dims[0]
    d1, d2 = template.dims[-2], template.dims[-1]  # d1="x", d2="y"

    out = xr.zeros_like(template, dtype=dtype or template.dtype)

    c1 = template[d1].values  # x coords (meters)
    c2 = template[d2].values  # y coords (meters)
    C1, C2 = np.meshgrid(c1, c2, indexing="ij")  # shape (len(x), len(y))

    if blob == "gaussian":

        def kernel(p1, p2, amp):
            rr2 = (C1 - p1) ** 2 + (C2 - p2) ** 2
            return amp * np.exp(-0.5 * rr2 / (sigma**2))

    elif blob == "tophat":

        def kernel(p1, p2, amp):
            rr2 = (C1 - p1) ** 2 + (C2 - p2) ** 2
            return amp * (rr2 <= radius**2)

    else:
        raise ValueError(f"Unknown blob={blob}")

    p1_col, p2_col = position

    def get_amp(row):
        if isinstance(amplitude, (int, float)):
            return float(amplitude)
        return float(row[amplitude])

    for _, row in features.iterrows():
        if time_key == "frame":
            tidx = int(row["frame"])
            selector = {tdim: out[tdim].values[tidx]}
        elif time_key == "time":
            selector = {tdim: np.datetime64(row["time"])}
        else:
            raise ValueError("time_key must be 'frame' or 'time'")

        p1 = float(row[p1_col])  # x position in meters
        p2 = float(row[p2_col])  # y position in meters
        amp = get_amp(row)

        blob2d = kernel(p1, p2, amp)
        current = out.sel(selector).values

        if mode == "add":
            out.loc[selector] = current + blob2d
        elif mode == "max":
            out.loc[selector] = np.maximum(current, blob2d)
        else:
            raise ValueError("mode must be 'add' or 'max'")

    return out
