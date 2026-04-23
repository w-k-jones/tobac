import numpy as np
import pandas as pd
import xarray as xr
import pytest

from tobac.utils.features_to_field import features_to_interest_field


def make_template():
    """
    Create a template interest field with time, x, and y dimensions.
    """
    time = np.arange(5)
    x = np.arange(50)
    y = np.arange(60)

    data = xr.DataArray(
        np.zeros((len(time), len(x), len(y))),
        dims=("time", "x", "y"),
        coords={"time": time, "x": x, "y": y},
        name="interest",
    )
    return data


def test_empty_features_returns_zero_field():
    """
    Test that empty features return an output field filled with zeros.
    """
    template = make_template()

    features = pd.DataFrame(columns=["frame", "hdim_1", "hdim_2", "threshold_value"])

    out = features_to_interest_field(
        features,
        template,
        position_mode="hdim",
        amp_from="threshold_value",
    )

    assert out.shape == template.shape
    assert float(out.max()) == 0.0
    assert float(out.min()) == 0.0


def test_amplitude_scaling_from_threshold():
    """
    Test that the output amplitude is scaled correctly from threshold_value.
    """
    template = make_template()

    features = pd.DataFrame(
        [
            {
                "frame": 0,
                "hdim_1": 20.0,
                "hdim_2": 30.0,
                "threshold_value": 0.3,
            }
        ]
    )

    out = features_to_interest_field(
        features,
        template,
        position_mode="hdim",
        amp_from="threshold_value",
        amp_factor=2.0,
        sigma=3.0,
        size_from=None,
        mode="max",
    )

    assert np.isclose(float(out.max()), 0.6, atol=1e-6)


def test_blob_center_is_at_feature_position():
    """
    Test that the blob maximum is located at the feature position.
    """
    template = make_template()

    features = pd.DataFrame(
        [
            {
                "frame": 2,
                "hdim_1": 10.0,
                "hdim_2": 15.0,
                "threshold_value": 1.0,
            }
        ]
    )

    out = features_to_interest_field(
        features,
        template,
        position_mode="hdim",
        amp_from="threshold_value",
        amp_factor=1.0,
        sigma=2.0,
        size_from=None,
        mode="max",
    )

    val = out.isel(time=2, x=10, y=15).item()
    assert np.isclose(val, 1.0, atol=1e-6)


def test_add_vs_max_overlap():
    """
    Test that overlapping blobs are combined differently for add and max modes.
    """
    template = make_template()

    features = pd.DataFrame(
        [
            {"frame": 0, "hdim_1": 25.0, "hdim_2": 30.0, "threshold_value": 1.0},
            {"frame": 0, "hdim_1": 25.0, "hdim_2": 30.0, "threshold_value": 1.0},
        ]
    )

    out_add = features_to_interest_field(
        features,
        template,
        position_mode="hdim",
        amp_from="threshold_value",
        amp_factor=1.0,
        sigma=2.0,
        size_from=None,
        mode="add",
    )

    out_max = features_to_interest_field(
        features,
        template,
        position_mode="hdim",
        amp_from="threshold_value",
        amp_factor=1.0,
        sigma=2.0,
        size_from=None,
        mode="max",
    )

    assert float(out_add.max()) > float(out_max.max())
    assert np.isclose(float(out_max.max()), 1.0, atol=1e-6)


def test_sigma_from_area_changes_blob_width():
    """
    Test that blob width is derived from area when size_from is set.
    """
    template = make_template()

    features = pd.DataFrame(
        [
            {
                "frame": 0,
                "hdim_1": 25.0,
                "hdim_2": 30.0,
                "threshold_value": 1.0,
                "area": 100.0,
            }
        ]
    )

    out = features_to_interest_field(
        features,
        template,
        position_mode="hdim",
        amp_from="threshold_value",
        amp_factor=1.0,
        sigma=1.0,
        size_from="area",
        mode="max",
    )

    center = out.isel(time=0, x=25, y=30).item()
    off = out.isel(time=0, x=25, y=35).item()

    assert off > 0.0
    assert off < center


def test_invalid_position_mode_raises():
    """
    Test that an invalid position_mode raises a ValueError.
    """
    template = make_template()

    features = pd.DataFrame(
        [{"frame": 0, "hdim_1": 10.0, "hdim_2": 10.0, "threshold_value": 1.0}]
    )

    with pytest.raises(ValueError):
        features_to_interest_field(
            features,
            template,
            position_mode="invalid",
        )
