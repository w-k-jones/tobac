import datetime

import tobac.utils as tb_utils
import tobac.testing as tb_test

import pandas as pd
import pandas.testing as pd_test
import numpy as np
from scipy import fft
import xarray as xr


def test_spectral_filtering():
    """Testing tobac.utils.spectral_filtering with random test data that contains a wave signal."""

    # set wavelengths for filtering and grid spacing
    dxy = 4000
    lambda_min = 400 * 1000
    lambda_max = 1000 * 1000

    # get wavelengths for domain
    matrix = np.zeros((200, 100))
    Ni = matrix.shape[-2]
    Nj = matrix.shape[-1]
    m, n = np.meshgrid(np.arange(Ni), np.arange(Nj), indexing="ij")
    alpha = np.sqrt(m**2 / Ni**2 + n**2 / Nj**2)
    # turn off warning for zero divide here, because it is not avoidable with normalized wavenumbers
    with np.errstate(divide="ignore", invalid="ignore"):
        lambda_mn = 2 * dxy / alpha

    # seed wave signal that lies within wavelength range for filtering
    signal_min = np.where(lambda_mn[0] < lambda_min)[0].min()
    signal_idx = np.random.randint(signal_min, matrix.shape[-1])
    matrix[0, signal_idx] = 1
    wave_data = fft.idctn(matrix)

    # use spectral filtering function on random wave data
    transfer_function, filtered_data = tb_utils.general.spectral_filtering(
        dxy, wave_data, lambda_min, lambda_max, return_transfer_function=True
    )

    # a few checks on the output
    wavelengths = transfer_function[0]
    # first element in wavelengths-space is inf because normalized wavelengths are 0 here
    assert wavelengths[0, 0] == np.inf
    # the first elements should correspond to twice the distance of the corresponding axis (in m)
    # this is because the maximum spatial scale is half a wavelength through the domain
    assert wavelengths[1, 0] == (dxy) * wave_data.shape[-2] * 2
    assert wavelengths[0, 1] == (dxy) * wave_data.shape[-1] * 2

    # check that filtered/ smoothed field exhibits smaller range of values
    assert (filtered_data.max() - filtered_data.min()) < (
        wave_data.max() - wave_data.min()
    )

    # because the randomly generated wave lies outside of range that is set for filtering,
    # make sure that the filtering results in the disappearance of this signal
    assert (
        abs(
            np.floor(np.log10(abs(filtered_data.mean())))
            - np.floor(np.log10(abs(wave_data.mean())))
        )
        >= 1
    )


def test_combine_tobac_feats():
    """tests tobac.utils.combine_tobac_feats
    Test by generating two single feature dataframes,
    combining them with this function, and then
    testing to see if a single dataframe
    matches.
    """

    single_feat_1 = tb_test.generate_single_feature(
        0, 0, start_date=datetime.datetime(2022, 1, 1, 0, 0), frame_start=0
    )
    single_feat_2 = tb_test.generate_single_feature(
        1, 1, start_date=datetime.datetime(2022, 1, 1, 0, 5), frame_start=0
    )

    combined_feat = tb_utils.combine_tobac_feats([single_feat_1, single_feat_2])

    tot_feat = tb_test.generate_single_feature(
        0, 0, spd_h1=1, spd_h2=1, num_frames=2, frame_start=0
    )

    pd_test.assert_frame_equal(combined_feat, tot_feat)

    # Now try preserving the old feature numbers.
    combined_feat = tb_utils.combine_tobac_feats(
        [single_feat_1, single_feat_2], preserve_old_feat_nums="old_feat_column"
    )
    assert np.all(list(combined_feat["old_feat_column"].values) == [1, 1])
    assert np.all(list(combined_feat["feature"].values) == [1, 2])


def test_transform_feature_points():
    """Tests tobac.utils.general.transform_feature_points"""

    orig_feat_df_1 = tb_test.generate_single_feature(0, 95)
    orig_feat_df_2 = tb_test.generate_single_feature(5, 105)

    orig_feat_df = tb_utils.combine_tobac_feats([orig_feat_df_1, orig_feat_df_2])

    orig_feat_df["latitude"] = orig_feat_df["hdim_1"]
    orig_feat_df["longitude"] = orig_feat_df["hdim_2"]

    test_lat = np.linspace(-25, 24, 50)
    test_lon = np.linspace(90, 139, 50)
    in_xr = xr.Dataset(
        {"data": (("latitude", "longitude"), np.empty((50, 50)))},
        coords={"latitude": test_lat, "longitude": test_lon},
    )

    new_feat_df = tb_utils.general.transform_feature_points(
        orig_feat_df, in_xr["data"].to_iris()
    )

    assert np.all(new_feat_df["hdim_1"] == [25, 30])
    assert np.all(new_feat_df["hdim_2"] == [5, 15])

    # now test max space apart
    test_lat = np.linspace(-49, 0, 50)
    in_xr = xr.Dataset(
        {"data": (("latitude", "longitude"), np.empty((50, 50)))},
        coords={"latitude": test_lat, "longitude": test_lon},
    )

    new_feat_df = tb_utils.general.transform_feature_points(
        orig_feat_df, in_xr["data"].to_iris(), max_space_away=20000
    )

    assert np.all(new_feat_df["hdim_1"] == [49])
    assert np.all(new_feat_df["hdim_2"] == [5])

    # now test max time apart
    test_lat = np.linspace(-25, 24, 50)
    in_xr = xr.Dataset(
        {"data": (("time", "latitude", "longitude"), np.empty((2, 50, 50)))},
        coords={
            "latitude": test_lat,
            "longitude": test_lon,
            "time": [
                datetime.datetime(2023, 1, 1, 0, 0),
                datetime.datetime(2023, 1, 1, 0, 5),
            ],
        },
    )

    orig_feat_df["time"] = datetime.datetime(2023, 1, 1, 0, 0, 5)
    new_feat_df = tb_utils.general.transform_feature_points(
        orig_feat_df,
        in_xr["data"].to_iris(),
        max_time_away=datetime.timedelta(minutes=1),
    )
    # we should still have both features, but they should have the new time.
    assert np.all(new_feat_df["hdim_1"] == [25, 30])
    assert np.all(new_feat_df["hdim_2"] == [5, 15])
    assert np.all(
        new_feat_df["time"]
        == [datetime.datetime(2023, 1, 1, 0, 0), datetime.datetime(2023, 1, 1, 0, 0)]
    )

    orig_feat_df["time"] = datetime.datetime(2023, 1, 2, 0, 0)
    new_feat_df = tb_utils.general.transform_feature_points(
        orig_feat_df,
        in_xr["data"].to_iris(),
        max_time_away=datetime.timedelta(minutes=1),
    )

    assert np.all(new_feat_df["hdim_1"] == [])
    assert np.all(new_feat_df["hdim_2"] == [])
