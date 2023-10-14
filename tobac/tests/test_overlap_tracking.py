"""
Tests for overlap tracking
"""
import numpy as np
import pandas as pd
import xarray as xr
import tobac.testing


# Test import
def test_import() -> None:
    import tobac.tracking.tracking_overlap


def test_calc_proportional_overlap():
    from tobac.tracking.tracking_overlap import calc_proportional_overlap

    assert calc_proportional_overlap(10, 10, 10) == 1
    assert calc_proportional_overlap(0, 10, 10) == 0
    assert calc_proportional_overlap(5, 10, 10) == 0.5
    assert calc_proportional_overlap(5, 10, 20) == 10 / 30
    assert calc_proportional_overlap(5, 10, 30) == 0.25

    # Test array input
    assert np.all(
        calc_proportional_overlap(
            np.array([10, 0, 5, 5, 5]),
            np.array([10, 10, 10, 10, 10]),
            np.array([10, 10, 10, 20, 30]),
        )
        == np.array([1, 0, 0.5, 10 / 30, 0.25])
    )


def test_find_overlapping_labels():
    from tobac.tracking.tracking_overlap import find_overlapping_labels

    test_labels = np.zeros([4, 6], dtype=int)
    test_labels[1:3, 1:3] = 1
    test_labels[3:4, 3:5] = 2
    test_labels[1:2, 5:6] = 4

    test_bins = np.bincount(test_labels.ravel())

    locs_array = np.arange(24, dtype=int).reshape([4, 6])

    # Test no expected overlap
    assert find_overlapping_labels(1, np.ndarray([0]), test_labels, test_bins) == []

    # Test overlap for label 1
    assert find_overlapping_labels(
        1, locs_array[1:3, 1:3].ravel(), test_labels, test_bins
    ) == [[1, 1, 4]]

    # Test partial overlap
    assert find_overlapping_labels(
        1, locs_array[1:3, 1:2].ravel(), test_labels, test_bins
    ) == [[1, 1, 2]]

    # Test overlap for label 2
    assert find_overlapping_labels(
        1, locs_array[3:4, 3:5].ravel(), test_labels, test_bins
    ) == [[1, 2, 2]]

    # Test overlap for label 4
    assert find_overlapping_labels(
        1, locs_array[1:2, 5:6].ravel(), test_labels, test_bins
    ) == [[1, 4, 1]]

    # Test changing label number
    assert find_overlapping_labels(
        42, locs_array[1:3, 1:3].ravel(), test_labels, test_bins
    ) == [[42, 1, 4]]

    # Test finding multiple labels
    assert find_overlapping_labels(
        1, locs_array[1:4, 1:5].ravel(), test_labels, test_bins
    ) == [[1, 1, 4], [1, 2, 2]]
    assert find_overlapping_labels(
        1, locs_array[1:4, 1:6].ravel(), test_labels, test_bins
    ) == [[1, 1, 4], [1, 2, 2], [1, 4, 1]]
    assert find_overlapping_labels(
        1, locs_array[2:4, 1:4].ravel(), test_labels, test_bins
    ) == [[1, 1, 2], [1, 2, 1]]

    # Test min_absolute_overlap
    assert find_overlapping_labels(
        1, locs_array[1:4, 1:6].ravel(), test_labels, test_bins, min_absolute_overlap=1
    ) == [[1, 1, 4], [1, 2, 2], [1, 4, 1]]
    assert find_overlapping_labels(
        1, locs_array[1:4, 1:6].ravel(), test_labels, test_bins, min_absolute_overlap=2
    ) == [[1, 1, 4], [1, 2, 2]]
    assert find_overlapping_labels(
        1, locs_array[1:4, 1:6].ravel(), test_labels, test_bins, min_absolute_overlap=3
    ) == [[1, 1, 4]]
    assert (
        find_overlapping_labels(
            1,
            locs_array[1:4, 1:6].ravel(),
            test_labels,
            test_bins,
            min_absolute_overlap=5,
        )
        == []
    )
    assert find_overlapping_labels(
        1, locs_array[2:4, 1:4].ravel(), test_labels, test_bins, min_absolute_overlap=2
    ) == [[1, 1, 2]]
    assert (
        find_overlapping_labels(
            1,
            locs_array[2:4, 1:4].ravel(),
            test_labels,
            test_bins,
            min_absolute_overlap=3,
        )
        == []
    )

    # Test min_relative_overlap
    assert find_overlapping_labels(
        1,
        locs_array[1:3, 1:2].ravel(),
        test_labels,
        test_bins,
        min_relative_overlap=4 / 6,
    ) == [[1, 1, 2]]
    assert (
        find_overlapping_labels(
            1,
            locs_array[1:3, 1:2].ravel(),
            test_labels,
            test_bins,
            min_relative_overlap=5 / 6,
        )
        == []
    )
    assert find_overlapping_labels(
        1,
        locs_array[1:4, 1:6].ravel(),
        test_labels,
        test_bins,
        min_relative_overlap=2 / 16,
    ) == [[1, 1, 4], [1, 2, 2], [1, 4, 1]]
    assert find_overlapping_labels(
        1,
        locs_array[1:4, 1:6].ravel(),
        test_labels,
        test_bins,
        min_relative_overlap=3 / 17,
    ) == [[1, 1, 4], [1, 2, 2]]
    assert find_overlapping_labels(
        1,
        locs_array[1:4, 1:6].ravel(),
        test_labels,
        test_bins,
        min_relative_overlap=8 / 19,
    ) == [[1, 1, 4]]
    assert (
        find_overlapping_labels(
            1,
            locs_array[1:4, 1:6].ravel(),
            test_labels,
            test_bins,
            min_relative_overlap=9 / 19,
        )
        == []
    )
    assert find_overlapping_labels(
        1,
        locs_array[2:4, 1:4].ravel(),
        test_labels,
        test_bins,
        min_relative_overlap=4 / 10,
    ) == [[1, 1, 2]]
    assert (
        find_overlapping_labels(
            1,
            locs_array[2:4, 1:4].ravel(),
            test_labels,
            test_bins,
            min_relative_overlap=5 / 10,
        )
        == []
    )

    # Test no locs
    assert (
        find_overlapping_labels(1, locs_array[1:1, 1:1].ravel(), test_labels, test_bins)
        == []
    )


def test_remove_stubs() -> None:
    from tobac.tracking.tracking_overlap import remove_stubs

    test_features = pd.DataFrame({"cell": [1, 1, 1]})

    assert np.all(remove_stubs(test_features, 1, -1).cell == 1)
    assert np.all(remove_stubs(test_features, 3, -1).cell == 1)
    assert np.all(remove_stubs(test_features, 4, -1).cell == -1)

    test_features = pd.DataFrame({"cell": [1, 1, 2, 2, 2]})
    assert np.all(
        remove_stubs(test_features, 3, 999).cell == np.array([999, 999, 2, 2, 2])
    )


def test_linking_overlap_timestep() -> None:
    from tobac.tracking.tracking_overlap import linking_overlap_timestep

    test_features = tobac.testing.generate_single_feature(
        5,
        5,
        min_h1=0,
        max_h1=10,
        min_h2=0,
        max_h2=15,
        frame_start=0,
        num_frames=2,
        spd_h1=0,
        spd_h2=0,
        PBC_flag="none",
    )

    test_features["cell"] = [-1, -1]

    test_current_step = np.zeros([10, 15], dtype=np.int64)
    test_current_step[4:7, 4:7] = 1
    test_current_step = xr.DataArray(test_current_step, dims=("hdim_1", "hdim_2"))

    test_next_step = np.zeros([10, 15], dtype=np.int64)
    test_next_step[4:7, 4:7] = 2
    test_next_step = xr.DataArray(test_next_step, dims=("hdim_1", "hdim_2"))

    # Test min_absolute_overlap
    assert np.all(
        linking_overlap_timestep(
            test_features.copy(), test_current_step, test_next_step, 1, -1
        )[0].cell
        == 1
    )
    assert np.all(
        linking_overlap_timestep(
            test_features.copy(),
            test_current_step,
            test_next_step,
            1,
            -1,
            min_absolute_overlap=10,
        )[0].cell
        == -1
    )

    # Test min_relative_overlap
    test_next_step = np.zeros([10, 15], dtype=np.int64)
    test_next_step[4:7, 5:8] = 2
    test_next_step = xr.DataArray(test_next_step, dims=("hdim_1", "hdim_2"))

    # relative overlap should be 2/3
    assert np.all(
        linking_overlap_timestep(
            test_features.copy(),
            test_current_step,
            test_next_step,
            2,
            -1,
            min_relative_overlap=0.66,
        )[0].cell
        == 2
    )
    assert np.all(
        linking_overlap_timestep(
            test_features.copy(),
            test_current_step,
            test_next_step,
            2,
            -1,
            min_relative_overlap=0.67,
        )[0].cell
        == -1
    )

    # Test max_dist
    test_features.loc[1, ["hdim_2"]] = 10
    assert np.all(
        linking_overlap_timestep(
            test_features.copy(), test_current_step, test_next_step, 3, -1, max_dist=6
        )[0].cell
        == 3
    )
    assert np.all(
        linking_overlap_timestep(
            test_features.copy(), test_current_step, test_next_step, 3, -1, max_dist=4
        )[0].cell
        == -1
    )


def test_linking_overlap() -> None:
    from tobac.tracking.tracking_overlap import linking_overlap

    test_features = tobac.testing.generate_single_feature(
        5,
        5,
        min_h1=0,
        max_h1=10,
        min_h2=0,
        max_h2=15,
        frame_start=0,
        num_frames=3,
        spd_h1=0,
        spd_h2=0,
        PBC_flag="none",
    )

    test_step1 = np.zeros([10, 15], dtype=np.int64)
    test_step1[4:7, 4:7] = 1

    test_step2 = np.zeros([10, 15], dtype=np.int64)
    test_step2[4:7, 4:7] = 2

    test_step3 = np.zeros([10, 15], dtype=np.int64)
    test_step3[4:7, 4:7] = 3

    test_segments = np.stack([test_step1, test_step2, test_step3], 0)
    test_segments = xr.DataArray(test_segments, dims=("time", "hdim_1", "hdim_2"))

    assert np.all(linking_overlap(test_features, test_segments, 300, 1).cell == 1)
    assert np.all(
        linking_overlap(test_features, test_segments, 300, 1, stubs=4).cell == -1
    )

    # Check that we do not modify the input features dataframe
    assert "cell" not in test_features
