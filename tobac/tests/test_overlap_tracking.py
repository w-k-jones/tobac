import numpy as np
from tobac.tracking import calc_proportional_overlap, find_overlapping_labels


def test_calc_proportional_overlap():
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
