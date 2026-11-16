"""Tests for tobac.merge_split.merge_split_overlap"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import tobac
import tobac.testing
from tobac.merge_split import merge_split_overlap


class TestMergeSplitOverlap:
    """Test suite for merge_split_overlap."""

    @staticmethod
    def _make_mask_and_features(times, blobs, grid_size=20):
        """Build a labeled 2D mask and a matching tracks dataframe.

        Parameters
        ----------
        times : list of pd.Timestamp
            Time coordinates of the mask.
        blobs : list of tuple (frame_indices, row_slice, col_slice, cell)
            Each blob is labeled with consecutive feature IDs (1-based) at the
            given frame indices and assigned the given cell ID.
        grid_size : int
            Size of the square (grid_size x grid_size) horizontal grid.

        Returns
        -------
        mask : xr.DataArray
            Labeled mask with dims (time, hdim_1, hdim_2).
        tracks : pd.DataFrame
            Tracks dataframe with one row per blob per frame index, including
            a 'cell' column.
        """
        mask_data = np.zeros((len(times), grid_size, grid_size), dtype=int)
        feature_rows = []
        feature_id = 0
        for frame_indices, rows, cols, cell in blobs:
            for frame in frame_indices:
                feature_id += 1
                mask_data[frame, rows, cols] = feature_id
                feature_rows.append(
                    {
                        "feature": feature_id,
                        "frame": frame,
                        "time": times[frame],
                        "hdim_1": float(np.mean(np.arange(grid_size)[rows])),
                        "hdim_2": float(np.mean(np.arange(grid_size)[cols])),
                        "cell": cell,
                    }
                )
        mask = xr.DataArray(
            mask_data,
            coords={
                "time": list(times),
                "hdim_1": np.arange(grid_size),
                "hdim_2": np.arange(grid_size),
            },
            dims=["time", "hdim_1", "hdim_2"],
        )
        return mask, pd.DataFrame(feature_rows)

    @staticmethod
    def _make_mask_and_features_3d(times, blobs, grid_size=10, vdim_size=4):
        """Build a labeled 3D mask and a matching tracks dataframe.

        Parameters
        ----------
        times : list of pd.Timestamp
            Time coordinates of the mask.
        blobs : list of tuple (frame_indices, vdim_slice, row_slice, col_slice, cell)
            Each blob is labeled with consecutive feature IDs (1-based) at the
            given frame indices and assigned the given cell ID.
        grid_size : int
            Size of the square horizontal grid.
        vdim_size : int
            Size of the vertical dimension.

        Returns
        -------
        mask : xr.DataArray
            Labeled mask with dims (time, vdim, hdim_1, hdim_2).
        tracks : pd.DataFrame
            Tracks dataframe with one row per blob per frame index, including
            'vdim' and 'cell' columns.
        """
        mask_data = np.zeros((len(times), vdim_size, grid_size, grid_size), dtype=int)
        feature_rows = []
        feature_id = 0
        for frame_indices, vdim, rows, cols, cell in blobs:
            for frame in frame_indices:
                feature_id += 1
                mask_data[frame, vdim, rows, cols] = feature_id
                feature_rows.append(
                    {
                        "feature": feature_id,
                        "frame": frame,
                        "time": times[frame],
                        "vdim": float(np.mean(np.arange(vdim_size)[vdim])),
                        "hdim_1": float(np.mean(np.arange(grid_size)[rows])),
                        "hdim_2": float(np.mean(np.arange(grid_size)[cols])),
                        "cell": cell,
                    }
                )
        mask = xr.DataArray(
            mask_data,
            coords={
                "time": list(times),
                "vdim": np.arange(vdim_size),
                "hdim_1": np.arange(grid_size),
                "hdim_2": np.arange(grid_size),
            },
            dims=["time", "vdim", "hdim_1", "hdim_2"],
        )
        return mask, pd.DataFrame(feature_rows)

    def test_basic_merge_2d(self):
        """Two cells that merge into one should share a single track."""
        times = pd.date_range("2024-01-01", periods=4, freq="h")
        mask, tracks = self._make_mask_and_features(
            times,
            [
                ([0, 1], slice(2, 6), slice(2, 6), 1),
                ([0, 1], slice(6, 10), slice(6, 10), 2),
                ([2, 3], slice(3, 9), slice(3, 9), 3),  # merged cell overlaps both
            ],
        )
        result = merge_split_overlap(tracks, mask)

        assert "track" in result.columns
        assert "time_track" in result.columns
        assert len(result) == len(tracks)
        # note: the output index values are preserved but get name='index'
        pd.testing.assert_index_equal(
            result.index, tracks.index, check_names=False
        )
        assert result.track.nunique() == 1
        assert (result.track == 1).all()

    def test_split_2d(self):
        """A cell that splits into two should keep all children in one track.

        Regression test: currently fails because cell_map is built with
        dict(zip(...)), which keeps only one edge when a single origin cell
        overlaps multiple destination cells at the same timestep.
        """
        times = pd.date_range("2024-01-01", periods=2, freq="h")
        mask, tracks = self._make_mask_and_features(
            times,
            [
                ([0], slice(4, 10), slice(4, 10), 1),
                ([1], slice(4, 7), slice(4, 7), 2),
                ([1], slice(7, 10), slice(7, 10), 3),
            ],
        )
        result = merge_split_overlap(tracks, mask)
        assert result.track.nunique() == 1

    def test_split_with_overlapping_children(self):
        """Split where the two children also overlap each other.

        Regression test: same dict(zip(...)) edge-loss bug as test_split_2d.
        """
        times = pd.date_range("2024-01-01", periods=2, freq="h")
        mask, tracks = self._make_mask_and_features(
            times,
            [
                ([0], slice(2, 10), slice(2, 10), 1),
                ([1], slice(2, 7), slice(2, 7), 2),
                ([1], slice(5, 10), slice(5, 10), 3),
            ],
        )
        result = merge_split_overlap(tracks, mask)
        assert result.track.nunique() == 1

    def test_chained_merges(self):
        """Cells merging over several consecutive timesteps form one track."""
        times = pd.date_range("2024-01-01", periods=5, freq="h")
        mask, tracks = self._make_mask_and_features(
            times,
            [
                ([0], slice(2, 6), slice(2, 6), 1),
                ([0], slice(6, 10), slice(6, 10), 2),
                ([1, 2], slice(3, 9), slice(3, 9), 3),
                ([3], slice(3, 9), slice(3, 9), 4),
                ([4], slice(2, 10), slice(2, 10), 5),
            ],
        )
        result = merge_split_overlap(tracks, mask)
        assert result.track.nunique() == 1
        assert (result.track == 1).all()

    def test_no_overlap_between_timesteps(self):
        """Cells with no overlap between timesteps get separate tracks."""
        times = pd.date_range("2024-01-01", periods=4, freq="h")
        mask, tracks = self._make_mask_and_features(
            times,
            [
                ([0, 1], slice(2, 5), slice(2, 5), 1),
                ([2, 3], slice(12, 15), slice(12, 15), 2),
            ],
        )
        result = merge_split_overlap(tracks, mask)
        assert result.track.tolist() == [1, 1, 2, 2]

    def test_minimum_overlap(self):
        """Test the minimum_overlap threshold between cells."""
        times = pd.date_range("2024-01-01", periods=2, freq="h")
        # 3x3 blob at t0, shifted by one column at t1: overlap is 3x2 = 6 px
        mask, tracks = self._make_mask_and_features(
            times,
            [
                ([0], slice(2, 5), slice(2, 5), 1),
                ([1], slice(2, 5), slice(3, 6), 2),
            ],
        )
        result_linked = merge_split_overlap(tracks, mask, minimum_overlap=6)
        result_unlinked = merge_split_overlap(tracks, mask, minimum_overlap=7)

        assert result_linked.track.tolist() == [1, 1]
        assert result_unlinked.track.tolist() == [1, 2]

    def test_minimum_relative_overlap(self):
        """Test the minimum_relative_overlap threshold between cells."""
        times = pd.date_range("2024-01-01", periods=2, freq="h")
        # 4x4 blob at t0, 4x4 blob at t1 shifted so only 1 px overlaps:
        # relative overlap is 1/16 = 0.0625 of the smaller feature
        mask, tracks = self._make_mask_and_features(
            times,
            [
                ([0], slice(2, 6), slice(2, 6), 1),
                ([1], slice(5, 9), slice(5, 9), 2),
            ],
        )
        result_high = merge_split_overlap(
            tracks, mask, minimum_relative_overlap=0.5
        )
        result_low = merge_split_overlap(
            tracks, mask, minimum_relative_overlap=0.0625
        )

        assert result_high.track.tolist() == [1, 2]
        assert result_low.track.tolist() == [1, 1]

    def test_pbc_crossing_boundary(self):
        """A cell moving across the hdim_2 boundary only links with the matching PBC."""
        times = pd.date_range("2024-01-01", periods=2, freq="h")
        mask, tracks = self._make_mask_and_features(
            times,
            [
                ([0], slice(2, 5), slice(8, 10), 1),  # right edge
                ([1], slice(2, 5), slice(0, 2), 2),  # left edge: moved +2 across boundary
            ],
            grid_size=10,
        )
        # no shared pixels, so no link even with PBCs
        result = merge_split_overlap(tracks, mask, PBC_flag="both")
        assert result.track.tolist() == [1, 2]

        # constant velocity of +2 px/h in hdim_2 wraps onto the left edge
        # only if hdim_2 is periodic
        kwargs = dict(
            translate_method="constant",
            velocity_constant=np.array([0, 2 / 3600]),
            minimum_overlap=4,
        )
        result_pbc = merge_split_overlap(tracks, mask, PBC_flag="both", **kwargs)
        assert result_pbc.track.tolist() == [1, 1]

        result_no_pbc = merge_split_overlap(tracks, mask, PBC_flag="none", **kwargs)
        assert result_no_pbc.track.tolist() == [1, 2]

        result_wrong_pbc = merge_split_overlap(tracks, mask, PBC_flag="hdim_1", **kwargs)
        assert result_wrong_pbc.track.tolist() == [1, 2]

    def test_basic_merge_3d(self):
        """Test a merge in 3D data with a vertical coordinate."""
        times = pd.date_range("2024-01-01", periods=2, freq="h")
        mask, tracks = self._make_mask_and_features_3d(
            times,
            [
                ([0], slice(1, 3), slice(2, 5), slice(2, 5), 1),
                ([1], slice(1, 3), slice(3, 6), slice(3, 6), 2),
            ],
        )
        result = merge_split_overlap(tracks, mask, vertical_coord="vdim")
        assert result.track.tolist() == [1, 1]

    def test_translate_method_constant(self):
        """Constant translation links cells whose raw overlap is insufficient."""
        times = pd.date_range("2024-01-01", periods=2, freq="h")
        # 4x4 blob at t0, shifted by 1 px at t1: raw overlap is 3x3 = 9 px
        mask, tracks = self._make_mask_and_features(
            times,
            [
                ([0], slice(2, 6), slice(2, 6), 1),
                ([1], slice(3, 7), slice(3, 7), 2),
            ],
        )
        result_plain = merge_split_overlap(tracks, mask, minimum_overlap=10)
        result_const = merge_split_overlap(
            tracks,
            mask,
            translate_method="constant",
            velocity_constant=np.array([1 / 3600, 1 / 3600]),
            minimum_overlap=10,
        )
        assert result_plain.track.tolist() == [1, 2]
        assert result_const.track.tolist() == [1, 1]

    def test_translate_method_predict(self):
        """Predictive translation with a constant velocity prior links shifted cells."""
        times = pd.date_range("2024-01-01", periods=2, freq="h")
        mask, tracks = self._make_mask_and_features(
            times,
            [
                ([0], slice(2, 6), slice(2, 6), 1),
                ([1], slice(3, 7), slice(3, 7), 2),
            ],
        )
        result = merge_split_overlap(
            tracks,
            mask,
            translate_method="predict",
            velocity_method="constant",
            velocity_constant=np.array([1 / 3600, 1 / 3600]),
            minimum_overlap=10,
        )
        assert result.track.tolist() == [1, 1]

    def test_memory_with_empty_timestep(self):
        """memory > 0 should bridge a timestep that has no features.

        Regression test: currently raises UnboundLocalError because
        matched_overlaps is not initialised when the first destination
        timestep is empty (the same bug class that was fixed for
        linking_overlap by initialising matched_overlaps = {} before the loop).
        """
        times = pd.date_range("2024-01-01", periods=3, freq="h")
        mask, tracks = self._make_mask_and_features(
            times,
            [
                ([0], slice(2, 6), slice(2, 6), 1),
                # timestep 1 intentionally has no features
                ([2], slice(2, 6), slice(2, 6), 2),
            ],
        )
        result = merge_split_overlap(tracks, mask, memory=1)
        assert result.track.tolist() == [1, 1]

    def test_memory_bridges_gap(self):
        """memory bridges a gap between timesteps; memory=0 does not."""
        times = pd.date_range("2024-01-01", periods=4, freq="h")
        mask, tracks = self._make_mask_and_features(
            times,
            [
                ([0, 1], slice(2, 6), slice(2, 6), 1),
                # timesteps 2 is empty
                ([3], slice(2, 6), slice(2, 6), 2),
            ],
        )
        result_memory = merge_split_overlap(tracks, mask, memory=2)
        result_no_memory = merge_split_overlap(tracks, mask, memory=0)

        assert result_memory.track.tolist() == [1, 1, 1]
        assert result_no_memory.track.tolist() == [1, 1, 2]

    def test_empty_tracks(self):
        """An empty tracks dataframe should return an empty dataframe with track columns.

        Regression test: currently raises TypeError because max_cell is NaN
        for an empty dataframe when building the cell graph.
        """
        times = pd.date_range("2024-01-01", periods=1)
        mask = xr.DataArray(
            np.zeros((1, 10, 10), dtype=int),
            coords={
                "time": times,
                "hdim_1": np.arange(10),
                "hdim_2": np.arange(10),
            },
            dims=["time", "hdim_1", "hdim_2"],
        )
        tracks = pd.DataFrame(
            {
                "feature": [],
                "frame": [],
                "time": pd.to_datetime([]),
                "hdim_1": [],
                "hdim_2": [],
                "cell": [],
            }
        )
        result = merge_split_overlap(tracks, mask)

        assert len(result) == 0
        assert "track" in result.columns
        assert "time_track" in result.columns

    def test_single_timestep(self):
        """With a single timestep each cell gets its own track."""
        times = pd.date_range("2024-01-01", periods=1)
        mask, tracks = self._make_mask_and_features(
            times,
            [
                ([0], slice(2, 6), slice(2, 6), 1),
                ([0], slice(12, 16), slice(12, 16), 2),
            ],
        )
        result = merge_split_overlap(tracks, mask)
        assert result.track.tolist() == [1, 2]

    def test_all_cells_unassigned(self):
        """If all cells are unassigned, all tracks are unassigned and time_track is NaT."""
        times = pd.date_range("2024-01-01", periods=3, freq="h")
        mask, tracks = self._make_mask_and_features(
            times,
            [
                ([0, 1], slice(2, 6), slice(2, 6), -1),
                ([2], slice(2, 6), slice(2, 6), -1),
            ],
        )
        result = merge_split_overlap(tracks, mask)
        assert (result.track == -1).all()
        assert result.time_track.isna().all()

    def test_track_unassigned_cells(self):
        """track_unassigned_cells assigns tracks to unassigned cells that overlap tracked cells."""
        times = pd.date_range("2024-01-01", periods=4, freq="h")
        mask, tracks = self._make_mask_and_features(
            times,
            [
                ([0, 1], slice(2, 6), slice(2, 6), 1),
                ([2, 3], slice(2, 6), slice(2, 6), 1),  # tracked cell 1
                ([0, 1], slice(12, 16), slice(12, 16), -1),  # standalone unassigned
                ([2, 3], slice(3, 7), slice(3, 7), -1),  # unassigned, overlaps cell 1
            ],
        )
        result = merge_split_overlap(tracks, mask, track_unassigned_cells=True)
        by_feature = result.set_index("feature")

        # tracked cell keeps track 1
        assert (by_feature.loc[[1, 2, 3, 4], "track"] == 1).all()
        # standalone unassigned cell stays unassigned
        assert (by_feature.loc[[5, 6], "track"] == -1).all()
        # unassigned cell overlapping the tracked cell joins its track
        assert (by_feature.loc[[7, 8], "track"] == 1).all()
        # the cell column of formerly unassigned features is restored to -1
        assert (by_feature.loc[[5, 6, 7, 8], "cell"] == -1).all()

    def test_cell_number_unassigned_zero(self):
        """Test with cell_number_unassigned=0 (as produced by some tracking methods)."""
        times = pd.date_range("2024-01-01", periods=3, freq="h")
        mask, tracks = self._make_mask_and_features(
            times,
            [
                ([0, 1], slice(2, 6), slice(2, 6), 1),
                ([2], slice(2, 6), slice(2, 6), 1),
                ([0, 1], slice(12, 16), slice(12, 16), 0),
                ([2], slice(12, 16), slice(12, 16), 0),
            ],
        )
        result = merge_split_overlap(tracks, mask, cell_number_unassigned=0)
        by_feature = result.set_index("feature")

        assert (by_feature.loc[[1, 2, 3], "track"] == 1).all()
        assert (by_feature.loc[[4, 5, 6], "track"] == -1).all()

    def test_non_contiguous_cell_ids(self):
        """Cell IDs need not be contiguous for track assignment to work."""
        times = pd.date_range("2024-01-01", periods=3, freq="h")
        mask, tracks = self._make_mask_and_features(
            times,
            [
                ([0, 1], slice(2, 6), slice(2, 6), 50),
                ([2], slice(2, 6), slice(2, 6), 1000),
            ],
        )
        result = merge_split_overlap(tracks, mask)
        assert result.track.tolist() == [1, 1, 1]

    def test_mask_label_without_feature_row(self):
        """Mask labels with no corresponding track row should be ignored."""
        times = pd.date_range("2024-01-01", periods=2, freq="h")
        mask, tracks = self._make_mask_and_features(
            times,
            [
                ([0], slice(2, 6), slice(2, 6), 1),
                ([1], slice(2, 6), slice(2, 6), 2),
            ],
        )
        # label 99 exists in the mask but has no row in tracks
        mask.values[1, 12:16, 12:16] = 99
        result = merge_split_overlap(tracks, mask)
        assert result.track.tolist() == [1, 1]

    def test_input_not_modified(self):
        """Inputs are not modified and the output preserves index and columns."""
        times = pd.date_range("2024-01-01", periods=2, freq="h")
        mask, tracks = self._make_mask_and_features(
            times,
            [
                ([0], slice(2, 6), slice(2, 6), 1),
                ([1], slice(2, 6), slice(2, 6), 1),
            ],
        )
        tracks.index = [100, 101]  # non-default index
        tracks["custom"] = ["a", "b"]
        tracks_copy, mask_copy = tracks.copy(), mask.copy()

        result = merge_split_overlap(tracks, mask)

        pd.testing.assert_frame_equal(tracks, tracks_copy)
        xr.testing.assert_equal(mask, mask_copy)
        # index values are preserved (the output index gets name='index')
        pd.testing.assert_index_equal(
            result.index, tracks.index, check_names=False
        )
        # original columns keep their order, new columns are appended
        assert list(result.columns) == list(tracks.columns) + ["track", "time_track"]

    def test_time_track(self):
        """time_track is the time since the start of each track."""
        times = pd.date_range("2024-01-01", periods=3, freq="h")
        mask, tracks = self._make_mask_and_features(
            times,
            [
                ([0, 1], slice(2, 6), slice(2, 6), 1),
                ([2], slice(2, 6), slice(2, 6), 1),
            ],
        )
        result = merge_split_overlap(tracks, mask)
        assert result.time_track.tolist() == [
            pd.Timedelta(0),
            pd.Timedelta("1h"),
            pd.Timedelta("2h"),
        ]

    def test_missing_cell_column_raises(self):
        """tracks without a 'cell' column should raise an AssertionError."""
        times = pd.date_range("2024-01-01", periods=2, freq="h")
        mask, tracks = self._make_mask_and_features(
            times,
            [([0], slice(2, 6), slice(2, 6), 1), ([1], slice(2, 6), slice(2, 6), 1)],
        )
        tracks = tracks.drop(columns="cell")
        with pytest.raises(AssertionError, match="cell"):
            merge_split_overlap(tracks, mask)

    def test_output_structure(self):
        """Output is a DataFrame with integer track and timedelta time_track."""
        times = pd.date_range("2024-01-01", periods=2, freq="h")
        mask, tracks = self._make_mask_and_features(
            times,
            [([0], slice(2, 6), slice(2, 6), 1), ([1], slice(2, 6), slice(2, 6), 1)],
        )
        result = merge_split_overlap(tracks, mask)

        assert isinstance(result, pd.DataFrame)
        assert pd.api.types.is_integer_dtype(result["track"])
        assert pd.api.types.is_timedelta64_dtype(result["time_track"])

    def test_sample_data_2D_single_cell(self):
        """End-to-end: feature detection + linking_overlap + merge_split_overlap on sample data."""
        data = tobac.testing.make_simple_sample_data_2D(data_type="xarray")
        labels, features = tobac.feature_detection_multithreshold(
            data,
            dxy=1000,
            threshold=7.5,
            position_threshold="weighted_abs",
            return_labels=True,
        )
        tracks = tobac.tracking.linking_overlap(
            features, labels, minimum_relative_overlap=0.75, translate_method=None
        )
        result = merge_split_overlap(tracks, labels)
        assert (result.track == 1).all()

    def test_sample_data_2D_3blobs(self):
        """End-to-end with three separate blobs: three cells map to three tracks."""
        data = tobac.testing.make_sample_data_2D_3blobs(data_type="xarray")
        labels, features = tobac.feature_detection_multithreshold(
            data, 1000, threshold=5, position_threshold="weighted_abs", return_labels=True
        )
        tracks = tobac.tracking.linking_overlap(
            features,
            labels,
            minimum_relative_overlap=0.5,
            translate_method="predict",
            velocity_method="nearest",
        )
        result = merge_split_overlap(tracks, labels)
        assert np.unique(result.track).tolist() == [1, 2, 3]
        # each cell belongs to exactly one track
        assert (result.groupby("cell").track.nunique() == 1).all()
