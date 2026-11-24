"""Tests for append_tracks_overlap and the save_unfiltered_cell option of
linking_overlap."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import tobac
import tobac.testing
from tobac.tracking.overlap import linking_overlap, append_tracks_overlap


def _make_mask_and_features(times, blobs, grid_size=30):
    """Build a labeled mask and matching features dataframe.

    Parameters
    ----------
    times : list of pd.Timestamp
        Time coordinates of the mask.
    blobs : list of (frame_indices, row_center, col_center, half_size)
        Each blob is labeled with consecutive feature IDs (1-based) at the
        given frame indices, as a square of (2*half_size+1)^2 pixels centered
        on (row_center, col_center).
    grid_size : int
        Size of the square horizontal grid.

    Returns
    -------
    mask : xr.DataArray
    features : pd.DataFrame
    """
    mask_data = np.zeros((len(times), grid_size, grid_size), dtype=int)
    feature_rows = []
    feature_id = 0
    for frame_indices, row, col, half in blobs:
        for frame in frame_indices:
            feature_id += 1
            mask_data[
                frame, row - half : row + half + 1, col - half : col + half + 1
            ] = feature_id
            feature_rows.append(
                {
                    "feature": feature_id,
                    "frame": frame,
                    "time": times[frame],
                    "hdim_1": float(row),
                    "hdim_2": float(col),
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
    features = pd.DataFrame(feature_rows)
    return mask, features


def _moving_blobs_data(n_times=6, grid_size=30):
    """Two blobs moving at constant velocity (+1,+1) and (-1,+1) per timestep."""
    times = pd.date_range("2024-01-01", periods=n_times, freq="h")
    mask_data = np.zeros((n_times, grid_size, grid_size), dtype=int)
    feature_rows = []
    feature_id = 0
    for f in range(n_times):
        for dr, dc in [(1, 1), (-1, 1)]:
            r = 5 + dr * f if dr == 1 else 20 + dr * f
            c = 5 + dc * f
            feature_id += 1
            mask_data[f, r - 1 : r + 2, c - 1 : c + 2] = feature_id
            feature_rows.append(
                {
                    "feature": feature_id,
                    "frame": f,
                    "time": times[f],
                    "hdim_1": float(r),
                    "hdim_2": float(c),
                }
            )
    mask = xr.DataArray(
        mask_data,
        coords={
            "time": times,
            "hdim_1": np.arange(grid_size),
            "hdim_2": np.arange(grid_size),
        },
        dims=["time", "hdim_1", "hdim_2"],
    )
    return mask, pd.DataFrame(feature_rows)


class TestSaveUnfilteredCell:
    def test_save_unfiltered_cell_false_by_default(self):
        times = pd.date_range("2024-01-01", periods=3, freq="h")
        mask, features = _make_mask_and_features(
            times, [([0, 1, 2], 5, 5, 1), ([2], 15, 15, 1)]
        )
        tracks = linking_overlap(features, mask)
        assert "cell_unfiltered" not in tracks.columns

    def test_save_unfiltered_cell(self):
        times = pd.date_range("2024-01-01", periods=3, freq="h")
        # long blob (3 timesteps) and a single-timestep blob (stub for stubs=2)
        mask, features = _make_mask_and_features(
            times, [([0, 1, 2], 5, 5, 1), ([2], 15, 15, 1)]
        )
        tracks = linking_overlap(features, mask, stubs=2, save_unfiltered_cell=True)
        assert "cell_unfiltered" in tracks.columns
        # every feature gets an unfiltered cell number
        assert (tracks.cell_unfiltered != -1).all()
        assert tracks.cell_unfiltered.nunique() == 2
        # the kept cell has the same number in cell and cell_unfiltered
        kept = tracks.cell != -1
        assert (tracks.loc[kept, "cell"] == tracks.loc[kept, "cell_unfiltered"]).all()
        # the stub is unassigned in cell but keeps its identity in cell_unfiltered
        stub = ~kept
        assert (tracks.loc[stub, "cell"] == -1).all()
        assert (tracks.loc[stub, "cell_unfiltered"] > 0).all()
        # filtered tracking is unchanged compared to not saving the column
        tracks_plain = linking_overlap(features, mask, stubs=2)
        assert tobac.testing.check_tracking_identical(tracks, tracks_plain)

    def test_save_unfiltered_cell_no_stubs(self):
        times = pd.date_range("2024-01-01", periods=2, freq="h")
        mask, features = _make_mask_and_features(times, [([0, 1], 5, 5, 1)])
        tracks = linking_overlap(features, mask, save_unfiltered_cell=True)
        assert (tracks.cell == tracks.cell_unfiltered).all()

    def test_save_unfiltered_cell_unmatched_features_get_unique_ids(self):
        # two features that never overlap anything: both unassigned in cell,
        # but with distinct cell_unfiltered numbers
        times = pd.date_range("2024-01-01", periods=2, freq="h")
        mask, features = _make_mask_and_features(
            times, [([0], 5, 5, 1), ([1], 20, 20, 1)]
        )
        tracks = linking_overlap(features, mask, save_unfiltered_cell=True)
        assert (tracks.cell == -1).all()
        assert tracks.cell_unfiltered.nunique() == 2
        assert (tracks.cell_unfiltered > 0).all()


class TestAppendTracksOverlap:
    def test_append_equals_full_linking_2d(self):
        mask, features = _moving_blobs_data(n_times=6)
        params = dict(minimum_relative_overlap=0.3)
        full = linking_overlap(features, mask, **params)

        times = pd.DatetimeIndex(features.time.unique())
        init_mask = mask.isel(time=slice(0, 2))
        init_features = features[features.time <= times[1]]
        init = linking_overlap(init_features, init_mask, **params)

        # append everything at once, passing only the future features/mask;
        # tracks_mask only needs the last max(stubs, memory) = 1 timestep of
        # the original tracks
        appended = append_tracks_overlap(
            init,
            mask.isel(time=slice(0, 2)),
            features[features.time > times[1]],
            mask.isel(time=slice(2, None)),
            **params,
        )
        assert len(appended) == len(features)
        assert tobac.testing.check_tracking_identical(full, appended)

        # duplicate features/timesteps already in tracks are allowed and
        # filtered out: passing the full features and mask gives the same result
        appended_dup = append_tracks_overlap(
            init, mask.isel(time=slice(0, 2)), features, mask, **params
        )
        assert tobac.testing.check_tracking_identical(full, appended_dup)

        # a longer tracks_mask (covering more of the original period) also works
        appended_full_tm = append_tracks_overlap(
            init,
            mask,
            features[features.time > times[1]],
            mask.isel(time=slice(2, None)),
            **params,
        )
        assert tobac.testing.check_tracking_identical(full, appended_full_tm)

    def test_append_one_timestep_at_a_time(self):
        mask, features = _moving_blobs_data(n_times=6)
        params = dict(minimum_relative_overlap=0.3)
        full = linking_overlap(features, mask, **params)

        times = pd.DatetimeIndex(features.time.unique())
        init = linking_overlap(
            features[features.time <= times[1]], mask.isel(time=slice(0, 2)), **params
        )
        curr = init
        for i in range(2, len(times)):
            curr = append_tracks_overlap(
                curr,
                mask.isel(time=slice(max(0, i - 2), i)),
                features[features.time == times[i]],
                mask.isel(time=[i]),
                **params,
            )
        assert tobac.testing.check_tracking_identical(full, curr)

    def test_append_list_of_periods(self):
        mask, features = _moving_blobs_data(n_times=6)
        params = dict(minimum_relative_overlap=0.3)
        full = linking_overlap(features, mask, **params)

        times = pd.DatetimeIndex(features.time.unique())
        init = linking_overlap(
            features[features.time <= times[1]], mask.isel(time=slice(0, 2)), **params
        )
        # append two future periods as lists of (features, mask)
        feats_list = [features[features.time == times[i]] for i in (2, 3)]
        mask_list = [mask.isel(time=[2]), mask.isel(time=[3])]
        appended = append_tracks_overlap(
            init,
            mask.isel(time=slice(0, 2)),
            feats_list,
            mask_list,
            **params,
        )
        assert tobac.testing.check_tracking_identical(
            linking_overlap(
                features[features.time <= times[3]],
                mask.isel(time=slice(0, 4)),
                **params,
            ),
            appended,
        )

    def test_append_predict(self):
        mask, features = _moving_blobs_data(n_times=6)
        params = dict(
            minimum_relative_overlap=0.3,
            translate_method="predict",
            velocity_method="nearest",
        )
        full = linking_overlap(features, mask, **params)
        times = pd.DatetimeIndex(features.time.unique())
        init = linking_overlap(
            features[features.time <= times[1]], mask.isel(time=slice(0, 2)), **params
        )
        appended = append_tracks_overlap(
            init,
            mask.isel(time=slice(0, 2)),
            features[features.time > times[1]],
            mask.isel(time=slice(2, None)),
            **params,
        )
        assert tobac.testing.check_tracking_identical(full, appended)

    def test_append_predict_stubs_requires_unfiltered(self):
        mask, features = _moving_blobs_data(n_times=6)
        params = dict(
            minimum_relative_overlap=0.3,
            stubs=3,
            translate_method="predict",
            velocity_method="mean",
        )
        times = pd.DatetimeIndex(features.time.unique())
        init = linking_overlap(
            features[features.time <= times[1]], mask.isel(time=slice(0, 2)), **params
        )
        # stubs=3 -> tracks_mask must cover the last 3 timesteps of tracks
        # (here all 2 timesteps of the initial period)
        tracks_mask = mask.isel(time=slice(0, 2))
        with pytest.raises(ValueError, match="save_unfiltered_cell=True"):
            append_tracks_overlap(
                init,
                tracks_mask,
                features[features.time > times[1]],
                mask.isel(time=slice(2, None)),
                **params,
            )

        # with the unfiltered cell numbers saved, appending works and matches
        # full tracking
        init_u = linking_overlap(
            features[features.time <= times[1]],
            mask.isel(time=slice(0, 2)),
            save_unfiltered_cell=True,
            **params,
        )
        full_u = linking_overlap(features, mask, save_unfiltered_cell=True, **params)
        appended = append_tracks_overlap(
            init_u,
            tracks_mask,
            features[features.time > times[1]],
            mask.isel(time=slice(2, None)),
            **params,
        )
        assert "cell_unfiltered" in appended.columns
        assert tobac.testing.check_tracking_identical(full_u, appended)
        assert tobac.testing.check_tracking_identical(
            full_u, appended, cell_column="cell_unfiltered"
        )

    def test_append_memory(self):
        # blob present at t=0, missing at t=1, present again at t=2
        times = pd.date_range("2024-01-01", periods=3, freq="h")
        mask, features = _make_mask_and_features(
            times, [([0], 5, 5, 1), ([2], 5, 5, 1)]
        )
        params = dict(memory=1)
        full = linking_overlap(features, mask, **params)
        assert full.cell.tolist() == [1, 1]

        init = linking_overlap(
            features[features.time <= times[0]], mask.isel(time=slice(0, 1)), **params
        )
        appended = append_tracks_overlap(
            init,
            mask.isel(time=[0]),
            features[features.time > times[0]],
            mask.isel(time=slice(1, None)),
            **params,
        )
        assert tobac.testing.check_tracking_identical(full, appended)
        assert appended.cell.tolist() == [1, 1]

    def test_append_memory_stepwise(self):
        """Chained appends one timestep at a time with memory across an empty
        timestep reproduce full tracking."""
        times = pd.date_range("2024-01-01", periods=4, freq="h")
        # blob at t=0 and t=2 only (t=1 and t=3 empty)
        mask, features = _make_mask_and_features(
            times, [([0], 5, 5, 1), ([2], 5, 5, 1)]
        )
        params = dict(memory=1)
        full = linking_overlap(features, mask, **params)
        assert full.cell.tolist() == [1, 1]

        init = linking_overlap(
            features[features.time <= times[0]], mask.isel(time=slice(0, 1)), **params
        )
        curr = init
        for i in range(1, len(times)):
            # tracks_mask must cover the last 2 + memory = 3 timesteps in curr
            curr_times = pd.DatetimeIndex(curr.time.unique())
            tracks_mask = mask.sel(time=list(curr_times[-3:]))
            curr = append_tracks_overlap(
                curr,
                tracks_mask,
                features[features.time == times[i]],
                mask.isel(time=[i]),
                **params,
            )
        assert tobac.testing.check_tracking_identical(full, curr)
        assert curr.cell.tolist() == [1, 1]

    def test_append_max_time_gap(self):
        times = pd.to_datetime(
            ["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 05:00"]
        )
        mask, features = _make_mask_and_features(times, [([0, 1, 2], 5, 5, 1)])
        params = dict(max_time_gap=pd.Timedelta("2h"))
        full = linking_overlap(features, mask, **params)
        assert full.cell.tolist() == [1, 1, -1]

        init = linking_overlap(
            features[features.time <= times[1]], mask.isel(time=slice(0, 2)), **params
        )
        appended = append_tracks_overlap(
            init,
            mask.isel(time=slice(0, 2)),
            features[features.time > times[1]],
            mask.isel(time=slice(2, None)),
            **params,
        )
        assert tobac.testing.check_tracking_identical(full, appended)
        assert appended.cell.tolist() == [1, 1, -1]

    def test_append_errors(self):
        mask, features = _moving_blobs_data(n_times=4)
        times = pd.DatetimeIndex(features.time.unique())
        init = linking_overlap(
            features[features.time <= times[1]], mask.isel(time=slice(0, 2))
        )
        future_features = features[features.time > times[1]]
        future_mask = mask.isel(time=slice(2, None))
        tracks_mask = mask.isel(time=slice(0, 2))
        # tracks without a cell column
        with pytest.raises(ValueError, match="cell"):
            append_tracks_overlap(features, tracks_mask, future_features, future_mask)
        # tracks_mask does not include the last timestep of tracks
        with pytest.raises(ValueError, match="tracks_mask must include"):
            append_tracks_overlap(
                init, mask.isel(time=[2]), future_features, future_mask
            )
        # new features at a timestep not in the mask
        with pytest.raises(ValueError, match="not in the mask"):
            append_tracks_overlap(
                init, tracks_mask, features, mask.isel(time=slice(0, 3))
            )
        # mismatched list lengths
        with pytest.raises(ValueError, match="same length"):
            append_tracks_overlap(init, tracks_mask, [features, features], [mask])
        # exact duplicate rows (same feature and timestep) are allowed and
        # deduplicated, as when chunks overlap at their boundaries
        dup_features = pd.concat([future_features, future_features.iloc[[0]]])
        appended_dup = append_tracks_overlap(
            init, tracks_mask, dup_features, future_mask
        )
        appended = append_tracks_overlap(
            init, tracks_mask, future_features, future_mask
        )
        assert len(appended_dup) == len(appended)

    def test_append_relabels_duplicate_feature_values(self):
        """Feature values duplicated on different timesteps (e.g. reused in the
        new features while already present in tracks) are relabeled, and the
        mask labels remapped, giving the same tracks as full linking."""
        mask, features = _moving_blobs_data(n_times=6)
        params = dict(minimum_relative_overlap=0.3)
        full = linking_overlap(features, mask, **params)

        times = pd.DatetimeIndex(features.time.unique())
        init = linking_overlap(
            features[features.time <= times[1]], mask.isel(time=slice(0, 2)), **params
        )
        # new features with feature ids restarting at 1 (colliding with the
        # tracks' feature ids 1 and 2), and a correspondingly relabeled new mask
        new_feats = features[features.time > times[1]].copy()
        new_mask_full = mask.isel(time=slice(2, None))
        id_map = {fid: fid - 2 for fid in new_feats["feature"]}
        new_feats["feature"] = new_feats["feature"].map(id_map)
        lut = np.arange(int(new_mask_full.values.max()) + 1)
        for old, new in id_map.items():
            lut[old] = new
        new_mask = xr.DataArray(
            lut[new_mask_full.values],
            coords=new_mask_full.coords,
            dims=new_mask_full.dims,
        )

        appended = append_tracks_overlap(
            init, mask.isel(time=slice(0, 2)), new_feats, new_mask, **params
        )
        # all feature values unique in the output; tracks ids preserved
        assert not appended["feature"].duplicated().any()
        assert set(init["feature"]) <= set(appended["feature"])

        # same cell groupings as full tracking (compare by position, since the
        # colliding feature ids were relabeled)
        def cell_sigs(df):
            return {
                frozenset(zip(group["frame"], group["hdim_1"], group["hdim_2"]))
                for _, group in df.groupby("cell")
            }

        assert cell_sigs(appended) == cell_sigs(full)

    def test_append_output_structure(self):
        mask, features = _moving_blobs_data(n_times=4)
        times = pd.DatetimeIndex(features.time.unique())
        init = linking_overlap(
            features[features.time <= times[1]], mask.isel(time=slice(0, 2))
        )
        appended = append_tracks_overlap(
            init,
            mask.isel(time=slice(0, 2)),
            features[features.time > times[1]],
            mask.isel(time=slice(2, None)),
        )
        # no internal columns left behind
        for col in appended.columns:
            assert not col.startswith("_")
        assert "cell" in appended.columns
        assert "time_cell" in appended.columns
        assert len(appended) == len(features)
        # time_cell is set correctly
        for cell_id, group in appended.groupby("cell"):
            if cell_id == -1:
                assert group.time_cell.isna().all()
            else:
                assert group.time_cell.min() == pd.Timedelta(0)

    def test_append_3d(self):
        """3D version of the append test."""
        n_times = 4
        grid = 30
        times = pd.date_range("2024-01-01", periods=n_times, freq="h")
        mask_data = np.zeros((n_times, 5, grid, grid), dtype=int)
        rows = []
        fid = 0
        for f in range(n_times):
            r, c = 5 + f, 5 + f
            fid += 1
            mask_data[f, 1:3, r - 1 : r + 2, c - 1 : c + 2] = fid
            rows.append(
                dict(
                    feature=fid,
                    frame=f,
                    time=times[f],
                    vdim=1.5,
                    hdim_1=float(r),
                    hdim_2=float(c),
                )
            )
        mask = xr.DataArray(
            mask_data,
            coords={
                "time": times,
                "vdim": np.arange(5),
                "hdim_1": np.arange(grid),
                "hdim_2": np.arange(grid),
            },
            dims=["time", "vdim", "hdim_1", "hdim_2"],
        )
        features = pd.DataFrame(rows)
        params = dict(minimum_relative_overlap=0.3)
        full = linking_overlap(features, mask, vertical_coord="vdim", **params)
        times_u = pd.DatetimeIndex(features.time.unique())
        init = linking_overlap(
            features[features.time <= times_u[1]],
            mask.isel(time=slice(0, 2)),
            vertical_coord="vdim",
            **params,
        )
        appended = append_tracks_overlap(
            init,
            mask.isel(time=slice(0, 2)),
            features[features.time > times_u[1]],
            mask.isel(time=slice(2, None)),
            vertical_coord="vdim",
            **params,
        )
        assert tobac.testing.check_tracking_identical(full, appended)

    def test_append_sample_data_3blobs(self):
        """Append tracking on the sample 2D 3-blobs data matches full tracking."""
        data = tobac.testing.make_sample_data_2D_3blobs(data_type="xarray").isel(
            time=slice(0, 20)
        )
        dxy = 1000
        labels, features = tobac.feature_detection_multithreshold(
            data,
            dxy,
            threshold=5,
            position_threshold="weighted_abs",
            return_labels=True,
        )
        params = dict(
            minimum_relative_overlap=0.5,
            translate_method="predict",
            velocity_method="nearest",
        )
        full = linking_overlap(features, labels, **params)

        times = pd.DatetimeIndex(labels.time.values)
        split = 8
        init = linking_overlap(
            features[features.time <= times[split - 1]],
            labels.isel(time=slice(0, split)),
            **params,
        )
        appended = append_tracks_overlap(
            init,
            labels.isel(time=slice(split - 3, split)),
            features[features.time > times[split - 1]],
            labels.isel(time=slice(split, None)),
            **params,
        )
        assert tobac.testing.check_tracking_identical(full, appended)

        # append one timestep at a time
        curr = init
        for i in range(split, len(times)):
            curr = append_tracks_overlap(
                curr,
                labels.isel(time=slice(max(0, i - 2), i)),
                features[features.time == times[i]],
                labels.isel(time=[i]),
                **params,
            )
        assert tobac.testing.check_tracking_identical(full, curr)

    def test_append_sample_data_3blobs_pbc(self):
        """Append tracking with periodic boundaries and prediction matches full
        tracking."""
        data_pbc = tobac.testing.make_sample_data_2D_3blobs(data_type="xarray").roll(
            x=50
        )
        dxy = 1000
        labels, features = tobac.feature_detection_multithreshold(
            data_pbc,
            dxy,
            threshold=5,
            position_threshold="weighted_abs",
            return_labels=True,
            PBC_flag="hdim_2",
        )
        params = dict(
            minimum_relative_overlap=0.5,
            translate_method="predict",
            velocity_method="nearest",
            PBC_flag="hdim_2",
        )
        full = linking_overlap(features, labels, **params)
        assert np.unique(full.cell).tolist() == [1, 2, 3]

        times = pd.DatetimeIndex(labels.time.values)
        split = 50
        init = linking_overlap(
            features[features.time <= times[split - 1]],
            labels.isel(time=slice(0, split)),
            **params,
        )
        appended = append_tracks_overlap(
            init,
            labels.isel(time=slice(split - 3, split)),
            features[features.time > times[split - 1]],
            labels.isel(time=slice(split, None)),
            **params,
        )
        assert tobac.testing.check_tracking_identical(full, appended)
        assert np.unique(appended.cell).tolist() == [1, 2, 3]

    def test_append_pbc_wrapped_blob(self):
        """Append tracking of a blob crossing the periodic boundary in hdim_2."""
        grid = 20
        times = pd.date_range("2024-01-01", periods=3, freq="h")
        mask_data = np.zeros((3, grid, grid), dtype=int)
        rows = []
        centers = [(10, 18), (10, 19), (10, 0)]
        for f, (r, c) in enumerate(centers):
            fid = f + 1
            cols = (c + np.arange(-1, 2)) % grid
            mask_data[f, r, cols] = fid
            rows.append(
                {
                    "feature": fid,
                    "frame": f,
                    "time": times[f],
                    "hdim_1": float(r),
                    "hdim_2": float(c % grid),
                }
            )
        mask = xr.DataArray(
            mask_data,
            coords={
                "time": times,
                "hdim_1": np.arange(grid),
                "hdim_2": np.arange(grid),
            },
            dims=["time", "hdim_1", "hdim_2"],
        )
        features = pd.DataFrame(rows)
        params = dict(PBC_flag="hdim_2", minimum_relative_overlap=0.3)
        full = linking_overlap(features, mask, **params)
        assert full.cell.tolist() == [1, 1, 1]

        init = linking_overlap(
            features[features.time <= times[1]], mask.isel(time=slice(0, 2)), **params
        )
        appended = append_tracks_overlap(
            init,
            mask.isel(time=slice(0, 2)),
            features[features.time > times[1]],
            mask.isel(time=slice(2, None)),
            **params,
        )
        assert appended.cell.tolist() == [1, 1, 1]
        assert tobac.testing.check_tracking_identical(full, appended)

    def test_append_unmatched_boundary_feature(self):
        """A feature at the last timestep of the original tracks that was never
        matched there can still be linked in the new period, as in full tracking."""
        times = pd.date_range("2024-01-01", periods=3, freq="h")
        # blob A at (5,5) for all 3 timesteps; blob B at (15,15) at t=0 and t=2
        # only: B never overlaps anything at t=1, and at t=0 it is unmatched
        mask, features = _make_mask_and_features(
            times, [([0, 1, 2], 5, 5, 1), ([0], 15, 15, 1), ([2], 15, 15, 1)]
        )
        params = dict()
        full = linking_overlap(features, mask, **params)
        init = linking_overlap(
            features[features.time <= times[1]], mask.isel(time=slice(0, 2)), **params
        )
        appended = append_tracks_overlap(
            init,
            mask.isel(time=slice(0, 2)),
            features[features.time > times[1]],
            mask.isel(time=slice(2, None)),
            **params,
        )
        assert tobac.testing.check_tracking_identical(full, appended)

    def test_append_stubs_requires_unfiltered(self):
        """Append with stubs > 1 requires tracks saved with
        save_unfiltered_cell=True; with it, append reproduces full tracking."""
        mask, features = _moving_blobs_data(n_times=6)
        params = dict(minimum_relative_overlap=0.3, stubs=3)
        full = linking_overlap(features, mask, **params)
        times = pd.DatetimeIndex(features.time.unique())
        init = linking_overlap(
            features[features.time <= times[1]], mask.isel(time=slice(0, 2)), **params
        )
        # stubs=3 -> tracks_mask must cover the last 3 timesteps of tracks
        tracks_mask = mask.isel(time=slice(0, 2))
        # without the unfiltered cell numbers, appending raises
        with pytest.raises(ValueError, match="save_unfiltered_cell=True"):
            append_tracks_overlap(
                init,
                tracks_mask,
                features[features.time > times[1]],
                mask.isel(time=slice(2, None)),
                **params,
            )

        # with the unfiltered cell numbers, appending reproduces full tracking
        full_u = linking_overlap(features, mask, save_unfiltered_cell=True, **params)
        init_u = linking_overlap(
            features[features.time <= times[1]],
            mask.isel(time=slice(0, 2)),
            save_unfiltered_cell=True,
            **params,
        )
        appended = append_tracks_overlap(
            init_u,
            tracks_mask,
            features[features.time > times[1]],
            mask.isel(time=slice(2, None)),
            **params,
        )
        assert "cell_unfiltered" in appended.columns
        assert tobac.testing.check_tracking_identical(full_u, appended)
        assert tobac.testing.check_tracking_identical(
            full_u, appended, cell_column="cell_unfiltered"
        )
        # filtered cells match the full run as well
        assert tobac.testing.check_tracking_identical(full, appended)

    def test_append_time_cell_min(self):
        times = pd.date_range("2024-01-01", periods=4, freq="h")
        mask, features = _make_mask_and_features(
            times,
            [([0, 1, 2, 3], 5, 5, 1), ([2, 3], 15, 15, 1)],
        )
        params = dict(time_cell_min=pd.Timedelta("2h"))
        full = linking_overlap(features, mask, **params)
        # the short blob (1 h span) is filtered
        assert full.set_index("feature").cell.to_dict() == {
            1: 1,
            2: 1,
            3: 1,
            4: 1,
            5: -1,
            6: -1,
        }
        init = linking_overlap(
            features[features.time <= times[1]], mask.isel(time=slice(0, 2)), **params
        )
        # time_cell_min filtering also requires the unfiltered cell numbers
        with pytest.raises(ValueError, match="save_unfiltered_cell=True"):
            append_tracks_overlap(
                init,
                mask.isel(time=slice(0, 2)),
                features[features.time > times[1]],
                mask.isel(time=slice(2, None)),
                **params,
            )

        full_u = linking_overlap(features, mask, save_unfiltered_cell=True, **params)
        init_u = linking_overlap(
            features[features.time <= times[1]],
            mask.isel(time=slice(0, 2)),
            save_unfiltered_cell=True,
            **params,
        )
        appended = append_tracks_overlap(
            init_u,
            mask.isel(time=slice(0, 2)),
            features[features.time > times[1]],
            mask.isel(time=slice(2, None)),
            **params,
        )
        assert tobac.testing.check_tracking_identical(full_u, appended)
        assert appended.set_index("feature").cell.to_dict() == {
            1: 1,
            2: 1,
            3: 1,
            4: 1,
            5: -1,
            6: -1,
        }

    def test_append_crossing_blobs_memory_predict(self):
        """Append with memory and predictive tracking on crossing blobs matches
        full tracking (both blobs cross over)."""
        data = tobac.testing.make_sample_data_3D_2crossing_blobs_xarray()
        dxy = 1000
        features = tobac.feature_detection_multithreshold(
            data, dxy, threshold=8, position_threshold="weighted_abs"
        )
        labels, features = tobac.segmentation.watershed_segmentation.segmentation(
            features, data, dxy, threshold=8
        )
        params = dict(memory=1, translate_method="predict", velocity_method="nearest")
        full = linking_overlap(features, labels, **params)
        # both blobs tracked continuously and cross over
        assert np.unique(full.cell).tolist() == [1, 2]

        times = pd.DatetimeIndex(labels.time.values)
        split = 10
        init = linking_overlap(
            features[features.time <= times[split - 1]],
            labels.isel(time=slice(0, split)),
            **params,
        )
        appended = append_tracks_overlap(
            init,
            labels.isel(time=slice(split - 3, split)),
            features[features.time > times[split - 1]],
            labels.isel(time=slice(split, None)),
            **params,
        )
        assert tobac.testing.check_tracking_identical(full, appended)
        assert np.unique(appended.cell).tolist() == [1, 2]


class TestUnlabeledFeatures:
    """Features whose 'feature' value has no corresponding label in the mask
    can never be linked; they must be kept in the output (unassigned) but
    excluded from the linking candidates, also with predictive tracking."""

    def test_linking_overlap_unlabeled_feature(self):
        times = pd.date_range("2024-01-01", periods=3, freq="h")
        mask, features = _make_mask_and_features(times, [([0, 1, 2], 5, 5, 1)])
        # add a feature with no corresponding label in the mask at t=1
        extra = pd.DataFrame(
            [
                {
                    "feature": 99,
                    "frame": 1,
                    "time": times[1],
                    "hdim_1": 20.0,
                    "hdim_2": 20.0,
                }
            ]
        )
        features = pd.concat([features, extra], ignore_index=True)
        for params in [
            dict(),
            dict(translate_method="predict", velocity_method="nearest"),
        ]:
            tracks = linking_overlap(features, mask, **params)
            # the unlabeled feature remains in the output, unassigned
            assert len(tracks) == 4
            cell_by_feature = tracks.set_index("feature").cell
            assert cell_by_feature[99] == -1
            assert cell_by_feature[[1, 2, 3]].tolist() == [1, 1, 1]
            assert pd.isna(tracks.set_index("feature").loc[99, "time_cell"])

    def test_append_unlabeled_features_predict(self):
        """Append with predictive tracking where tracks_mask does not label a
        boundary feature of tracks and the new mask does not label a new
        feature: both are kept unassigned in the output and the result matches
        full linking."""
        mask, features = _moving_blobs_data(n_times=4)
        times = pd.DatetimeIndex(mask.time.values)
        # feature 99 at t=1 (original period) and feature 98 at t=3 (new
        # period) have no corresponding label in the mask
        extra = pd.DataFrame(
            [
                {
                    "feature": 99,
                    "frame": 1,
                    "time": times[1],
                    "hdim_1": 25.0,
                    "hdim_2": 25.0,
                },
                {
                    "feature": 98,
                    "frame": 3,
                    "time": times[3],
                    "hdim_1": 25.0,
                    "hdim_2": 20.0,
                },
            ]
        )
        features = pd.concat([features, extra], ignore_index=True)
        params = dict(translate_method="predict", velocity_method="nearest")

        full = linking_overlap(features, mask, **params)
        cell_by_feature = full.set_index("feature").cell
        assert cell_by_feature[99] == -1
        assert cell_by_feature[98] == -1

        init = linking_overlap(
            features[features.time <= times[1]],
            mask.isel(time=slice(0, 2)),
            **params,
        )
        appended = append_tracks_overlap(
            init,
            mask.isel(time=slice(0, 2)),
            features[features.time > times[1]],
            mask.isel(time=slice(2, None)),
            **params,
        )
        # unlabeled features remain in the output, unassigned
        assert len(appended) == len(full)
        cell_by_feature = appended.set_index("feature").cell
        assert cell_by_feature[99] == -1
        assert cell_by_feature[98] == -1
        assert tobac.testing.check_tracking_identical(full, appended)


def _as_geodataframe(features, geom_col="geometry"):
    """Convert a features dataframe to a GeoDataFrame with point geometries
    at (hdim_1, hdim_2) and a CRS."""
    gpd = pytest.importorskip("geopandas")
    gdf = gpd.GeoDataFrame(
        features.copy(),
        geometry=gpd.points_from_xy(features.hdim_1, features.hdim_2),
        crs="EPSG:4326",
    )
    if geom_col != "geometry":
        gdf = gdf.rename_geometry(geom_col)
    return gdf


def _check_geodataframe(tracks, expected_features, geom_col="geometry"):
    """Check that tracks is a GeoDataFrame maintaining the geometries (in row
    order) and CRS of expected_features."""
    gpd = pytest.importorskip("geopandas")
    assert isinstance(tracks, gpd.GeoDataFrame)
    assert tracks.geometry.name == geom_col
    assert str(tracks.crs) == str(expected_features.crs)
    assert len(tracks) == len(expected_features)
    for g_out, g_exp in zip(tracks.geometry, expected_features.geometry):
        assert g_out.equals(g_exp)


class TestGeoDataFrameInput:
    """If the input features is a GeoDataFrame, the output tracks must also be
    a GeoDataFrame maintaining the input geometries (and CRS)."""

    def test_linking_overlap_geodataframe(self):
        mask, features = _moving_blobs_data(n_times=6)
        gdf = _as_geodataframe(features)
        tracks = linking_overlap(gdf, mask)
        _check_geodataframe(tracks, gdf)
        # same with the unfiltered cell column
        tracks_u = linking_overlap(gdf, mask, stubs=2, save_unfiltered_cell=True)
        _check_geodataframe(tracks_u, gdf)
        assert "cell_unfiltered" in tracks_u.columns

    def test_linking_overlap_plain_dataframe_stays_plain(self):
        mask, features = _moving_blobs_data(n_times=4)
        tracks = linking_overlap(features, mask)
        assert type(tracks) is pd.DataFrame

    def test_linking_overlap_custom_geometry_column(self):
        mask, features = _moving_blobs_data(n_times=4)
        gdf = _as_geodataframe(features, geom_col="geom")
        tracks = linking_overlap(gdf, mask)
        _check_geodataframe(tracks, gdf, geom_col="geom")

    def test_append_tracks_overlap_geodataframe(self):
        mask, features = _moving_blobs_data(n_times=6)
        gdf = _as_geodataframe(features)
        times = pd.DatetimeIndex(gdf.time.unique())
        params = dict(translate_method="predict", velocity_method="nearest")
        full = linking_overlap(gdf, mask, **params)
        _check_geodataframe(full, gdf)
        init = linking_overlap(
            gdf[gdf.time <= times[1]], mask.isel(time=slice(0, 2)), **params
        )
        new = gdf[gdf.time > times[1]]
        appended = append_tracks_overlap(
            init,
            mask.isel(time=slice(0, 2)),
            new,
            mask.isel(time=slice(2, None)),
            **params,
        )
        _check_geodataframe(appended, gdf)
        assert tobac.testing.check_tracking_identical(full, appended)

    def test_append_tracks_overlap_geodataframe_list_of_periods(self):
        mask, features = _moving_blobs_data(n_times=6)
        gdf = _as_geodataframe(features)
        times = pd.DatetimeIndex(gdf.time.unique())
        init = linking_overlap(gdf[gdf.time <= times[1]], mask.isel(time=slice(0, 2)))
        new = gdf[gdf.time > times[1]]
        appended = append_tracks_overlap(
            init,
            mask.isel(time=slice(0, 2)),
            [new[new.time <= times[3]], new[new.time > times[3]]],
            [mask.isel(time=slice(2, 4)), mask.isel(time=slice(4, None))],
        )
        _check_geodataframe(appended, gdf)

    def test_append_tracks_overlap_plain_tracks_geodataframe_features(self):
        """GeoDataFrame new features with plain-DataFrame tracks still give a
        GeoDataFrame output; the new features keep their geometries, the
        original track rows get missing geometry."""
        gpd = pytest.importorskip("geopandas")
        mask, features = _moving_blobs_data(n_times=6)
        gdf = _as_geodataframe(features)
        times = pd.DatetimeIndex(gdf.time.unique())
        plain = pd.DataFrame(gdf[gdf.time <= times[1]].drop(columns="geometry"))
        init = linking_overlap(plain, mask.isel(time=slice(0, 2)))
        assert type(init) is pd.DataFrame
        new = gdf[gdf.time > times[1]]
        appended = append_tracks_overlap(
            init,
            mask.isel(time=slice(0, 2)),
            new,
            mask.isel(time=slice(2, None)),
        )
        assert isinstance(appended, gpd.GeoDataFrame)
        assert str(appended.crs) == str(gdf.crs)
        assert len(appended) == len(gdf)
        # rows of the original (plain) tracks have no geometry
        assert appended.geometry.iloc[: len(init)].isna().all()
        # rows of the new features keep their input geometries
        for g_out, g_exp in zip(appended.geometry.iloc[len(init) :], new.geometry):
            assert g_out.equals(g_exp)
