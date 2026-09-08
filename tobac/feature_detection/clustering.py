

from typing import Literal

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from sklearn.cluster import DBSCAN

from tobac.utils import decorators


@decorators.irispandas_to_xarray()
def feature_detection_clustering(
    hdim_1: xr.DataArray,
    hdim_2: xr.DataArray,
    time: xr.DataArray,
    time_bins,
    max_dist: float,
    min_samples: int,
    coordinate_type: Literal["xy", "latlon"] = "xy",
    return_geometry: bool = False,
    **dbscan_kwargs, 
):
    if coordinate_type == "latlon":
        hdim_1 = np.radians(hdim_1)
        hdim_2 = np.radians(hdim_2)
        max_dist = np.radians(max_dist)
        dbscan_kwargs["algorithm"] = "ball_tree"
        dbscan_kwargs["metric"] = "haversine"
        

    features = pd.concat(
        (
            feature_detection_clustering_timestep(
                hdim_1_group,
                hdim_2_group,
                frame,
                time_group.mid,
                max_dist,
                min_samples,
                return_geometry=return_geometry,
                **dbscan_kwargs,
            )
            for frame, ((time_group, hdim_1_group), (_, hdim_2_group)) in enumerate(
                zip(hdim_1.groupby_bins(time, time_bins), hdim_2.groupby_bins(time, time_bins))
            )
        )
    ).reset_index(drop=True)
    features["feature"] = range(1, len(features)+1)
    if coordinate_type == "latlon":
        features["hdim_1"] = np.degrees(features["hdim_1"])
        features["hdim_2"] = np.degrees(features["hdim_2"])
    return features



# Feature detection based on lat/lon clusters

def feature_detection_clustering_timestep(
    hdim_1: xr.DataArray,
    hdim_2: xr.DataArray,
    frame: int,
    time: np.datetime64,
    max_dist: float,
    min_samples: int,
    return_geometry: bool = False,
    **dbscan_kwargs, 
):
    """
    Cluster point objects at a single timestep using DBSCAN
    """
    dbscan = DBSCAN(eps=max_dist, min_samples=min_samples, **dbscan_kwargs)
    labels = xr.DataArray(
        dbscan.fit(
            np.stack([hdim_1, hdim_2],1)
        ).labels_,
        dims=hdim_1.dims,
        name="cluster"
    )
    wh = labels!=-1
    labels=labels[wh]

    features = pd.DataFrame(
        data=dict(
            frame=frame,
            idx=labels.groupby(labels).first()+1,
            hdim_1=hdim_1[wh].groupby(labels).mean(),
            hdim_2=hdim_2[wh].groupby(labels).mean(),
            num=hdim_1[wh].groupby(labels).count(),
            time=time,
            **{coord:hdim_1[coord][wh].groupby(labels).mean() for coord in hdim_1.coords},
        )
    )

    if return_geometry:
        points_gdf = gpd.GeoDataFrame(
            data=dict(labels=labels),
            geometry=gpd.points_from_xy(
                hdim_2[wh],
                hdim_1[wh]
            )
        )
        features = gpd.GeoDataFrame(
            data=features,
            geometry=points_gdf.dissolve(
                "labels", aggfunc="sum", 
            ).convex_hull.reset_index(drop=True)
        )

    return features

    