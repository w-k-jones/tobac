from typing import Literal, Union

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from tobac.utils.periodic_boundaries import build_distance_function


class FeatureBallTree(BallTree):
    """A child class of scikit-learn's BallTree that handles feature dataframe input
    and periodic boundary conditions (PBC).
    """

    def __init__(
        self,
        features: pd.DataFrame,
        PBC_flag: Union[None, Literal["none", "hdim_1", "hdim_2", "both"]] = None,
        min_h1: int = 0,
        max_h1: int = 0,
        min_h2: int = 0,
        max_h2: int = 0,
        **kwargs,
    ) -> None:
        """Initialize a FeatureBallTree from a features dataframe.

        Parameters
        ----------
        features : pd.DataFrame
            Features dataframe with columns 'hdim_1', 'hdim_2', and optionally 'vdim'
            for 2D and 3D data respectively.
        PBC_flag : {'none', 'hdim_1', 'hdim_2', 'both'}, optional
            Specification of which dimensions have periodic boundary conditions.
            Default is None (no periodic boundaries).
        min_h1 : int, optional
            Minimum value of first horizontal dimension for PBC. Default is 0.
        max_h1 : int, optional
            Maximum value of first horizontal dimension for PBC. Default is 0.
        min_h2 : int, optional
            Minimum value of second horizontal dimension for PBC. Default is 0.
        max_h2 : int, optional
            Maximum value of second horizontal dimension for PBC. Default is 0.
        **kwargs
            Additional keyword arguments passed to BallTree constructor.

        """
        self.is_3D = "vdim" in features.columns
        self.index = features.index.values
        if PBC_flag in ["hdim_1", "hdim_2", "both"]:
            kwargs["metric"] = "pyfunc"
            kwargs["func"] = build_distance_function(
                min_h1, max_h1, min_h2, max_h2, PBC_flag, self.is_3D
            )
        super().__init__(self._get_feature_locations(features), **kwargs)

    def _get_feature_locations(self, features: pd.DataFrame) -> np.ndarray:
        """Extract coordinate array from features dataframe.

        Parameters
        ----------
        features : pd.DataFrame
            Features dataframe with position columns.

        Returns
        -------
        np.ndarray
            Coordinate array of shape (n_features, n_dims) containing either
            ['hdim_1', 'hdim_2'] for 2D or ['vdim', 'hdim_1', 'hdim_2'] for 3D data.

        Raises
        ------
        AssertionError
            If dimensionality of query features does not match the tree dimensionality.

        """
        assert (
            "vdim" in features.columns
        ) == self.is_3D, "Query features must match dimensionality of original features"
        return (
            features[["vdim", "hdim_1", "hdim_2"]].to_numpy()
            if self.is_3D
            else features[["hdim_1", "hdim_2"]].to_numpy()
        )

    def query(self, features: pd.DataFrame, *args, **kwargs) -> np.ndarray:
        """Query the tree for nearest neighbors.

        Wraps the parent BallTree.query method to handle feature dataframe input
        and return results mapped to original feature indices.

        Parameters
        ----------
        features : pd.DataFrame
            Query features dataframe with position columns matching tree dimensionality.
        *args
            Positional arguments passed to BallTree.query.
        **kwargs
            Keyword arguments passed to BallTree.query.

        Returns
        -------
        indices : np.ndarray
            Feature indices of nearest neighbors, shape (n_queries,) or
            (n_queries, k) depending on input.
        distances : np.ndarray, optional
            Distances to nearest neighbors if return_distance=True in kwargs.

        """
        query_result = super().query(
            self._get_feature_locations(features), *args, **kwargs
        )
        if isinstance(query_result, tuple):
            return (
                self.index[query_result[1]],
                query_result[0],
            )  # flip around results to match query_radius
        return self.index[query_result]

    def query_radius(self, features: pd.DataFrame, *args, **kwargs) -> np.ndarray:
        """Query the tree for neighbors within a specified radius.

        Wraps the parent BallTree.query_radius method to handle feature dataframe
        input and return results mapped to original feature indices.

        Parameters
        ----------
        features : pd.DataFrame
            Query features dataframe with position columns matching tree dimensionality.
        *args
            Positional arguments passed to BallTree.query_radius.
        **kwargs
            Keyword arguments passed to BallTree.query_radius.

        Returns
        -------
        neighbors : np.ndarray
            Array of feature indices of neighbors within radius for each query point.
            Shape is (n_queries,) with dtype=object containing variable-length arrays.
        distances : np.ndarray, optional
            Distances to neighbors if return_distance=True in kwargs.

        """
        query_result = super().query_radius(
            self._get_feature_locations(features), *args, **kwargs
        )
        if isinstance(query_result, tuple):
            return (
                np.array([self.index[inds] for inds in query_result[0]], dtype=object),
                query_result[1],
            )
        return np.array([self.index[inds] for inds in query_result], dtype=object)
