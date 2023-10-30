"""Regressor for Nearest Neighbors"""


import numpy as np

from ._base import _get_weights
from ._base import NearestNeighbor


class KNeighborsRegressor(NearestNeighbor):
    """
    Regression based on k-nearest neighbors.

    The target is predicted by local interpolation of the targets
    associated of the nearest neighbors in the training set.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use by default

    weights : {'uniform', 'distance'}, default='uniform'
        Weight function used in prediction. Possible values:
        
        - 'uniform' : uniform weights.
          All points in each neighborhood are weighted equally
        - 'distance': weight points by the inverse of their distance.
          Closer neighbors of a query point will have a greater influence
          than neighbors which are further away.

    p : int, default=2
        Power parameter for the Minkowski distance.
        - p=1 equivalent to the Manhattan distance (L1)
        - p=2 equivalent to the Euclidean distance (L2)

    Returns
    -------
    None

    Examples
    --------
    >>> from ml_from_scratch.neighbors import KNeighborsRegressor
    >>> X = [[0], [1], [2], [3]]
    >>> y = [0, 0, 1, 1]
    >>> neigh = KNeighborsRegressor(n_neighbors=2)
    >>> neigh.fit(X, y)
    >>> print(neigh.predict([[1.5]]))
    [0.5]
    """
    def __init__(
        self,
        n_neighbors=5,
        weights='uniform',
        p=2
    ):
        super().__init__(
            n_neighbors=n_neighbors,
            p=p
        )
        self.weights = weights

    def predict(self, X):
        """
        Predict the target for the provided data.

        Parameters
        ----------
        X : {array-like} of shape (n_queries, n_features)
            Test samples.

        Returns
        -------
        y : {array-like} of shape (n_queries)
            Target values.
        """
        # Convert input to ndarray
        X = np.array(X)

        # Calculate weights
        if self.weights == 'uniform':
            # In that case, we do not need the distances to perform
            # the weighting so we do not compute them.
            neigh_ind = self._kneighbors(X, return_distance = False)
            neigh_dist = None
        else:
            neigh_dist, neigh_ind = self._kneighbors(X)

        weights = _get_weights(neigh_dist, self.weights)
        
        # Get the prediction
        _y = self._y
        if self.weights == 'uniform':
            y_pred = np.mean(_y[neigh_ind], axis=1)
        else:
            num = np.sum(_y[neigh_ind] * weights, axis=1)
            denom = np.sum(weights, axis=1)
            y_pred = num/denom

        return y_pred