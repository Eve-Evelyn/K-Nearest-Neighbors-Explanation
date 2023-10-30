"""Classifier with Nearest Neighbors"""


import numpy as np

from ._base import _get_weights
from ._base import NearestNeighbor


class KNeighborsClassifier(NearestNeighbor):
    """
    Classifier implementing the k-nearest neighbors vote.

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
    >>> from ml_from_scratch.neighbors import KNeighborsClassifier
    >>> X = [[0], [1], [2], [3]]
    >>> y = [0, 0, 1, 1]
    >>> neigh = KNeighborsClassifier(n_neighbors=3)
    >>> neigh.fit(X, y)
    >>> print(neigh.predict([[1.1]]))
    [0]
    >>> print(neigh.predict_proba([[0.9]]))
    [[0.66666667 0.33333333]]
    >>> print(neigh.classes_)
    [0 1]
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

    def predict_proba(self, X):
        """
        Predict probability estimates for the test data X

        Parameters
        ----------
        X : {array-like} of shape (n_queries, n_features)
            Test samples

        Returns
        -------
        p : ndarray of shape (n_queries, n_classes)
            The class probabilities of the input samples
        """
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
        neigh_y = _y[neigh_ind]
        n_queries = X.shape[0]

        self.classes_ = np.unique(neigh_y)
        n_classes = len(self.classes_)

        neigh_proba = np.empty((n_queries, n_classes))
        for i in range(n_queries):
            # Extract neighbor output
            neigh_y_i = neigh_y[i]

            # Iterate over class
            for j, class_ in enumerate(self.classes_):
                # Calculate the I(y = class) for every neighbors
                i_class = (neigh_y_i == class_).astype(int)

                # Calculate the class counts
                if self.weights == 'uniform':
                    class_counts_ij = np.sum(i_class)
                else:
                    weights_i = weights[i]
                    class_counts_ij = np.dot(weights_i, i_class)

                # Append
                neigh_proba[i, j] = class_counts_ij

        # Normalize counts --> get probability
        for i in range(n_queries):
            sum_i = np.sum(neigh_proba[i])
            neigh_proba[i] /= sum_i

        return neigh_proba

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
        # Predict neighbor probability
        neigh_proba = self.predict_proba(X)

        # Predict y
        ind_max = np.argmax(neigh_proba, axis=1)
        y_pred = self.classes_[ind_max]

        return y_pred