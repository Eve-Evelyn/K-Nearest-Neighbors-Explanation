"""Base for Nearest Neighbors"""


import numpy as np


def _get_weights(dist, weights):
    """
    Get the weights from an array of distances
    Assume weights have already been validated

    Parameters
    ----------
    dist : ndarray
        The input distances
    
    weights : {'uniform', 'distance'}
        The kind of weighting used

    Returns
    -------
    weights_arr : array of the same shape as 'dist'
        If weights=='uniform', then returns None
    """
    if weights == 'uniform':
        weights_arr = None
    else:
        weights_arr = 1.0/(dist**2)
    
    return weights_arr


class NearestNeighbor:
    """
    Nearest Neighbor base class
    Ref: https://github.com/scikit-learn/scikit-learn/blob/9aaed4987/sklearn/neighbors/_regression.py#L23
    
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
    """
    def __init__(
        self,
        n_neighbors=5,
        p=2
    ):
        self.n_neighbors = n_neighbors
        self.p = p

    def _compute_distance(self, x1, x2):
        """
        Function to compute distance between 2 points according to the Minkowski distance

        Parameters
        ----------
        x1 : {array-like} of (1, n_features)
            First point
        
        x2 : {array-like} of (1, n_features)
            Second point

        Returns
        -------
        dist : float
            Distance between x1-x2
        """
        abs_diff = np.abs(x1-x2)
        sigma_diff = np.sum(np.power(abs_diff, self.p))
        dist = np.power(sigma_diff, 1./self.p)

        return dist

    def _kneighbors(self, X, return_distance=True):
        """
        Find the k-neighbors of a point.
        Return indices of and distances to the neighbors of each point

        Parameters
        ----------
        X : {array-like} shape (n_queries, n_features)
            The query point or points

        return_distance : bool, default=True
            Whether or not to return the distances.

        Returns
        -------
        neigh_dist : ndarray of shape (n_queries, n_neighbors)
            Array representing the lengths to points, 
            only present if return_distance=True
        
        neigh_ind : ndarray of shape (n_queries, n_neighbors)
           Indices of the nearest points in the population 
        """
        # Calculate the distance
        n_queries = X.shape[0]
        n_samples = self._X.shape[0]
        list_dist = np.empty((n_queries, n_samples))
        for i in range(n_queries):
            # Define query point
            X_i = X[i]

            for j in range(n_samples):
                # Find the population-i
                X_j = self._X[j]

                # Find the distance between X_i-X_j
                dist_ij = self._compute_distance(x1 = X_i, 
                                                 x2 = X_j)
            
                # Append
                list_dist[i,j] = dist_ij

        # Sort the distance, ascending order
        # and extract the neighbor data
        neigh_ind = np.argsort(list_dist, axis=1)[:, :self.n_neighbors]

        if return_distance:
            neigh_dist = np.sort(list_dist, axis=1)[:, :self.n_neighbors]

            return neigh_dist, neigh_ind
        else:
            return neigh_ind

    def fit(self, X, y):
        """
        Fit the k-nearest neighbors from the training dataset
        We use Naive method --> only save the training dataset

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            Training data
        y : {array-like} of shape (n_samples, 1)
            Target values.

        Returns
        -------
        self : NearestNeighbors
        """
        self._X = np.array(X)
        self._y = np.array(y)