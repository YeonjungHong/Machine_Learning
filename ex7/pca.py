import numpy as np
from scipy import linalg

def pca(X):
    """computes eigenvectors of the covariance matrix of X
      Returns the eigenvectors U, the eigenvalues (on diagonal) in S
    """

    # Useful values
    # (m = number of data points, n = number of dimension)

    m, n = X.shape

    # You need to return the following variables correctly.

    # ====================== YOUR CODE HERE ======================
    # Instructions: You should first compute the covariance matrix. Then, you
    #               should use the "svd" function to compute the eigenvectors
    #               and eigenvalues of the covariance matrix.
    #
    # Note: When computing the covariance matrix, remember to divide by m (the
    #       number of examples).
    #

    # First, compute the covariance matrix

    sigma = np.matmul(X.T, X)/m

    # Next, SVD to compute the eigenvectors and eivenvalues

    # U: n-by-n matrix containing eigenvectors
    # S: n eigenvalues listed in a decreasing order
    # V: identical to U when it comes to eigenvalue decomposition

    U, S, V = linalg.svd(sigma)


# =========================================================================
    return U, S, V