import numpy as np

def projectData(X, U, K):
    """computes the projection of
    the normalized inputs X into the reduced dimensional space spanned by
    the first K columns of U. It returns the projected examples in Z.
    """

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the projection of the data using only the top K
    #               eigenvectors in U (first K columns).
    #               For the i-th example X(i,:), the projection on to the k-th
    #               eigenvector is given as follows:
    #                    x = X(i, :)'
    #                    projection_k = x' * U(:, k)
    #

    # X: m-by-n
    # U_reduce: n-by-K
    # Z: m-by-K


    # U_reduce contains top K principal components
    U_reduce = U[:, :K] # n-by-K

    # project original data to the reduced dimensional space spanned by the first K columns of U.
    Z = np.matmul(X, U_reduce)


    # =============================================================


    return Z
