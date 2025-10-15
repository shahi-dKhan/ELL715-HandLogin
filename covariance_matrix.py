import numpy as np

def covariance_descriptor(F):
    """
    Compute the covariance descriptor for the feature matrix F.

    Parameters:
    F (numpy.ndarray): A 2D array where each column represents a feature vector.

    Returns:
    numpy.ndarray: The covariance matrix of the features.
    """
    # Center the features by subtracting the mean
    # F_centered = F - np.mean(F, axis=1, keepdims=True)
    cov_matrix = np.cov(F)
    utmatrix = cov_matrix[np.triu_indices_from(cov_matrix)]
    return cov_matrix, utmatrix


