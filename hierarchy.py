import numpy as np

def temporal_hierar_cov(feature_sequence, num_levels=3):
    """
    Builds the temporal hierarchy covariance descriptor for a gesture sequence
   

    Args:
        F : np.ndarray (14,N) [x, y, t, z, dE , dW , dN , dS , dN E , dSW , dSE , dN W , dT +, dT −]
    Returns:
        np.ndarray: descriptor - a vector of size 735
        concatenation of 7 upper triangular covariance matrices
    """
    t = feature_sequence[2, :] # the index containing the normalized time information
    descriptors = []
    D = feature_sequence.shape[0]
    ut_indices = np.triu_indices(D) # indices for upper triangular part of DxD matrix
    
    for level in range(1,num_levels+1):
        num_parts = 2**(level-1)
        for i in range(num_parts):
            start_t = i / num_parts
            end_t = (i + 1) / num_parts 
            mask = (t >= start_t) & (t < end_t)
            if np.sum(mask) < 2:
                cov = np.zeros((D, D))
            else:
                cov = np.cov(feature_sequence[:, mask])
            ut = cov[ut_indices]
            descriptors.append(ut)

    descriptor = np.concatenate(descriptors)
    return descriptor
    
    
    
    
    
    
    
    
    # for level in levels:
    #     num_parts = level
    #     part_length = T // num_parts
    #     for i in range(num_parts):
    #         start_idx = i * part_length
    #         end_idx = (i + 1) * part_length if i != num_parts - 1 else T
            
    #         partition = feature_sequence[start_idx:end_idx, :]
            
            
    #         if partition.shape[0] == 1:
    #             cov_matrix = np.zeros((F, F))
    #         else:
    #             cov_matrix = np.cov(partition, rowvar=False)
    #         utmatrix = cov_matrix[np.triu_indices_from(cov_matrix)]
    #         descriptors.append(utmatrix)
            
    # gesture_descriptor = np.concatenate(descriptors)
    # return gesture_descriptor

            
            
            
        