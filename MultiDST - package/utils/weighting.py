#importing dependencies
import numpy as np

def weighted_p_list(p_values, weights):
    """
    Generate weighted p-values based on the provided weights.

    Parameters:
        p_values (array-like): Array of original p-values.
        weights (array-like or str): Array of weights corresponding to each p-value.
            If 'random', randomly generate weights for each p-value.

    Returns:
        tuple: A tuple containing:
            - weights (ndarray): Array of weights used for each p-value.
            - weighted_p_values (ndarray): Array of weighted p-values.
    """
    
    # Generate hypothesis weights
    if weights == "random":
        random_numbers = np.random.rand(len(p_values))
        weights = random_numbers / np.sum(np.abs(random_numbers)) 
    else:
        weights = np.array(weights)

    # Combining p-values and weights into a 2D array
    data = np.column_stack((p_values, weights))

    # Separating p-values and weights
    p_values = data[:, 0]
    weights = data[:, 1]

    # Applying weights to p-values
    weighted_p_values = p_values * weights

    # Normalizing weighted p-values
    weighted_p_values /= np.sum(weighted_p_values)

    return weights, weighted_p_values
 
p_values = [0.1, 0.05, 0.6]
weighted_p = weighted_p_list(p_values)[1]

