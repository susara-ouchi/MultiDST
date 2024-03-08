import numpy as np
from A01_sim_data import p_values

def weighted_p_list(p_values, weights=np.random.rand(len(p_values))):
    # Generate hypothesis weights (Need to find a procedure to assign weights)
    # This gets stores as a numpy nd array.
    # If input is a list use: weight = np.array(weight_list)
    random_numbers = np.random.rand(len(p_values))
    weight = random_numbers / np.sum(np.abs(random_numbers)) 

    # Combining p-values and weights into a 2D array
    data = np.column_stack((p_values, weight))
    # Separating p-values and weights
    p_values = data[:, 0]
    weight = data[:, 1]

    # Applying weights to p-values (e.g., multiply by weights)
    weighted_p_values = p_values * weight
    T = sum(weighted_p_values)
    weighted_p_values = weighted_p_values / T

    p_values = weighted_p_values
    return weight, p_values

weighted_p = weighted_p_list(p_values)[1]

