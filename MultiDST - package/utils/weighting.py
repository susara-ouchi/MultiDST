#importing dependencies
import numpy as np

def weighted_p_list(p_values, weights=None):
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
import numpy as np
from scipy.optimize import minimize

# def multiweights(p_values, sig_bonf_p, sig_holm_p, sig_sgof_p, sig_bh_p, sig_by_p, sig_q):
#     # Combine all indices into a single list
#     all_indices = np.concatenate((sig_bonf_p, sig_holm_p, sig_sgof_p, sig_bh_p, sig_by_p, sig_q))
#     max_index = np.max(all_indices) + 1
    
#     # Count occurrences of each index across all lists
#     index_counts = np.bincount(all_indices, minlength=max_index)
    
    # # Define objective function to minimize
    # def objective_function(weights):
    #     weighted_sum = np.dot(p_values, weights)
    #     return abs(weighted_sum - 10724)
    
    # # Initial guess for weights
    # initial_guess = np.ones(len(p_values)) / len(p_values)
    
    # # Define constraints: 
    # # 1. Weights must be positive
    # # 2. Elements occurring more frequently should be less than or equal to 1
    # # 3. Elements occurring less frequently can have a higher upper bound (e.g., 2)
    # constraints = [{'type': 'ineq', 'fun': lambda weights: weights},
    #                {'type': 'ineq', 'fun': lambda weights: 1 - weights[:len(index_counts)]},
    #                {'type': 'ineq', 'fun': lambda weights: weights[len(index_counts):] - 1}]
    
    # # Minimize the objective function
    # result = minimize(objective_function, initial_guess, constraints=constraints)

    # # Get optimized weights
    # optimized_weights = result.x
    
    # # Apply optimized weights to p-values
    # weighted_p_values = p_values * optimized_weights
    
    # return weighted_p_values




# Example usage:
p_values = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10])
sig_bonf_p = np.array([0, 1, 3, 5, 6])
sig_holm_p = np.array([0, 2, 3, 6, 7])
sig_sgof_p = np.array([0, 3, 4, 5, 6])
sig_bh_p = np.array([0, 1, 4, 5, 7])
sig_by_p = np.array([0, 2, 4, 6, 7])
sig_q = np.array([0, 1, 2, 4, 6, 7])

weighted_p_values = multiweights(p_values, sig_bonf_p, sig_holm_p, sig_sgof_p, sig_bh_p, sig_by_p, sig_q)
print(weighted_p_values)


import numpy as np
from scipy.optimize import minimize

def multiweights(p_values, sig_bonf_p, sig_holm_p, sig_sgof_p, sig_bh_p, sig_by_p, sig_q):
    # Combine all indices into a single list
    all_indices = np.concatenate((sig_bonf_p, sig_holm_p, sig_sgof_p, sig_bh_p, sig_by_p, sig_q))
    max_index = np.max(all_indices) + 1
    
    # Count occurrences of each index across all lists
    index_counts = np.bincount(all_indices, minlength=max_index)
    
    # Define objective function to minimize
    def objective_function(weights):
        weighted_sum = np.dot(p_values, weights)
        return abs(weighted_sum - 10724)
    
    # Initial guess for weights
    initial_guess = np.ones(len(p_values)) / len(p_values)
    
    # Define constraints: 
    # 1. Weights must be positive
    # 2. Elements occurring more frequently should be less than or equal to 1
    # 3. Elements occurring less frequently can have a higher upper bound (e.g., 2)
    constraints = [{'type': 'ineq', 'fun': lambda weights: weights},
                   {'type': 'ineq', 'fun': lambda weights: 1 - weights[:len(index_counts)]},
                   {'type': 'ineq', 'fun': lambda weights: weights[len(index_counts):] - 1}]
    
    # Minimize the objective function
    result = minimize(objective_function, initial_guess, constraints=constraints)

    # Get optimized weights
    optimized_weights = result.x
    
    # Apply optimized weights to p-values
    weighted_p_values = p_values * optimized_weights
    
    return weighted_p_values





import numpy as np
from scipy.optimize import minimize

def multiweights(p_values, sig_bonf_p, sig_holm_p, sig_sgof_p, sig_bh_p, sig_by_p, sig_q):
    # Combine all indices into a single list
    all_indices = np.concatenate((sig_bonf_p, sig_holm_p, sig_sgof_p, sig_bh_p, sig_by_p, sig_q))
    max_index = np.max(all_indices) + 1
    
    # Count occurrences of each index across all lists
    index_counts = np.bincount(all_indices, minlength=max_index)
    
    # Define objective function to minimize
    def objective_function(weights):
        weighted_sum = np.dot(p_values, weights)
        return abs(weighted_sum - 10724)
    
    # Initial guess for weights
    initial_guess = np.ones(len(p_values)) / len(p_values)
    
    # Define constraints: 
    # 1. Weights must be positive
    # 2. Elements occurring more frequently should be less than or equal to 1
    # 3. Elements occurring less frequently can have a higher upper bound (e.g., 2)
    constraints = [{'type': 'ineq', 'fun': lambda weights: weights},
                   {'type': 'ineq', 'fun': lambda weights: 1 - weights[:len(index_counts)]},
                   {'type': 'ineq', 'fun': lambda weights: weights[len(index_counts):] - 1}]
    
    # Minimize the objective function
    result = minimize(objective_function, initial_guess, constraints=constraints)

    # Get optimized weights
    optimized_weights = result.x
    
    # Yield or store each weighted p-value
    for weight in optimized_weights:
        yield p_values * weight

# Example usage:
p_values = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10])
sig_bonf_p = np.array([0, 1, 3, 5, 6])
sig_holm_p = np.array([0, 2, 3, 6, 7])
sig_sgof_p = np.array([0, 3, 4, 5, 6])
sig_bh_p = np.array([0, 1, 4, 5, 7])
sig_by_p = np.array([0, 2, 4, 6, 7])
sig_q = np.array([0, 1, 2, 4, 6, 7])

# Store weighted p-values in a list
weighted_p_values = list(multiweights(p_values, sig_bonf_p, sig_holm_p, sig_sgof_p, sig_bh_p, sig_by_p, sig_q))

# Print or use the weighted p-values as needed
print(weighted_p_values)
