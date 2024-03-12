#importing dependencies
import numpy as np

def weighted_p_list(p_values,l0,l1,l2,l3,l4,l5,l6, weights=None, max_weight=1.5, min_weight = 0.5):
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
    if weights == "multi":
        # Optimize weights
        opti_weights = optimize_weights(p_values, l0, l1, l2, l3, l4, l5, l6, k2=max_weight, k1 = min_weight)
        weights = opti_weights[0]

        lists = [l0, l1, l2, l3, l4, l5, l6]  # List of all lists

        # Initialize a list to store the weighted values
        weighted_values = []

        # Iterate over each list and apply the corresponding weight to its elements
        for i, sublist in enumerate(lists):
            for index in sublist:
                weighted_values.append(weights[i] * p_values[index])

        # Sort the weighted values based on their original indices
        weighted_values_sorted = [val for _, val in sorted(zip(sum(lists, []), weighted_values))]
        # setting too large values to 1
        weighted_values_sorted = [val if val <= 1.0 else 1.0 for val in weighted_values_sorted]


        weighted_p_values = weighted_values_sorted
    
    else:
        weights = np.array(weights)

    # # Combining p-values and weights into a 2D array
    # data = np.column_stack((p_values, weights))

    # # Separating p-values and weights
    # p_values = data[:, 0]
    # weights = data[:, 1]

    # # Applying weights to p-values
    # weighted_p_values = p_values * weights

    # # Normalizing weighted p-values
    # weighted_p_values /= np.sum(weighted_p_values)

    return weights, weighted_p_values


p_values = np.random.random(22)

p_values
l0 = [1, 2,3]  # Example list l0
l1 = [4, 5, 6]  # Example list l1
l2 = []  # Example list l2
l3 = [10, 11, 12,7,8,9]  # Example list l3
l4 = [13, 14, 15]  # Example list l4
l5 = [16, 17, 18]  # Example list l5
l6 = [19, 20, 21,0]  # Example list l6

weighting = weighted_p_list(p_values, l0, l1, l2, l3, l4, l5, l6, weights="multi")
weighting[0]
weighting[1]

########################################### Optimal Weights using Multi weighting ########################################3

from scipy.optimize import minimize
import numpy as np

def optimize_weights(p_values, l0, l1, l2, l3, l4, l5, l6, k2=1.5, k1=0.5):
    m = len(p_values)

    # Assert that the length of p_values is equal to the sum of lengths of l0 to l6
    assert len(p_values) == len(l0) + len(l1) + len(l2) + len(l3) + len(l4) + len(l5) + len(l6), "Length of p_values should be equal to the sum of lengths of l0 to l6"

    # Define the objective function to be minimized
    def objective_function(w):
        return sum(w[i] * p_values[i] for i in range(len(w)))
    
    print("no issue till obj")

        # Define a function to calculate the product of the weighted p-values
    def weighted_product(w):
        return np.prod([w[i] * p_values[i] for i in range(len(w))])

    # Additional constraint to ensure the product is close to m
    def product_constraint(w):
        return weighted_product(w) - m


    # Define the constraints
    def constraint(w):
        tolerance = 0.1  # Small tolerance value to avoid numerical precision issues
        # return [2-w[0]-tolerance,  w[0] - w[1] - tolerance, w[1]-1.5 - tolerance, w[1] - w[2] - tolerance, 
        #         w[2] - w[3] - tolerance, 1 - w[4] - tolerance, w[4] - w[5] - tolerance, 
        #         w[5] - w[6] - tolerance]
        return [w[0] - w[1], w[1] - w[2], w[2] - w[3], 1-w[4]-tolerance, w[4] - w[5]-tolerance, w[5] - w[6]-tolerance]

    print("no issue till constraints")

    # Define the equality constraint for w3=1
    def w3_constraint(w):
        return w[3]-1

    # Define the equality constraint
    def equality_constraint(w):
        return sum(w[i] * p_values[i] for i in range(len(w))) - len(p_values)

    # Define the initial guess
    initial_guess = [3, 2, 1, 0, -1, -2, -3]  

    print("no issue till guess")

    # Define the bounds for each variable
    bounds = [(0.001, None)] * len(initial_guess)

    # Define the optimization problem
    optimization_result = minimize(objective_function, initial_guess, constraints=[{'type': 'ineq', 'fun': constraint},
                                                                                      {'type': 'eq', 'fun': w3_constraint},
                                                                                      {'type': 'eq', 'fun': equality_constraint},
                                                                                      {'type': 'eq', 'fun': product_constraint}],
                                    bounds=bounds, method='SLSQP')
    print("no issue at weights")
    # Extract the optimal values
    optimal_weights1 = optimization_result.x
    optimal_weights = optimal_weights1

    # Calculate the weighted list
    weighted_list = [optimal_weights[i] * p_values[index] for i, sublist in enumerate([l0, l1, l2, l3, l4, l5, l6]) for index in sublist]
    print("All G")

    # Apply transformation to the optimal weights
    transformed_weights = optimal_weights.copy()

    # # Apply transformation to negative values
    neg_indices = transformed_weights <= 1
    transformed_weights[neg_indices] = k1 / (1 + np.exp(-transformed_weights[neg_indices]))

    # Scale down positive values proportionally
    pos_indices = transformed_weights > 1
    # pos_values = transformed_weights[pos_indices]
    # max_allowed_value = max_weight
    # scaled_positive_values = pos_values * (max_allowed_value / np.max(pos_values))
    # scaled_positive_values = np.clip(scaled_positive_values, 1, max_allowed_value)
    transformed_weights[pos_indices] = k2 / (1 + np.exp(-transformed_weights[pos_indices]))

    # # Set zero weights to 1
    # zero_indices = transformed_weights == 0
    # transformed_weights[zero_indices] = 1

    optimal_weights = transformed_weights.tolist()
    # Ensure that the weighted list elements do not exceed 1
    #weighted_list = [min(value, 1.0) for value in weighted_list]
    return optimal_weights, weighted_list, optimal_weights1



# Define p_values and lists l0 to l6
p_values = np.random.random(14)
l0 = [0, 1]
l1 = [2, 3, 4]
l2 = [5]
l3 = [6, 7]
l4 = [8, 9]
l5 = [10, 11, 12]
l6 = [13]

# Call the optimize_weights function
optimal_weights, weighted_list, optiw = optimize_weights(p_values, l0, l1, l2, l3, l4, l5, l6)

# Print the results
print("Optimal Weights:", optimal_weights)
print("Weighted List:", weighted_list)





# Example usage:
p_values = np.random.random(21)  # Example p_values
l0 = [1, 2, 3]  # Example list l0
l1 = [4, 5, 6]  # Example list l1
l2 = [7, 8, 9]  # Example list l2
l3 = [10, 11, 12]  # Example list l3
l4 = [13, 14, 15]  # Example list l4
l5 = [16,17,18]  # Example list l5
l6 = [19,20,0]  # Example list l6

# Optimize weights and get the weighted list
weighted_list = optimize_weights(p_values, l0, l1, l2, l3, l4, l5, l6)[0]

# Print the weighted list
print("Weighted list:", weighted_list)


# Set the seed
import random

random.seed(42)
p_values = [random.random() for _ in range(21)]
print(p_values)
  # Example p_values
l0 = [1, 2, 3]  # Example list l0
l1 = [4, 5, 6]  # Example list l1
l2 = [7, 8, 9]  # Example list l2
l3 = [10, 11, 12]  # Example list l3
l4 = [13, 14, 15]  # Example list l4
l5 = [16,17,18]  # Example list l5
l6 = [19,20]  # Example list l6

[p_values[index] for i, sublist in enumerate([l0, l1, l2, l3, l4, l5, l6]) for index in sublist]

# p_values_selected = [p_values[index] for sublist in [l0, l1, l2, l3, l4, l5, l6] for index in sublist]
[weighted_list[0][i] * p_values[index] for i, sublist in enumerate([l0, l1, l2, l3, l4, l5, l6]) for index in sublist]

weighted_list[0][1]*p_values[1]



# Intuition

p_values = [i+1 for i in range(14)]
print(p_values)
  # Example p_values
l0 = [0]  # Example list l0
l1 = [1]  # Example list l1
l2 = [2,13]  # Example list l2
l3 = [3]  # Example list l3
l4 = [4,11,12]  # Example list l4
l5 = [5]  # Example list l5
l6 = [6,7,8,9,10]  # Example list l6

[weighted_list[0][i]*p_values[index] for i, sublist in enumerate([l0, l1, l2, l3, l4, l5, l6]) for index in sublist]
[weighted_list[0][i] * p_values[index] for i, sublist in enumerate([l0, l1, l2, l3, l4, l5, l6]) for index in sublist]
weighted_list[0]





import pandas as pd

# Opening the file in read mode
with open('MultiDST/MultiDST - Real Dataset/experimental_pvalues.txt', 'r') as file:
    # Read the entire content of the file
    content = file.read()

# Split the content into lines
lines = content.splitlines()

# Split each line into columns (assuming a space delimiter)
table_data = [line.split() for line in lines]
colnames = lines[0].split()

# Create a DataFrame using pandas
df = pd.DataFrame(table_data[1:], columns=colnames)

### Weighted MPRA
random_ind = df.loc[df['Type'] == "Random"].index

P_MPRA = df['p.value.MPRA']
P_STARR = df['p.value.STARR']

###################################### For the full dataset #################################
from functions import multi_DST

import matplotlib.pyplot as plt
p_valuesMPRA = P_MPRA.values.astype(float)
p_valuesSTARR = P_STARR.values.astype(float)

p_values = p_valuesSTARR

results = multi_DST(p_values, alpha=0.05, weights=False)
print(results)   

sig_uncorrected = results["Uncorrected"]
sig_bonf_p = results["Bonferroni"]
sig_holm_p = results["Holm"]
sig_sgof_p = results["SGoF"]
sig_bh_p = results["BH"]
sig_by_p = results["BY"]
sig_q = results["Q-value"]
pi0_est = results["pi0 estimate"]



from utils.common_indices import common_indices

l0,l1,l2,l3,l4,l5,l6,l7 = common_indices(p_values, sig_bonf_p, sig_holm_p, sig_sgof_p, sig_bh_p, sig_by_p, sig_q)

len(l0)
len(l1)
len(l2)
len(l3)
len(l4)
len(l5)
len(l6)

optimize_weights(p_values, l0, l1, l2, l3, l4, l5, l6,k1=1,k2=2)[0]