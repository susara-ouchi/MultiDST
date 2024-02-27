import pandas as pd

#loading p values
from A01_sim_data import p_values, fire_index, nonfire_index
from A01_weighting import weighted_p

#Define function for bonferroni procedure

def bonferroni(p_values, alpha=0.05, weights = False):
    '''
    Apply Bonferroni correction to a vector of p-values.

    Parameters:
        p_values (list or numpy array): Vector of original p-values.
        alpha: Threshold of significance
        weights: Whether or not to use weighted approach

    Returns:
        corrected_p_values(list): Vector of corrected p-values after Bonferroni correction.
    '''

    if weights == True:
        p_values = weighted_p
        adj_p = [min(p * len(p_values), 1.0) for p in p_values]
        sig_index = [index for index,p in enumerate(adj_p) if p < alpha]

    else:
        # Apply Bonferroni correction to each raw p-value
        adj_p = [min(p * len(p_values), 1.0) for p in p_values]
        sig_index = [index for index,p in enumerate(adj_p) if p < alpha]
        
    return adj_p, sig_index

#Overall significance(unweighted)
bonf_test = bonferroni(p_values,alpha=0.05, weights = False)
bonf_p, bonf_sig_index = bonf_test[0], bonf_test[1]

#Overall significance(Weighted)
bonf_test = bonferroni(p_values,alpha=0.05, weights = True)
bonf_w_p, bonf_w_sig_index = bonf_test[0], bonf_test[1]


import numpy as np

def weighted_bonferroni_correction(p_values, weights):
    """
    Apply weighted Bonferroni method for p-value correction.

    Parameters:
    - p_values (array-like): List or array of uncorrected p-values.
    - weights (array-like): List or array of weights for each hypothesis.

    Returns:
    - corrected_p_values (ndarray): Array of corrected p-values.
    """
    # Convert inputs to numpy arrays
    p_values = np.array(p_values)
    weights = np.array(weights)

    # Calculate the sum of weights
    sum_weights = np.sum(weights)

    # Adjust the significance threshold
    alpha_adjusted = 0.05 / sum_weights  # For example, adjust to an overall significance level of 0.05

    # Apply correction using weighted Bonferroni method
    corrected_p_values = np.minimum(p_values * sum_weights, 1)

    return corrected_p_values

# Example usage:
# Assuming p_values is a list or array of uncorrected p-values
# and weights is a list or array of weights for each hypothesis
p_values = [0.02, 0.03, 0.001, 0.005, 0.1]
weights = [1, 2, 1, 3, 1]

# Weighted Bonferroni correction
corrected_p_values_weighted = weighted_bonferroni_correction(p_values, weights)
print("Weighted Bonferroni corrected p-values:", corrected_p_values_weighted)
