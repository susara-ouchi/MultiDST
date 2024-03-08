import pandas as pd

#loading p values
from A01_sim_data import p_values, fire_index, nonfire_index
from A01_weighting import weighted_p

#Define function for Holm Procedure 
def holm(p_values, alpha=0.05, weights = True):
    """
    Apply Holm correction to lists of p-values.

    Parameters:
        p_values (list): List of original p-values.
        alpha: Threshold of significance
        weights: Whether or not to use weighted approach

    Returns:
        adj_p: Holm adjusted -values
        sig_index: significat indices after Holm correction
    """
    def holm_adj_p(p_values):
        # Sort the p-values in ascending order and keep track of their original indices
        sorted_p_values_with_indices = sorted(enumerate(p_values), key=lambda x: x[1])
        # Calculate the adjusted significance level for each p-value
        n = len(p_values)
        adjusted_alpha_values = [sorted_p_values_with_indices[i][1] / (n - i) for i in range(n)]
        # Apply Holm correction
        adj_p_indices = [(sorted_p_values_with_indices[i][0], min(adjusted_alpha_values[i:])) for i in range(n)]
        # Reversing the sort the corrected p-values based on their original indices
        holm_adj_p = [p for i, p in sorted(adj_p_indices)]
        return holm_adj_p

    m = len(p_values)
    if weights == True:
        p_values = weighted_p
        adj_p = holm_adj_p(p_values)
        sig_index = [index for index,p in enumerate(adj_p) if p < alpha]

    else:
        adj_p = holm_adj_p(p_values)
        sig_index = [index for index,p in enumerate(adj_p) if p < alpha]

    return adj_p, sig_index


#Overall significance(unweighted)
holm_test = holm(p_values,alpha=0.05, weights = False)
holm_p, holm_sig_index = holm_test[0], holm_test[1]

#Overall significance(Weighted)
holm_test = holm(p_values,alpha=0.05, weights = True)
holm_w_p, holm_w_sig_index = holm_test[0], holm_test[1]


#step down
import numpy as np

def holm_bonferroni_correction(p_values):
    """
    Apply Holm-Bonferroni method for p-value correction.

    Parameters:
    - p_values (array-like): List or array of uncorrected p-values.

    Returns:
    - corrected_p_values (ndarray): Array of corrected p-values.
    """
    # Sort p-values in ascending order
    sorted_indices = np.argsort(p_values)
    sorted_p_values = np.array(p_values)[sorted_indices]

    # Calculate correction factors
    n = len(sorted_p_values)
    correction_factors = np.arange(n, 0, -1)

    # Apply correction using Holm-Bonferroni method
    corrected_p_values = np.minimum(sorted_p_values * correction_factors, 1)

    # Restore original order
    corrected_p_values = np.zeros_like(sorted_p_values)
    corrected_p_values[sorted_indices] = corrected_p_values

    return corrected_p_values

# Example usage:
# Assuming p_values is a list or array of uncorrected p-values
p_values = [0.02, 0.03, 0.001, 0.005, 0.1]

# Holm-Bonferroni correction
corrected_p_values_hb = holm_bonferroni_correction(p_values)
print("Holm-Bonferroni corrected p-values:", corrected_p_values_hb)
