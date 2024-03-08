import numpy as np
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

#loading p values
from A01_sim_data import p_values, fire_index, nonfire_index
from A01_weighting import weighted_p

#Define function for Simes(1986) Procedure 
def simes(p_values, alpha=0.05, weights = True):
    """
    Apply Simes correction to lists of p-values.

    Parameters:
        p_values (list): List of original p-values.
        alpha: Threshold of significance
        weights: Whether or not to use weighted approach

    Returns:
        adj_p: Simes adjusted -values
        sig_index: significat indices after Holm correction
    """
    def simes_adj(p_values, alpha):
        # Sort the p-values along with their original indices
        sorted_p_values_with_indices = sorted(enumerate(p_values), key=lambda x: x[1])
        n = len(p_values)
        # BH Adjusted p-values with original ordering
        adj_p_sorted = [[i, ind,min(p_val*n/(i+1),1.0)] for i,(ind,p_val) in enumerate(sorted_p_values_with_indices)]
        simes_adj_p = [sublist[2] for sublist in adj_p_sorted]
        return simes_adj_p

    m = len(p_values)
    if weights == True:
        p_values = weighted_p
        adj_p = simes_adj(p_values,alpha)
        sig_index = [index for index,p in enumerate(adj_p) if p < alpha]
    else:
        adj_p = simes_adj(p_values,alpha)
        sig_index = [index for index,p in enumerate(adj_p) if p < alpha]

    return adj_p, sig_index



#Overall significance(unweighted)
simes_test = simes(p_values,alpha=0.05, weights = False)
simes_p, simes_sig_index = simes_test[0], simes_test[1]

#Overall significance(Weighted)
simes_test = simes(p_values,alpha=0.05, weights = True)
simes_w_p, simes_w_sig_index = simes_test[0], simes_test[1]

simes_p

# Step-up
import numpy as np

def simes_correction(p_values):
    """
    Apply Simes' method for p-value correction.

    Parameters:
    - p_values (array-like): List or array of uncorrected p-values.

    Returns:
    - corrected_p_values (ndarray): Array of corrected p-values.
    """
    # Sort p-values in ascending order
    sorted_p_values = np.sort(p_values)

    # Calculate correction factors for each p-value
    correction_factors = np.arange(1, len(sorted_p_values) + 1) / len(sorted_p_values)

    # Apply correction using Simes' method
    corrected_p_values = np.minimum(np.maximum(sorted_p_values * len(sorted_p_values) / correction_factors, 0), 1)

    return corrected_p_values

# Example usage:
# Assuming p_values is a list or array of uncorrected p-values
p_values = [0.02, 0.03, 0.001, 0.005, 0.1]
corrected_p_values = simes_correction(p_values)
print("Original p-values:", p_values)
print("Corrected p-values:", corrected_p_values)