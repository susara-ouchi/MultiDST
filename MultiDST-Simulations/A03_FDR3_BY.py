import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import scipy as sp
from scipy import interpolate

#loading p 
from A01_sim_data import p_values, fire_index, nonfire_index
from A01_weighting import weighted_p

# Function to calculate adjusted q_values
def q_adj(p_values, m=None, verbose=True, pi0=None):
    """
    Estimates q-values from p-values

    Args
    =====
    p_values: list
        List of p-values
    m: int, optional
        Number of tests. If not specified m = len(p_values)
    verbose: bool, optional
        Print verbose messages? (default False)
    lowmem: bool, optional
        Use memory-efficient in-place algorithm
    pi0: float or None, optional
        If None, it's estimated as suggested in Storey and Tibshirani, 2003.
        For most GWAS this is not necessary, since pi0 is extremely likely to be 1.
    """
    #assert(all(0 <= p <= 1 for p in p_values)), "p-values should be between 0 and 1"

    if m is None:
        m = float(len(p_values))
    else:
        m *= 1.0

    if len(p_values) < 100 and pi0 is None:
        pi0 = 1.0
    elif pi0 is not None:
        pi0 = pi0
    else:
        pi0 = []
        lam = np.arange(0, 0.90, 0.01)
        counts = np.array([(p_values > i).sum() for i in np.arange(0, 0.9, 0.01)])
        for l in range(len(lam)):
            pi0.append(counts[l]/(m*(1-lam[l])))

        pi0 = np.array(pi0)

        tck = interpolate.splrep(lam, pi0, k=3)
        pi0 = interpolate.splev(lam[-1], tck)
        if verbose:
            print("qvalues pi0=%.3f, estimated proportion of null features " % pi0)

        if pi0 > 1:
            if verbose:
                print("got pi0 > 1 (%.3f) while estimating qvalues, setting it to 1" % pi0)
            pi0 = 1.0

    assert(0 <= pi0 <= 1), "pi0 is not between 0 and 1: %f" % pi0

    p_ordered = sorted(range(len(p_values)), key=lambda k: p_values[k])
    p_values = [p_values[i] for i in p_ordered]
    q_adj = [0] * len(p_values)
    q_adj[-1] = min(pi0 * m / len(p_values) * p_values[-1], 1.0)

    for i in range(len(p_values) - 2, -1, -1):
        q_adj[i] = min(pi0 * m * p_values[i] / (i + 1.0), q_adj[i + 1])

    q_adj_temp = q_adj.copy()
    q_adj = [0] * len(p_values)
    for i, idx in enumerate(p_ordered):
        q_adj[idx] = q_adj_temp[i]

    return q_adj

#Define function for Sidak Procedure 
def q_value(p_values, alpha=0.05, weights = True):
    """
    Apply Storey's Q value correction to lists of p-values.

    Parameters:
        p_values (list): List of original p-values.
        p_value_fire (list): List of original p-values for fire condition.
        p_value_nonfire (list): List of original p-values for non-fire condition.
        alpha (float): Original significance level.

    Returns:
        sidak_p_values (list): List of Šidák-corrected p-values.
        sidak_p_value_fire (list): List of Šidák-corrected p-values for fire condition.
        sidak_p_value_nonfire (list): List of Šidák-corrected p-values for non-fire condition.
    """
    m = len(p_values)
    if weights == True:
        p_values = weighted_p

        adj_p = q_adj(p_values)
        sig_index = [index for index,p in enumerate(adj_p) if p < alpha]
    else:
        # Q value correction
        adj_p = q_adj(p_values)
        sig_index = [index for index,p in enumerate(adj_p) if p < alpha]

    return adj_p, sig_index


#Overall significance(unweighted)
q_val = q_value(p_values,alpha=0.05, weights = False)
storey_q, q_sig_index = q_val[0], q_val[1]

#Overall significance(Weighted)
q_val = q_value(p_values,alpha=0.05, weights = True)
storey_q_w, q_w_sig_index = q_val[0], q_val[1]

storey_q
q_sig_index

import numpy as np

def eBH_correction(p_values):
    """
    Apply eBH (extended Benjamini-Hochberg) method for p-value correction.

    Parameters:
    - p_values (array-like): List or array of uncorrected p-values.

    Returns:
    - corrected_p_values (ndarray): Array of corrected p-values.
    """
    # Sort p-values in ascending order
    sorted_p_values = np.sort(p_values)

    # Calculate the critical value lambda
    n = len(sorted_p_values)
    lambda_val = 1 / n * np.sum(1 / np.arange(1, n + 1))

    # Apply correction using eBH method
    corrected_p_values = np.minimum(sorted_p_values * n / (np.arange(1, n + 1) * lambda_val), 1)

    return corrected_p_values

def BY_correction(p_values, m=None):
    """
    Apply BY (Benjamini-Yekutieli) method for p-value correction.

    Parameters:
    - p_values (array-like): List or array of uncorrected p-values.
    - m (int or None): Total number of hypotheses. If None, it's set to the number of p-values.

    Returns:
    - corrected_p_values (ndarray): Array of corrected p-values.
    """
    # Sort p-values in ascending order
    sorted_p_values = np.sort(p_values)

    # If total number of hypotheses is not provided, set it to the number of p-values
    if m is None:
        m = len(sorted_p_values)

    # Calculate the critical value lambda
    lambda_val = np.sum(1 / np.arange(1, m + 1))

    # Apply correction using BY method
    corrected_p_values = np.minimum(sorted_p_values * m / (np.arange(1, m + 1) * lambda_val), 1)

    return corrected_p_values

# Example usage:
# Assuming p_values is a list or array of uncorrected p-values
p_values = [0.02, 0.03, 0.001, 0.005, 0.1]

# eBH correction
corrected_p_values_eBH = eBH_correction(p_values)
print("eBH corrected p-values:", corrected_p_values_eBH)

# BY correction
corrected_p_values_BY = BY_correction(p_values)
print("BY corrected p-values:", corrected_p_values_BY)


import numpy as np

def benjamini_yekutieli(p_values, alpha):
    """
    Apply Benjamini-Yekutieli procedure for FDR control.

    Parameters:
    - p_values (array-like): List or array of uncorrected p-values.
    - alpha (float): Desired significance level (e.g., 0.05).

    Returns:
    - adjusted_p_values (ndarray): Adjusted p-values based on the Benjamini-Yekutieli procedure.
    """
    # Convert p_values to numpy array
    p_values = np.array(p_values)

    # Sort p-values in ascending order
    sorted_indices = np.argsort(p_values)
    sorted_p_values = p_values[sorted_indices]

    # Calculate critical value (c)
    m = len(p_values)
    c = alpha / np.sum(1 / np.arange(1, m + 1))

    # Initialize array to store adjusted p-values
    adjusted_p_values = np.zeros_like(sorted_p_values)

    # Test hypotheses sequentially
    for i, p_value in enumerate(sorted_p_values):
        # Calculate adjusted significance level (alpha_adj)
        alpha_adj = c * (1 / (i + 1))

        # If p-value is less than or equal to adjusted significance level
        if p_value <= alpha_adj:
            # Reject null hypothesis
            adjusted_p_values[i] = min(1, p_value * m / (i + 1))
        else:
            # Do not reject null hypothesis
            adjusted_p_values[i] = min(1, alpha_adj)

    # Restore original order
    adjusted_p_values = adjusted_p_values[np.argsort(sorted_indices)]

    return adjusted_p_values

# Example usage:
# p_values = [0.01, 0.03, 0.05, 0.1, 0.2]
# alpha = 0.05
# adjusted_p_values = benjamini_yekutieli(p_values, alpha)
# print(adjusted_p_values)


# Import necessary libraries
import numpy as np

def benjamini_yekutieli(p_values):
    # Number of comparisons
    m = len(p_values)

    # Calculate the Benjamini-Yekutieli adjustment factor
    q = np.sum(1 / np.arange(1, m + 1))

    # Sort the p-values in ascending order
    sorted_indices = np.argsort(p_values)
    sorted_p_values = p_values[sorted_indices]

    # Calculate the adjusted p-values
    adjusted_p_values = np.minimum(1, q * m / np.arange(m, 0, -1) * sorted_p_values)

    # Ensure that the adjusted p-values are monotonically increasing
    adjusted_p_values = np.maximum.accumulate(adjusted_p_values[::-1])[::-1]

    # Return the adjusted p-values in their original order
    return adjusted_p_values[np.argsort(sorted_indices)]

# Define your p-values
p_values = np.array([0.01, 0.03, 0.05, 0.1, 0.2])  # insert your p-values here

# Adjust the p-values using the Benjamini-Yekutieli procedure
adjusted_p_values = benjamini_yekutieli(p_values)

print('Adjusted p-values:', adjusted_p_values)



from statsmodels.stats.multitest import fdrcorrection

# Define your p-values
p_values = [0.01, 0.03, 0.05, 0.1, 0.2]

# Perform the Benjamini-Yekutieli procedure
reject, adjusted_p_values = fdrcorrection(p_values, method='n')

print('Adjusted p-values:', adjusted_p_values)
