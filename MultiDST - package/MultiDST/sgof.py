# Importing dependencies
import numpy as np
import pandas as pd
from scipy.stats import chi2

from ..utils.weighting import weighted_p

def sgof_adj(p_values, alpha):
    """
    Perform Sequential Goodness of Fit (SGoF) adjustment on a list of p-values.

    Parameters:
        p_values (array-like): List or array of p-values.
        alpha (float): Threshold value for significance testing.

    Returns:
        tuple: A tuple containing:
            - significant_tests (ndarray): Array of p-values deemed significant after adjustment.
            - significant_indices (ndarray): Array of indices corresponding to significant p-values.
    """
    # Step 01: Sort p-values in ascending order
    sorted_p_values = np.sort(p_values)
    
    # Step 02: Initialize R (number of p-values below threshold)
    R = np.sum(sorted_p_values <= alpha)
    
    # Step 03: Test if observed discoveries deviate significantly
    while R > 0:
        # Perform chi-square test
        observed = R
        expected = len(sorted_p_values) * alpha
        chi_square_statistic = (observed - expected) ** 2 / expected
        p_value_chi2 = 1 - chi2.cdf(chi_square_statistic, df=1)  # df=1 for chi-square

        # Check significance
        if p_value_chi2 < alpha:
            R -= 1
        else:
            break
    
    # Step 04: Extract significant tests and their indices
    significant_tests = sorted_p_values[:R]
    significant_indices = np.where(np.isin(p_values, significant_tests))[0]
    
    return significant_tests, significant_indices


def sgof_test(p_values, alpha=0.05, weights=None):
    """
    Perform Sequential Goodness of Fit (SGoF) test on a list of p-values.

    Parameters:
        p_values (array-like): List or array of p-values.
        alpha (float, optional): Threshold value for significance testing. Default is 0.05.
        weights (array-like or None, optional): Array of weights corresponding to each p-value. 
            If None, no weighting is applied. Default is None.

    Returns:
        tuple: A tuple containing:
            - significant_tests (ndarray): Array of p-values deemed significant after adjustment.
            - significant_indices (ndarray): Array of indices corresponding to significant p-values.
    """
    if weights is not None:
        p_values = weighted_p
    return sgof_adj(p_values, alpha)


control_1 = pd.read_csv('MultiDST/MultiDST - Real-dataset-2-paired/control_1.tsv', sep='\t')
p_values = control_1['5455178010_A.Detection Pval']

sgof_results = sgof_test(p_values,alpha=0.05, weights = None)
sgof_p, sig_sgof_p = sgof_results[0], sgof_results[1]
sgof_p
sig_sgof_p
len(sig_sgof_p)
len(sgof_p)
