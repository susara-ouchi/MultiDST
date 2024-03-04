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
        assert len(weights)==len(p_values)
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

