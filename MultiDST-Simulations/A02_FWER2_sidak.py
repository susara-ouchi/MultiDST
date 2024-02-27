import pandas as pd

#loading p values
from A01_sim_data import p_values, fire_index, nonfire_index
from A01_weighting import weighted_p

#Define function for Sidak Procedure 
def sidak(p_values, alpha=0.05, weights = True):
    """
    Apply Šidák correction to lists of p-values.

    Parameters:
        p_values (list): List of original p-values.
        alpha: Threshold of significance
        weights: Whether or not to use weighted approach

    Returns:
        adj_p: Sidak adjusted -values
        sig_index: significat indices after sidak correction
    """
    m = len(p_values)
    if weights == True:
        p_values = weighted_p
        adj_p = [min(1-(1-p)**m,1.0) for p in p_values]
        sig_index = [index for index,p in enumerate(adj_p) if p < alpha]

    else:
        # Šidák correction
        adj_p = [min(1-(1-p)**m,1.0) for p in p_values]
        sig_index = [index for index,p in enumerate(adj_p) if p < alpha]

    return adj_p, sig_index


#Overall significance(unweighted)
sidak_test = sidak(p_values,alpha=0.05, weights = False)
sidak_p, sidak_sig_index = sidak_test[0], sidak_test[1]

#Overall significance(Weighted)
sidak_test = sidak(p_values,alpha=0.05, weights = True)
sidak_w_p, sidak_w_sig_index = sidak_test[0], sidak_test[1]

