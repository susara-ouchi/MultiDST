import pandas as pd

#loading p values
#from A01_sim_data import p_values, fire_index, nonfire_index
#from A01_weighting import weighted_p
from ..utils.weighting import weighted_p

#Define function for bonferroni procedure

def bonferroni(p_values, alpha=0.05, weights = None):
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

p_values = [0.0005279804659690256, 0.05107595122255753, 0.005380747546894805, 0.008293070676726721, 0.015261930084251897, 0.09399292181095295, 0.04916062506442831, 0.08455877419751781, 0.026622720150619863, 0.060671184302609794, 0.014792473316734833, 0.029279038132892888, 0.039948575984906864, 0.05455860141093238, 0.06495646577203158, 0.01393407242591071, 0.06592036470024257, 0.03370049417508525, 0.08285377432610773, 0.055087308119778314]
bonf_results = bonferroni(p_values,alpha=0.05, weights = True)
bonf_p, sig_bonf_p = bonf_results[0], bonf_results[1]

