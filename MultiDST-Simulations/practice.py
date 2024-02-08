from A01_sim_data import p_values

p_values = [0.01, 0.03, 0.05, 0.07]

#Define function for Benjamini-Hochberg(1995) Procedure 
def bh_method(p_values, alpha=0.05, weights = True):
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
    def bh_adj_p(p_values, alpha):
        # Sort the p-values along with their original indices
        sorted_p_values_with_indices = sorted(enumerate(p_values), key=lambda x: x[1])
        n = len(p_values)
        # BH Adjusted p-values with original ordering
        adj_p_sorted = [[i, ind,min(p_val*n/(i+1),1.0)] for i,(ind,p_val) in enumerate(sorted_p_values_with_indices)]
        bh_adj_p = [sublist[2] for sublist in adj_p_sorted]
        return bh_adj_p

    m = len(p_values)
    if weights == True:
        p_values = weighted_p
        adj_p = bh_adj_p(p_values,alpha)
        sig_index = [index for index,p in enumerate(adj_p) if p < alpha]
    else:
        adj_p = bh_adj_p(p_values,alpha)
        sig_index = [index for index,p in enumerate(adj_p) if p < alpha]

    return adj_p, sig_index



bh_method(p_values, alpha=0.05, weights = False)[1]

import statsmodels.stats.multitest as smt

# p-values from your hypothesis tests
p_values = [0.01, 0.03, 0.05, 0.07]

def bh_adj_p(p_values, alpha):
    # Sort the p-values along with their original indices
    sorted_p_values_with_indices = sorted(enumerate(p_values), key=lambda x: x[1])
    n = len(p_values)
    # BH Adjusted p-values with original ordering
    adj_p_sorted = [[i, ind,min(p_val*n/(i+1),1.0)] for i,(ind,p_val) in enumerate(sorted_p_values_with_indices)]
    bh_adj_p = [sublist[2] for sublist in adj_p_sorted]
    return bh_adj_p

bh_adj_p(p_values,0.05)
sig_index = [index for index,p in enumerate(bh_adj_p) if p < 0.05]
sig_index




# Perform Benjamini-Hochberg correction
rejected, corrected_p_values, _, _ = smt.multipletests(p_values, method='fdr_bh')
corrected_p_values