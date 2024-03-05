from statsmodels.stats.multitest import multipletests

#loading p values
from A01_sim_data import p_values, fire_index, nonfire_index
from A01_weighting import weighted_p

#Define function for Benjamini-Hochberg(1995) Procedure 
def bh_method(p_values, alpha=0.05, weights = True):
    """
    Apply Benjamini Hochberg correction to lists of p-values.

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
        # Calculate BH critical value for each index
        bh_values = [p_value * n / (i + 1) for i, (index, p_value) in enumerate(sorted_p_values_with_indices)]

        # Find the largest index where p_value <= BH critical value
        max_significant_index = max(i for i, bh_value in enumerate(bh_values) if bh_value <= alpha)

        # Initialize adjusted p-values with original ordering
        adj_p_values = [min(p_value * n / (i + 1), 1.0) for i, p_value in enumerate(p_values)]

        # Adjust only the significant p-values
        for i in range(max_significant_index + 1):
            index = sorted_p_values_with_indices[i][0]
            adj_p_values[index] = min(adj_p_values[index], alpha)

        return adj_p_values

    m = len(p_values)
    if weights == True:
        p_values = weighted_p
        adj_p = bh_adj_p(p_values,alpha)
        sig_index = [index for index,p in enumerate(adj_p) if p < alpha]
    else:
        adj_p = bh_adj_p(p_values,alpha)
        sig_index = [index for index,p in enumerate(adj_p) if p < alpha]
    return adj_p, sig_index


def bh_method(p_values, alpha=0.05, weights = False):
    from statsmodels.stats.multitest import multipletests
    # Apply BH correction
    adj_p = multipletests(p_values, method='fdr_bh')[1]
    sig_index = [index for index,p in enumerate(adj_p) if p < alpha]
    return adj_p, sig_index

bh_results = bh_method(p_values,alpha=0.05, weights = False)
bh_p, sig_bh_p = bh_results[0], bh_results[1]
bh_p
sig_bh_p
