import numpy as np
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

#loading p values
from A01_sim_data import p_values, fire_index, nonfire_index
from A01_weighting import weighted_p

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


#Overall significance(unweighted)
bh_test = bh_method(p_values,alpha=0.05, weights = False)
bh_p, bh_sig_index = bh_test[0], bh_test[1]

#Overall significance(Weighted)
bh_test = bh_method(p_values,alpha=0.05, weights = True)
bh_w_p, bh_w_sig_index = bh_test[0], bh_test[1]

bh_p



def sim_eval(p_values, fire_index, nonfire_index, adj_p, sig_index, threshold =0.05):
    import pandas as pd
    significant_p = [p_values[index] for index in sig_index]
    significant_p_fire = [adj_p[index] for index in fire_index if adj_p[index] < threshold]
    significant_p_nonfire = [adj_p[index] for index in
     nonfire_index if adj_p[index] < threshold]
 
    # Confusion Matrix
    TP = len(significant_p_fire)
    FP = len(significant_p_nonfire)
    TN = len(nonfire_index) - FP
    FN = len(fire_index) - TP

    data = {
        'Actual Positive': [TP, FN],
        'Actual Negative': [FP, TN],
    }
    confusion_matrix = pd.DataFrame(data, index=['Predicted Positive', 'Predicted Negative'])
    confusion_matrix

    sig_p= len(significant_p)
    power = TP/(TP+FN)
    power
        
    #To find which genes are significant
    TP_index = [index for index in fire_index if adj_p[index] < threshold]
    FN_index = [index for index in fire_index if adj_p[index] >= threshold]
    FP_index = [index for index in nonfire_index if adj_p[index] < threshold]
    TN_index = [index for index in nonfire_index if adj_p[index] >= threshold]
    return power, confusion_matrix,TP_index