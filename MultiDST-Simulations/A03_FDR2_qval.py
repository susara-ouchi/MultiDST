import numpy as np
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

#loading p 
from A01_sim_data import p_values, fire_index, nonfire_index
from A01_weighting import weighted_p

#Define function for Sidak Procedure 
def sidak(p_values, alpha=0.05, weights = True):
    """
    Apply Šidák correction to lists of p-values.

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

        adj_p = [min(1-(1-p)**m,1.0) for p in p_values]
        sig_index = [index for index,p in enumerate(adj_p) if p < alpha]

    else:
        # Šidák correction
        adj_p = [min(1-(1-p)**m,1.0) for p in p_values]
        sig_index = [index for index,p in enumerate(adj_p) if p < alpha]

    return adj_p, sig_index


#Overall significance(unweighted)
sidak_test = sidak(p_values,alpha=0.05, weights = False)
sidak_p, bonf_sig_index = sidak_test[0], sidak_test[1]

#Overall significance(Weighted)
sidak_test = sidak(p_values,alpha=0.05, weights = True)
sidak_p, bonf_sig_index = sidak_test[0], sidak_test[1]

################################### Evaluating the Simulation ##################################################
    
#Pre-requisites
p_values
fire_index
nonfire_index
threshold = 0.05
adj_p = sidak_p

#significant p values
significant_p = [p_values[index] for index in bonf_sig_index]
sig_p= len(significant_p)
 
significant_p_fire = [adj_p[index] for index in fire_index if adj_p[index] < threshold]
significant_p_nonfire = [adj_p[index] for index in nonfire_index if adj_p[index] < threshold]

# COnfusion Matrix
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
TP_index = [index for index in fire_index if p_values[index] < threshold]
FN_index = [index for index in fire_index if p_values[index] >= threshold]
FP_index = [index for index in nonfire_index if p_values[index] < threshold]
TN_index = [index for index in nonfire_index if p_values[index] >= threshold]


