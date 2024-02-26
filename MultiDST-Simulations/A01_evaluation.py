################################### Evaluating the Simulation ##################################################
# Test #

#Pre-requisites
from A01_sim_data import p_values, fire_index, nonfire_index,significant_p
from A02_FWER1_bonferroni import bonf_p, bonf_sig_index, bonf_w_p, bonf_w_sig_index
from A02_FWER2_sidak import sidak_p,sidak_sig_index,sidak_w_p,sidak_w_sig_index
from A02_FWER3_holm import holm_p,holm_sig_index,holm_w_p,holm_w_sig_index
from A02_FWER4_simes import simes_p,simes_sig_index,simes_w_p,simes_w_sig_index
from A03_FDR1_bh import bh_p,bh_sig_index,bh_w_p,bh_w_sig_index
from A03_FDR2_qval import storey_q,q_sig_index
from A04_permutation_test import perm_p, perm_sig

def sim_eval(p_values, fire_index, nonfire_index, adj_p, sig_index, threshold =0.05):
    import pandas as pd
    import numpy as np
    significant_p = [p_values[index] for index in sig_index]
    significant_p_fire = [adj_p[index] for index in fire_index if adj_p[index] < threshold]
    significant_p_nonfire = [adj_p[index] for index in nonfire_index if adj_p[index] < threshold]
 
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
    precision = TP/(TP+FP)
    sensitivity = TP/(TP+FN)
    specificity = TN/(TN+FP)
    balanced_accuracy = (sensitivity+specificity)/2
    f1_score = 2*(precision*sensitivity)/(precision+sensitivity)
        
    #To find which genes are significant
    TP_index = [index for index in fire_index if adj_p[index] < threshold]
    FN_index = [index for index in fire_index if adj_p[index] >= threshold]
    FP_index = [index for index in nonfire_index if adj_p[index] < threshold]
    TN_index = [index for index in nonfire_index if adj_p[index] >= threshold]
    return sensitivity,specificity, balanced_accuracy, f1_score, confusion_matrix,significant_p_fire

#Getting Evaluation Results
corr_method = ["Uncorrected","Bonferroni","Weighted Bonf","Sidak","Weighted Sidak","Holm","Weighted Holm","Simes","Weighted Simes", "BH method","Weighted BH (Genovese) Method","Q-value","Permutation test"]
adj_p_list = [p_values, bonf_p, bonf_w_p,sidak_p,sidak_w_p,holm_p,holm_w_p,simes_p, simes_w_p,bh_p,bh_w_p,storey_q, perm_p]
sig_index_list = [significant_p,bonf_sig_index, bonf_w_sig_index,sidak_sig_index, sidak_w_sig_index,holm_sig_index,holm_w_sig_index,simes_sig_index, simes_w_sig_index,bh_sig_index,bh_w_sig_index,q_sig_index, perm_sig]
sim_results = sim_eval(p_values, fire_index, nonfire_index, adj_p_list[0], sig_index_list[0], threshold =0.05)
len(sim_results[5])

for i in range(len(corr_method)):
    print(f"\n* Results for {corr_method[i]}:\n")
    sim_results = sim_eval(p_values, fire_index, nonfire_index, adj_p_list[i], sig_index_list[i], threshold =0.05)
    sim_sensitivity = sim_results[0]
    sim_specificity = sim_results[1]
    balanced_accuracy = sim_results[2]
    f1_score = sim_results[3]
    print(f"Sensitivity: {sim_sensitivity}\nSpecificty: {sim_specificity}\nBalanced Accuracy: {balanced_accuracy}\nF1-score: {f1_score}")
    conf_sim = sim_results[4]
    conf_sim
    print("====================================================")


