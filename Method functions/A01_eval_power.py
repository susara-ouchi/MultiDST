#################### Simulation ##################################

# Simulating from Independent samples t-test #

#Hypothesis:
#  H0: p-values come from the same distribution 
#  H1: p-values comes from two different distributions
import numpy as np
import matplotlib.pyplot as plt

###################################### Simulation ###########################################
from A01_sim_data import simulation_01

'''
sim01 = simulation_01(42,9000,1000, effect = 0.3, threshold=0.05,show_plot=True)
p_values = sim01[0]
sig_p = sim01[1]
'''

####################################### Sim eval ##############################################
def sim_eval(seed,adj_p, sig_index, threshold =0.05):
    sim1 = simulation_01(seed,9500,500,threshold=0.05,show_plot=False)
    p_values, significant_p,fire_index,nonfire_index = sim1[0],sim1[1],sim1[2],sim1[3]
    p_fire = len(fire_index)
    p_nonfire = len(nonfire_index)
    #significant_p = [p_values[index] for index in sig_index]
    significant_p_fire = [adj_p[index] for index in fire_index if adj_p[index] < threshold]
    significant_p_nonfire = [adj_p[index] for index in nonfire_index if adj_p[index] < threshold]

    return p_values, significant_p_fire,significant_p_nonfire,p_fire,p_nonfire

########################## Getting simulation results ##############################

from A01_weighting import weighted_p_list
from A02_FWER1_bonferroni import bonferroni
from A02_FWER3_holm import holm
from A02_FWER5_sgof import sgof_test
from A03_FDR1_bh import bh_method
from A03_FDR2_qval import q_value
from A03_FDR3_BY import BY_method

# 1 - Bonferroni
bonf_results = bonferroni(p_values,alpha=0.05, weights = False)
bonf_p, sig_bonf_p = bonf_results[0], bonf_results[1]
sig_bonf_p
len(sig_bonf_p)

# 2 - Holm
holm_results = holm(p_values,alpha=0.05, weights = False)
holm_p, sig_holm_p = holm_results[0], holm_results[1]
sig_holm_p
len(sig_holm_p)

# 3 - SGoF
sgof_results = sgof_test(p_values,alpha=0.05, weights = False)
sgof_p, sig_sgof_p = sgof_results[0], sgof_results[1]
sig_sgof_p
len(sig_sgof_p)

# 4 - BH
bh_results = bh_method(p_values,alpha=0.05, weights = False)
bh_p, sig_bh_p = bh_results[0], bh_results[1]
bh_p
sig_bh_p
len(sig_bh_p)

# 5 - BY
by_results = BY_method(p_values,alpha=0.05, weights = False)
by_p, sig_by_p = by_results[0], by_results[1]
by_p
sig_by_p
len(sig_by_p)


# 6 - Qval
q_results = q_value(p_values,alpha=0.05, weights = False)
q, sig_q = q_results[0], q_results[1]
q
sig_q
len(sig_q)

len(sig_bonf_p)
len(sig_holm_p)
len(sig_sgof_p)
len(sig_bh_p)
len(sig_by_p)
len(sig_q)


#Try 0 - Uncorrected
import pandas as pd
sim_power = []
sim_acc =[]
sim_fdr =[]
sim_f1 = []

for i in range(5): 
    seed = i
    sim_result = simulation_01(seed, 9500, 500, effect=0.05, threshold=0.05, show_plot=False)
    p_values = sim_result[0]
    significant_p = sim_result[1]
    sim_eval_res = sim_eval(seed, p_values, significant_p, threshold=0.05)
    sig_fire = sim_eval_res[1]
    sig_nonfire = sim_eval_res[2]
    p_fire = sim_eval_res[3]
    p_nonfire = sim_eval_res[4]

    # Confusion Matrix
    TP = len(sig_fire)
    FP = len(sig_nonfire)
    TN = p_nonfire - FP
    FN = p_fire - TP

    sig_p= len(significant_p)
    fdr = FP/(TP+FP)
    precision = TP/(TP+FP)
    sensitivity = TP/(TP+FN)
    specificity = TN/(TN+FP)
    balanced_accuracy = (sensitivity+specificity)/2
    f1_score = 2*(precision*sensitivity)/(precision+sensitivity)

    sim_power.append(TP/p_fire)
    sim_fdr.append(fdr)
    sim_acc.append(balanced_accuracy)
    sim_f1.append(f1_score)
    print(f"working on iteration {i+1}")


power = np.mean(sim_power)
sd = np.std(sim_power)
print(f"power: {power}\nStd dev: {sd}")

fdr = np.mean(sim_fdr)
fdr_sd = np.std(sim_fdr)
print(f"fdr: {fdr}\nStd dev: {fdr_sd}")

acc = np.mean(sim_acc)
acc_sd = np.std(sim_acc)
print(f"f1: {acc}\nStd dev: {acc_sd}")

f1 = np.mean(sim_f1)
f1_sd = np.std(sim_f1)
print(f"f1: {f1}\nStd dev: {f1_sd}")