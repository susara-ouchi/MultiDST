from utils.weighting import weighted_p_list
from MultiDST.bonferroni import bonferroni
from MultiDST.holm import holm
from MultiDST.sgof import sgof_test
from MultiDST.BH import bh_method
from MultiDST.qval import q_value
from MultiDST.BY import BY_method

from utils.visualization import draw_histogram
from utils.visualization import sig_index_plot
from utils.visualization import draw_p_bar_chart
from utils.visualization import plot_heatmap 
from utils.visualization import fire_hist


import numpy as np
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import random

############################### Simulating t-test (independent samples) ########################################3

def simulation_01(seed,num_firing,num_nonfire,effect=0.5,n0=30,n1=30,threshold=0.05,show_plot=False,s0=1,s1=1):
    '''
    This is to create p-values from t-distribution & uniform distribution
    '''
    ############################### Simulating t-test (independent samples) ########################################3
    np.random.seed(seed)
    #Control Group Distribution
    m0 = 0
    s0
    n0

    #Treatment Group Distribution
    m1 = effect
    s1 
    n1

    p_value_fire = []
    p_value_nonfire = []

    for i in range(num_firing):
        control_group = np.random.normal(m0,s0,size =n0)
        treatment_group = np.random.normal(m1,s1,size=n1)
        p_value = sm.stats.ttest_ind(control_group, treatment_group,usevar='unequal')[1]
        p_value_fire.append(p_value)

    for i in range(num_nonfire):
        control_group2 = np.random.normal(m0,s0,size =n0)
        treatment_group2 = np.random.normal(m0,s1,size=n1)
        p_value2 = sm.stats.ttest_ind(control_group2, treatment_group2,usevar='unequal')[1]
        p_value_nonfire.append(p_value2)

    p_values = p_value_fire + p_value_nonfire
    #Getting Firing and Non-Firing Indices
    random.shuffle(p_values)
    
    fire_index = [index for index,p in enumerate(p_values) if p_values[index] in p_value_fire]
    nonfire_index = [index for index,p in enumerate(p_values) if p_values[index] in p_value_nonfire]
    #print(len(fire_index),len(nonfire_index))

    #significant p values
    significant_p =  [i for i,p in enumerate(p_values) if p < threshold]
   
    #Creating the plot
    hist_data = [p_value_fire, p_value_nonfire]
    plt.hist(hist_data, bins=30,alpha=1, label = ['firing','non-firing'],color=['skyblue','greenyellow'],edgecolor='black',stacked=True)
    plt.title(f'Distribution of uncorrected p-values for \n(effect = {effect} and pi0 = {num_nonfire/(num_firing+num_nonfire)})',fontname='Times New Roman')
    plt.xlabel('p-value',fontname='Times New Roman')
    plt.ylabel('Frequency',fontname='Times New Roman')
    plt.legend()
    if show_plot:
        plt.show()

    return p_values, significant_p, fire_index, nonfire_index

#Simulating Dataset for 500 F and 9500 NF 
sim1 = simulation_01(42,2000,8000,effect=1.5,n0=15,n1=15,threshold=0.05,show_plot=False,s0=1.0,s1=1.0)
p_values, significant_p,fire_index,nonfire_index = sim1[0],sim1[1],sim1[2],sim1[3]
og_p_values = p_values

p_value_fire = [p_values[i] for i in fire_index]
p_value_nonfire = [p_values[i] for i in nonfire_index]

fire_hist(p_values, fire_index, nonfire_index)



################################## Performance metrics ####################################

import numpy as np
from tabulate import tabulate

def confmat(sig_fire, sig_nonfire, len_p_fire, len_p_nonfire):
    # Counts of TP, TN, FP, FN
    TP = len(sig_fire)
    FP = len(sig_nonfire)
    TN = len_p_nonfire - FP
    FN = len_p_fire - TP

    if TP + FN == 0:
            sensitivity = 0  # If there are no actual positive cases, sensitivity is 0
    else:
        sensitivity = TP / (TP + FN)
    pi0 = len_p_nonfire/(TP+FP+TN+FN)

    print(f"TP: {TP}\nFP: {FP}\nTN: {TN}\nFN: {FN}\n\nPower:{sensitivity}\npi0 ={pi0}")
    #print(f"TP: {TP}\nFP: {FP}\nTN: {TN}\nFN: {FN}\n\nPower:{sensitivity}")
    return TP,TN,FP,FN,sensitivity,pi0

# Example usage:
sig_fire = [p for p in significant_p if p in fire_index]
sig_nonfire = [p for p in significant_p if p in nonfire_index]
len_p_fire = len(fire_index)
len_p_nonfire = len(nonfire_index)
confmat(sig_fire, sig_nonfire, len_p_fire, len_p_nonfire)

## Sequential testing function
def seq_test(rejections, og_p_values, p_value_fire, p_value_nonfire, threshold =0.05):
      p_index2  = [i for i, val in enumerate(og_p_values) if i not in rejections]
      p_values2 = [val for i, val in enumerate(og_p_values) if i not in rejections]
      fire_index2 = [i for i,val in enumerate(p_values2) if p_values2[i] in p_value_fire]
      nonfire_index2 = [i for i,val in enumerate(p_values2) if p_values2[i] in p_value_nonfire]
      len(p_values2) == len(p_index2)

      significant_p = [i for i,p in enumerate(p_values2) if p < threshold]
      sig_fire = [p for p in significant_p if p in fire_index2]
      sig_nonfire = [p for p in significant_p if p in nonfire_index2]
      len_p_fire = len(fire_index2)
      len_p_nonfire = len(nonfire_index2)
      result = confmat(sig_fire, sig_nonfire, len_p_fire, len_p_nonfire)
      return result


##########################################################################################

## Hypothesis testing


def DSTmulti_testing(p_values, alpha=0.05, weights=False):
    # 0 - Uncorrected
    sig_index = [index for index, p in enumerate(p_values) if p < alpha]
    uncorrected_count = len(sig_index)
    print("Uncorrected count:", uncorrected_count)

    # 1 - Bonferroni
    bonf_results = bonferroni(p_values, alpha=alpha, weights=weights)
    bonf_p, sig_bonf_p = bonf_results[0], bonf_results[1]
    bonf_count = len(sig_bonf_p)
    print("Bonferroni count:", bonf_count)

    # 2 - Holm
    holm_results = holm(p_values, alpha=alpha, weights=weights)
    holm_p, sig_holm_p = holm_results[0], holm_results[1]
    holm_count = len(sig_holm_p)
    print("Holm count:", holm_count)

    # 3 - SGoF
    sgof_results = sgof_test(p_values, alpha=alpha, weights=weights)
    sgof_p, sig_sgof_p = sgof_results[0], sgof_results[1]
    sgof_count = len(sig_sgof_p)
    print("SGoF count:", sgof_count)

    # 4 - BH
    bh_results = bh_method(p_values, alpha=alpha, weights=weights)
    bh_p, sig_bh_p = bh_results[0], bh_results[1]
    bh_count = len(sig_bh_p)
    print("BH count:", bh_count)

    # 5 - BY
    by_results = BY_method(p_values, alpha=alpha, weights=weights)
    by_p, sig_by_p = by_results[0], by_results[1]
    by_count = len(sig_by_p)
    print("BY count:", by_count)

    # 6 - Qval
    q_results = q_value(p_values, alpha=alpha, weights=weights)
    q, sig_q,pi0_est = q_results[0], q_results[1],q_results[2]
    q_count = len(sig_q)
    print("Q-value count:", q_count)

    return {
        "Uncorrected": sig_index,
        "Bonferroni": sig_bonf_p,
        "Holm": sig_holm_p,
        "SGoF": sig_sgof_p,
        "BH": sig_bh_p,
        "BY": sig_by_p,
        "Q-value": sig_q,
        "pi0 estimate": pi0_est
    }


DSTmulti_testing(p_values, alpha=0.05, weights=False)