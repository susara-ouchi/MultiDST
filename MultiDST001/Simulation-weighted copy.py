import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.weighting import weighted_p_list
from MultiDST.bonferroni import bonferroni
from MultiDST.holm import holm
from MultiDST.sgof import sgof_test
from MultiDST.BH import bh_method
from MultiDST.qval import q_value
from MultiDST.BY import BY_method

from utils.visualization import draw_histogram
from utils.visualization import draw_p_bar_chart
from utils.visualization import plot_heatmap 
from utils.visualization import fire_hist
from utils.common_indices import common_indices

from simulation_functions import simulation_01
from simulation_functions import confmat
from simulation_functions import seq_test
from simulation_functions import DSTmulti_testing

from utils.weighting import multiweights

np.random.seed(42)
# 01
#Simulating Dataset for 500 F and 9500 NF 
sim1 = simulation_01(42,2000,8000,effect=1,n0=15,n1=15,threshold=0.05,show_plot=False,s0=1.0,s1=1.0)
p_values, significant_p,fire_index,nonfire_index = sim1[0],sim1[1],sim1[2],sim1[3]
og_p_values = p_values
fire_index_og, nonfire_index_og = fire_index, nonfire_index
p_value_fire = [p_values[i] for i in fire_index]
p_value_nonfire = [p_values[i] for i in nonfire_index]

# Observing histogram
# fire_hist(p_values, fire_index, nonfire_index,title="Histogram of p-values",col1 = 'skyblue', col2 = 'greenyellow',left='firing',right='non-firing')

sig_fire = [p for p in significant_p if p in fire_index]
sig_nonfire = [p for p in significant_p if p in nonfire_index]
len_p_fire = len(fire_index)
len_p_nonfire = len(nonfire_index)
res = confmat(sig_fire, sig_nonfire, len_p_fire, len_p_nonfire)
power,pi0 = res[0],res[1]

rejections = []
seq_test(rejections, og_p_values, p_value_fire, p_value_nonfire, threshold = 0.05)

# Dataframe to store rejections 

# Initialize an empty dictionary with empty lists for each column
df_sigp_dict = {
    "Firing":[],
    "Uncorrected": [],
    "Bonferroni":[],
    "Holm": [],
    "SGoF": [],
    "BH": [],
    "BY": [],
    "Q-value":[],
    "TP":[],
    "TN":[],
    "FP":[],
    "FN":[],
    "Power":[],
    "pi0": [],
    "pi0 estimate": [],
    "cutoff": [],
    "Total": []
}
df_sigp = pd.DataFrame(df_sigp_dict)

###################### Try 01 - Applying the methods #################################
k = 1
num_iter = 1

k1_list = ["--",0.0005,0.5,1.2]
k2_list = ["--",0.5,1.5,3]

for minw in k1_list:
      for maxw in k2_list:
        # Picture it before everything :)
        # fire_hist(p_values, fire_index, nonfire_index,title=fr"Histogram of p-values (Unweighted)",col1 = 'skyblue', col2 = 'greenyellow')
        if minw=="--" and maxw=="--":
            p_values = og_p_values
        elif minw=="--" or maxw=="--":
            continue
        else:
          p_values = og_p_values
          l0,l1,l2,l3,l4,l5,l6,l7 = common_indices(p_values, sig_bonf_p, sig_holm_p, sig_sgof_p, sig_bh_p, sig_by_p, sig_q)
          weighted_p = weighted_p_list(p_values,l0,l1,l2,l3,l4,l5,l6, weights="multi", max_weight = maxw, min_weight=minw)
          p_values = weighted_p[1]
        og_p_values

        fire_hist(p_values, fire_index, nonfire_index,title=fr"Histogram of p-values (Weighted) under $k_1$={minw} & $k_2$={maxw}",col1 = 'skyblue', col2 = 'greenyellow')

        results = DSTmulti_testing(p_values, alpha=0.05, weights=False)
        print(results)   

        sig_uncorrected = results["Uncorrected"]
        sig_bonf_p = results["Bonferroni"]
        sig_holm_p = results["Holm"]
        sig_sgof_p = results["SGoF"]
        sig_bh_p = results["BH"]
        sig_by_p = results["BY"]
        sig_q = results["Q-value"]
        pi0_est = results["pi0 estimate"]

        ### Applying the methods
        methods = ['Bonferroni', 'Holm', 'SGoF', 'BH', 'BY', 'Q value', 'True  Signals']
        sig_indices = [sig_bonf_p, sig_holm_p, sig_sgof_p, sig_bh_p, sig_by_p, sig_q,fire_index]
        plot_heatmap(methods, sig_indices, title=fr"Significant index plot  under $k_1$={minw} & $k_2$={maxw}")

        # Create a dictionary to map each list to its name
        indexed_sig_indices = dict(zip(methods, sig_indices))
        min_name = min(indexed_sig_indices, key=lambda x: len(indexed_sig_indices[x]))
        print(f"Min elements are found in {min_name}")
        low_ind = [i for i in range(len(sig_indices)) if len(sig_indices[i]) > 0]
        valid_indices1 = [i for i in range(len(sig_indices)) if len(sig_indices[i]) > 0] 
        valid_indices = [min(valid_indices1) if len(valid_indices1)>0 else min(low_ind)]
        min_index = valid_indices[0]
        # min_list = sig_indices[min_index]
        min_list = sig_bonf_p

        # Create a sublist containing the values corresponding to the first 7 keys
        len(min_list)
        rejections = min_list
        # rejections = rejections + min_list
        len(rejections)

        p_index2  = [i for i, val in enumerate(og_p_values) if i not in rejections]
        p_values2 = [val for i, val in enumerate(og_p_values) if i not in rejections]
        fire_index2 = [i for i,val in enumerate(p_values2) if p_values2[i] in p_value_fire]
        nonfire_index2 = [i for i,val in enumerate(p_values2) if p_values2[i] in p_value_nonfire]

        len(nonfire_index2)
        len(p_values2)
        significant_p == min_list

        significant_p=sig_bonf_p

        sig_fire = [p for p in significant_p if p in fire_index]
        sig_nonfire = [p for p in significant_p if p in nonfire_index]

        sig_fireBH = [p for p in sig_bh_p if p in fire_index]
        sig_nonfireBH = [p for p in sig_bh_p if p in nonfire_index]

        TPbh = len(sig_fireBH)
        FPbh = len(sig_nonfireBH)
        TNbh = len_p_nonfire - FPbh
        FNbh = len_p_fire - TPbh

        len_p_fire = len(fire_index)
        len_p_nonfire = len(nonfire_index)
        res = confmat(sig_fire, sig_nonfire, len_p_fire, len_p_nonfire)
        TP,TN,FP,FN,power,pi0 = res[0],res[1],res[2],res[3],res[4],res[5]
        total = TP+TN+FP+FN

        # Cut off constraint
        cutoff = (1-pi0_est[0])*total/k
        print(cutoff)


        p_sig_dict = {
        "K1":minw,
        "K2":maxw,
        "Firing": len_p_fire,
        "Uncorrected": len(sig_uncorrected),
        "Bonferroni": len(sig_bonf_p),
        "Holm": len(sig_holm_p),
        "SGoF": len(sig_sgof_p),
        "BH": len(sig_bh_p),
        "BY": len(sig_by_p),
        "Q-value": len(sig_q),
        "Bonf TP":TP,
        "Bonf TN":TN,
        "Bonf FP":FP,
        "Bonf FN":FN,
        "Power":power,
        "pi0": pi0,
        "pi0 estimate": pi0_est,
        "cutoff": cutoff,
        "Total":total,
        "Bonf Errors":FP+FN,
        "BH Error": FPbh+FNbh,
        "BH FP": FPbh,
        "BH FN": FNbh,
        "Bonf Power": power,
        "BH Power": TPbh/(TPbh+FNbh)
        }
        p_sig_new = pd.DataFrame(p_sig_dict, index=[0])

        # Concatenate the p_values_df DataFrame with your existing DataFrame
        df_sigp = pd.concat([df_sigp, p_sig_new], ignore_index=True)
        df_sigp
        df_sigp2 = df_sigp[['K1','K2','Bonferroni', 'Holm', 'SGoF', 'BH', 'BY', 'Q-value','Bonf FP','Bonf FN','BH FP','BH FN','Bonf Errors','BH Error']]
        df_sigp2
        df_sigp3 = df_sigp[['K1','K2','Bonferroni', 'BH','Q-value','Bonf FP','Bonf FN','BH FP','BH FN','Bonf Errors','BH Error','Bonf Power', "BH Power"]]
        df_sigp3


p_values = p_values2
fire_index = fire_index2
nonfire_index = nonfire_index2

#df_sigp2['Total']
df_sigp2.to_csv('MultiDST/MultiDST/MultiDST001/Sim_results1/sim12.csv', index=False)