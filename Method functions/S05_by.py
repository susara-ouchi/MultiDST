#################### B-Y Simulation ##################################

# Simulating from Independent samples t-test #

#Hypothesis:
#  H0: p-values come from the same distribution 
#  H1: p-values comes from two different distributions

# from A01_sim_data import simulation_01
# simulation_01(1,9500,500,threshold=0.05,show_plot=False)

import numpy as np
import matplotlib.pyplot as plt
import time
from joblib import Parallel, delayed

###################################### Simulation distribution loading ###########################################
from A01_sim_data import simulation_01
from A03_FDR3_BY import BY_method

####################################### Sim eval ##############################################

def sim_eval(seed,adj_p, sig_index, threshold =0.05):
    sim1 = simulation_01(seed,9500,500,threshold=0.05,show_plot=False,s0=1,s1=1)
    p_values, significant_p,fire_index,nonfire_index = sim1[0],sim1[1],sim1[2],sim1[3]
    p_fire = len(fire_index)
    p_nonfire = len(nonfire_index)
    #significant_p = [p_values[index] for index in sig_index]
    significant_p_fire = [adj_p[index] for index in fire_index if adj_p[index] < threshold]
    significant_p_nonfire = [adj_p[index] for index in nonfire_index if adj_p[index] < threshold]

    return p_values, significant_p_fire,significant_p_nonfire,p_fire,p_nonfire

########################## Getting simulation results ##############################
n0_list = []
effect_list = []
pi0_list = []
s0_list = []
power_list,power_sd_list = [],[]
fdr_list,fdr_sd_list = [],[]
accuracy_list, accuracy_sd_list = [],[]
fpr_list, fpr_sd_list =[],[]
f1_list, f1_sd_list = [],[]

def power_sim1(num_simulations,n0,num_firing,num_nonfire,effect,pi0,s0=1):
    import pandas as pd
    n1 = n0
    s1 = s0
    sim_power = []
    sim_acc =[]
    sim_fdr =[]
    sim_fpr =[]
    sim_f1 = []

    for i in range(num_simulations): 
        seed = i
        sim_result = simulation_01(seed,num_firing,num_nonfire,effect,n0,n1,threshold=0.05,show_plot=False, s0=1, s1=1)
        p_values = sim_result[0]
        #significant p-values from method
        significant_p = BY_method(p_values,alpha=0.05, weights = False)[1]
        sim_eval_res = sim_eval(seed, p_values, significant_p, threshold=0.05)
        sig_fire = sim_eval_res[1]
        sig_nonfire = sim_eval_res[2]
        p_fire = sim_eval_res[3]
        p_nonfire = sim_eval_res[4]

        # Counts of TP,TN,FP,FN
        TP = len(sig_fire)
        FP = len(sig_nonfire)
        TN = p_nonfire - FP
        FN = p_fire - TP

        fdr = FP/(TP+FP)
        precision = TP/(TP+FP)
        sensitivity = TP/(TP+FN)
        specificity = TN/(TN+FP)
        fpr = 1 - specificity
        balanced_accuracy = (sensitivity+specificity)/2
        f1_score = 2*(precision*sensitivity)/(precision+sensitivity)

        sim_power.append(sensitivity)    # sensitivity = TPR = power
        sim_fdr.append(fdr)
        sim_acc.append(balanced_accuracy)
        sim_fpr.append(fpr)
        sim_f1.append(f1_score)
        print(f"working on iteration {i+1}")

    power = np.mean(sim_power)
    sd = np.std(sim_power)

    fdr = np.mean(sim_fdr)
    fdr_sd = np.std(sim_fdr)

    acc = np.mean(sim_acc)
    acc_sd = np.std(sim_acc)

    fpr = np.mean(sim_fpr)
    fpr_sd = np.std(sim_fpr)

    f1 = np.mean(sim_f1)
    f1_sd = np.std(sim_f1)

    # Intermediate table
    from tabulate import tabulate
    # Create a list of tuples for the data
    data = [
        ("Power (TPR)", power, sd),
        ("FDR", fdr, fdr_sd),
        ("Accuracy", acc, acc_sd),
        ("FPR", fpr, fpr_sd),
        ("F1", f1, f1_sd)
    ]
    # Print the table
    print(tabulate(data, headers=["Metric", "Value", "Std Dev"], tablefmt="grid"))
    print("\n---------------------------------------------\n")

    # Appending values to the lists
    n0_list.append(n0)
    effect_list.append(effect)
    s0_list.append(s0)
    pi0_list.append(pi0)
    power_list.append(power)
    power_sd_list.append(sd)
    fdr_list.append(fdr)
    fdr_sd_list.append(fdr_sd)
    accuracy_list.append(acc)
    accuracy_sd_list.append(acc_sd)
    fpr_list.append(fpr)
    fpr_sd_list.append(fpr_sd)
    f1_list.append(f1)
    f1_sd_list.append(f1_sd)
    print("Done!\n...")

def power_sim_sample(num_simulations):
    sample_size = [5,15,30] #[5,15,30]      # From Kang
    print("\n---------------------------------------------\n")
    for l in sample_size:
        n0 = l
        num_firing = [10000,9000,7500,5000] #[10000,9000,7500,5000, 3000]    # From BonEV
        total_p = 10000
        for k in num_firing:
            num_firing = k
            num_nonfire = total_p - k
            pi0 = num_firing/total_p
            effect_size = [0.1,0.3,0.5] #[0.05, 0.1, 0.3, 0.5]     # From SGoF
            for j in effect_size:
                effect= j
                s0_size = [0.5,1]
                for m in s0_size:
                    s0 = m
                    print(f"n0 = n1 = {n0}",
                        f"\nfiring: {num_firing}\nnon-firing: {num_nonfire}\npi0 = {pi0}",
                        f"\neffect size: {effect}\ns0 = s1: {s0}\n...")
                    power_sim1(num_simulations,n0,num_firing,num_nonfire,effect,pi0,s0)


t1 = time.time()
#results = [math.factorial(x) for x in range(10000)]
results = Parallel(n_jobs=-1)(power_sim_sample(num_simulations=50))
t2 = time.time()

n0_list
effect_list
pi0_list
s0_list
power_list
power_sd_list 
fdr_list
fdr_sd_list
accuracy_list
accuracy_sd_list 
fpr_list
fpr_sd_list
f1_list
f1_sd_list 

# Create a DataFrame
import pandas as pd

data = {
    'n0': n0_list,
    'Pi0': pi0_list,
    'Effect': effect_list,
    'S0':s0_list,
    'Power': power_list,
    'Power SD': power_sd_list,
    'FDR': fdr_list,
    'FDR SD': fdr_sd_list,
    'Accuracy': accuracy_list,
    'Accuracy SD': accuracy_sd_list,
    'TPR': power_list,
    'TPR SD': power_sd_list,
    'FPR': fpr_list,
    'FPR SD': fpr_sd_list,
    'F1': f1_list,
    'F1 SD': f1_sd_list
}

df_by = pd.DataFrame(data)

# Print the DataFrame
print(df_by)
print(t2-t1)

df_by.to_csv('MultiDST/Simulated datasets/by_sim_results.csv', index=False)

