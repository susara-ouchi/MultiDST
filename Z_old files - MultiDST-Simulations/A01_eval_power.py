#################### Simulation ##################################

# Simulating from Independent samples t-test #

#Hypothesis:
#  H0: p-values come from the same distribution 
#  H1: p-values comes from two different distributions
import numpy as np
import matplotlib.pyplot as plt

def simulation_01(seed,num_firing,num_nonfire,effect=0.5,threshold=0.05,show_plot=False):
    '''
    This is to create p-values from t-distribution & uniform distribution
    '''
    import numpy as np
    import seaborn as sns
    import pandas as pd
    import statsmodels.api as sm
    import matplotlib.pyplot as plt

    ############################### Simulating t-test (independent samples) ########################################3
    np.random.seed(seed)
    #Control Group Distribution
    m0 = 0
    s0 = 1
    n0 = 100

    #Treatment Group Distribution
    m1 = effect
    s1 = 1
    n1 = 100

    p_value_fire = []
    p_value_nonfire = []

    for i in range(num_firing):
        control_group = np.random.normal(m0,s0,size =n0)
        treatment_group = np.random.normal(m1,s1,size=n1)
        p_value = sm.stats.ttest_ind(control_group, treatment_group)[1]
        p_value_fire.append(p_value)

    for i in range(num_nonfire):
        p_value2 = np.random.uniform(0,1)
        p_value_nonfire.append(p_value2)

    p_values = p_value_fire + p_value_nonfire
    #Getting Firing and Non-Firing Indices
    fire_index = [index for index,p in enumerate(p_values) if p_values[index] in p_value_fire]
    nonfire_index = [index for index,p in enumerate(p_values) if p_values[index] in p_value_nonfire]
    #print(len(fire_index),len(nonfire_index))

    #significant p values
    significant_p =  [i for i,p in enumerate(p_values) if p < threshold]
   
    #Creating the plot
    hist_data = [p_value_fire, p_value_nonfire]
    plt.hist(hist_data, bins=30,alpha=0.5, label = ['firing','non-firing'],stacked=True)
    plt.title('Distribution of uncorrected p-values')
    plt.xlabel('p-value')
    plt.ylabel('Frequency')
    plt.legend()
    if show_plot:
        plt.show()

    return p_values, significant_p, fire_index, nonfire_index

####################################### Sim eval ##############################################
def sim_eval(seed,adj_p, sig_index, threshold =0.05):
    import pandas as pd
    import numpy as np
    sim1 = simulation_01(seed,9500,500,threshold=0.05,show_plot=False)
    p_values, significant_p,fire_index,nonfire_index = sim1[0],sim1[1],sim1[2],sim1[3]
    p_fire = len(fire_index)
    significant_p = [p_values[index] for index in sig_index]
    significant_p_fire = [adj_p[index] for index in fire_index if adj_p[index] < threshold]
    significant_p_nonfire = [adj_p[index] for index in nonfire_index if adj_p[index] < threshold]
    return p_values, significant_p_fire,p_fire

########################## Getting simulation results ##############################
from A02_FWER1_bonferroni import bonf_p, bonf_sig_index, bonf_w_p, bonf_w_sig_index
from A02_FWER2_sidak import sidak_p,sidak_sig_index,sidak_w_p,sidak_w_sig_index
from A02_FWER3_holm import holm_p,holm_sig_index,holm_w_p,holm_w_sig_index
from A02_FWER4_simes import simes_p,simes_sig_index,simes_w_p,simes_w_sig_index
from A03_FDR1_bh import bh_p,bh_sig_index,bh_w_p,bh_w_sig_index
from A03_FDR2_qval import storey_q,q_sig_index

#Uncorrected
#Uncorrected
sig_fire_results = []
for i in range(5): 
    seed = i
    sim_result = simulation_01(seed, 9500, 500, effect=0.05, threshold=0.05, show_plot=False)
    p_values = sim_result[0]
    significant_p = sim_result[1]
    sim_eval_res = sim_eval(seed, p_values, significant_p, threshold=0.05)
    sig_fire = sim_eval_res[1]
    p_fire = sim_eval_res[2]
    sig_fire_results.append(len(sig_fire)/p_fire)
 
p_fire
sig_fire_results
power = np.mean(sig_fire_results)
sd = np.std(sig_fire_results)
print(f"power: {power}\nStd dev: {sd}")





'''
# Plotting the data with error bars
plt.errorbar(0.2, power, yerr=sd, fmt='o', color='black', ecolor='red', capsize=5)
#plt.axhline(y=mean, color='blue', linestyle='--', label='Mean')

# Adding labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Plot with Mean and Standard Error Bars')
'''