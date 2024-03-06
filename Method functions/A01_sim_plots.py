#####################################################################################
################ Simulating from Independent samples t-test #########################


#Hypothesis:
#  H0: p-values come from the same distribution 
#  H1: p-values comes from two different distributions

def simulation_01_plots(seed,num_firing,num_nonfire,effect=0.5,n0=30,n1=30,threshold=0.05):
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
    n0

    #Treatment Group Distribution
    m1 = effect
    s1 = 1
    n1

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
   
    return p_values, significant_p, p_value_fire,p_value_nonfire, effect

#Simulating Dataset for 500 F and 9500 NF 
sim1 = simulation_01_plots(42,9000,1000,effect=0.5,n0=5,n1=5,threshold=0.05)
p_values, significant_p, p_value_fire,p_value_nonfire, effect = sim1[0],sim1[1],sim1[2],sim1[3], sim1[4]


import matplotlib.pyplot as plt
import numpy as np

#Creating the plot
sim1 = simulation_01_plots(42,9000,1000,effect=0.5,n0=5,n1=5,threshold=0.05)
hist_data = [p_value_fire, p_value_nonfire]
plt.hist(hist_data, bins=30,alpha=1, label = ['firing','non-firing'],color=['steelblue','greenyellow'],edgecolor='black',stacked=True)
plt.title(f'Distribution of uncorrected p-values for \n(effect = {effect} and pi0 = {len(p_value_fire)/len(p_values)})',fontname='Times New Roman')
plt.xlabel('p-value',fontname='Times New Roman')
plt.ylabel('Frequency',fontname='Times New Roman')
plt.legend()
plt.show()


simulation_01_plots(42,9000,1000,effect=0.5,n0=5,n1=5,threshold=0.05)


# Define the parameters for the simulations
seeds = [42, 42, 42, 42, 42, 42]  # Example seeds (replicated 6 times)
num_firing = [5000, 5000, 5000, 9000, 9000, 9000]
num_nonfire = [5000, 5000, 5000, 1000, 1000, 1000]
effects = [0.1, 0.3, 0.5, 0.1, 0.3, 0.5]  # Example effect sizes (replicated 6 times)
n0_values = [30 for i in range(6)]  # Example n0 values (replicated 6 times)
n1_values = [30 for i in range(6)]  # Example n1 values (replicated 6 times)
threshold = 0.05

# Create subplots with adjusted spacing
fig, axs = plt.subplots(2, 3, figsize=(10, 6))  # Adjusted to accommodate 6 plots in a (2, 3) grid
plt.subplots_adjust(hspace=0.5, wspace=1.5)  # Adjust vertical and horizontal spacing between subplots

# Iterate over the parameters and generate plots
for i in range(6):  # Adjusted to iterate 6 times
    sim_data = simulation_01_plots(seeds[i], num_firing[i], num_nonfire[i], effects[i], n0_values[i], n1_values[i], threshold)
    hist_data = [sim_data[2], sim_data[3]]  # Assuming the function returns a list containing p-values for firing and non-firing
    ax = axs[i // 3, i % 3]
    ax.hist(hist_data, bins=30, alpha=1, label=['firing', 'non-firing'], color=['skyblue', 'greenyellow'], edgecolor='midnightblue', stacked=True)
    ax.set_title(f'Distribution of uncorrected p-values for\n(effect = {effects[i]} and pi0 = {len(sim_data[2])/len(sim_data[0])})', fontname='Times New Roman')
    ax.set_xlabel('p-value', fontname='Times New Roman')
    ax.set_ylabel('Frequency', fontname='Times New Roman')
    ax.legend()

# Adjust layout
plt.tight_layout()
plt.show()