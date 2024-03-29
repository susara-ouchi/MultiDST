#####################################################################################
################ Simulating from Independent samples t-test #########################


#Hypothesis:
#  H0: p-values come from the same distribution 
#  H1: p-values comes from two different distributions

def simulation_01(seed,num_firing,num_nonfire,effect=0.5,n0=30,n1=30,threshold=0.05,show_plot=False,s0=1,s1=1):
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
    plt.hist(hist_data, bins=30,alpha=1, label = ['firing','non-firing'],color=['navy','lime'],edgecolor='black',stacked=True)
    plt.title(f'Distribution of uncorrected p-values for \n(effect = {effect} and pi0 = {num_firing/(num_firing+num_nonfire)})',fontname='Times New Roman')
    plt.xlabel('p-value',fontname='Times New Roman')
    plt.ylabel('Frequency',fontname='Times New Roman')
    plt.legend()
    if show_plot:
        plt.show()

    return p_values, significant_p, fire_index, nonfire_index

#Simulating Dataset for 500 F and 9500 NF 
sim1 = simulation_01(42,5000,5000,effect=0.5,n0=5,n1=5,threshold=0.05,show_plot=True,s0=1.0,s1=1.0)
p_values, significant_p,fire_index,nonfire_index = sim1[0],sim1[1],sim1[2],sim1[3]


