# Simulating from Paired t-test

def simulation_01(seed,num_firing,num_nonfire,threshold=0.05,show_plot=False):
    '''
    This is to create p-values from ANOVA & uniform distribution
    '''
    import numpy as np
    import seaborn as sns
    import pandas as pd
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    from scipy.stats import f_oneway

    ############################### Simulating t-test (independent samples) ########################################3
    np.random.seed(seed)

    # Sample size
    size1 = 100
    size2 = 100
    size3 = 100
    #Treatment Group Distribution
    m1,m2,m3 = 0,2,5
    s1,s2,s3 = 1,1,1  

    #Groupwise sample distributions
    group1 = np.random.normal(m1,s1,size1)
    group2 = np.random.normal(m2,s2,size2)
    group3 = np.random.normal(m3,s3,size3)

    p_value_fire = []
    p_value_nonfire = []

    for i in range(num_firing):
        p_value = f_oneway(group1,group2,group3)[1]
        p_value_fire.append(p_value)

    for i in range(num_nonfire):
        p_value2 = np.random.uniform(0,1)
        p_value_nonfire.append(p_value2)

    p_values = p_value_fire + p_value_nonfire
    #Getting Firing and Non-Firing Indices
    fire_index = [index for index,p in enumerate(p_values) if p_values[index] in p_value_fire]
    nonfire_index = [index for index,p in enumerate(p_values) if p_values[index] in p_value_nonfire]
    print(len(fire_index),len(nonfire_index))

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

#Simulating Dataset for 500 F and 9500 NF 
sim1 = simulation_01(42,500,9500,threshold=0.05,show_plot=False)
p_values, significant_p,fire_index,nonfire_index = sim1[0],sim1[1],sim1[2],sim1[3]
p_values
