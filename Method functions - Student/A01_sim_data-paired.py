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

    ############################### Simulating MW-test (independent samples) ########################################3
    np.random.seed(seed)
    import numpy as np
    from scipy.stats import ttest_rel

    np.random.seed(seed)

    # Generate paired samples for control and treatment groups
    paired_samples = np.random.normal(loc=0, scale=s0, size=(n0, num_firing))
    paired_samples_treatment = paired_samples + np.random.normal(loc=effect, scale=np.sqrt(s0**2 + s1**2), size=(n0, num_firing))

    # Calculate paired t-test statistics and p-values
    p_values = []
    for i in range(num_firing):
        statistic, p_value = ttest_rel(paired_samples[:, i], paired_samples_treatment[:, i])
        p_values.append(p_value)

    # Non-firing p-values (for comparison)
    p_value_nonfire = np.random.uniform(0, 1, size=num_nonfire)

    # Combine firing and non-firing p-values
    p_values = np.concatenate((p_values, p_value_nonfire))

    # Determine significant p-values
    significant_p = [i for i, p in enumerate(p_values) if p < threshold]

    # Plotting (optional)
    plt.hist(p_values, bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('p-value')
    plt.ylabel('Frequency')
    plt.title('Distribution of p-values (Paired t-test)')
    plt.show()

    return p_values, significant_p, fire_index, nonfire_index

#Simulating Dataset for 500 F and 9500 NF 
sim1 = simulation_01(42,5000,5000,effect=0.5,n0=5,n1=5,threshold=0.05,show_plot=True,s0=1.0,s1=1.0)
p_values, significant_p,fire_index,nonfire_index = sim1[0],sim1[1],sim1[2],sim1[3]


