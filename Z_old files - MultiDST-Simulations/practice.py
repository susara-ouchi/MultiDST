import numpy as np
from scipy.stats import ttest_ind

# Parameters
sample_size = 100  # Sample size
effect_size = 0.05  # Effect size
num_simulations = 100  # Number of simulations
alpha = 0.05  # Significance level

# Simulate data
null_data = np.random.uniform(0, 1, size=(sample_size, num_simulations))  # Null distribution
alternative_data = np.random.normal(0.5 + effect_size, 1, size=(sample_size, num_simulations))  # Alternative distribution

# Perform t-test and calculate power
reject_null = np.zeros(num_simulations, dtype=bool)
for i in range(num_simulations):
    _, p_value = ttest_ind(null_data[:, i], alternative_data[:, i])
    reject_null[i] = p_value < alpha
power = np.mean(reject_null)
print("Power:", power)


from A01_sim_data import p_values,fire_index,nonfire_index

#Simulations
num_simulations = 1000
fire_index = [p_values[i] for i in fire_index]
reject_null = np.zeros(num_simulations, dtype=bool)
for i in range(num_simulations):
    for p_value in fire_index:
        reject_null[i] = p_value < 0.05
reject_null 
power = np.mean(reject_null)
print("Power:", power)

def sim_eval(p_values, fire_index, nonfire_index, adj_p, sig_index, threshold =0.05):
    import pandas as pd
    import numpy as np
    significant_p = [p_values[index] for index in sig_index]
    significant_p_fire = [adj_p[index] for index in fire_index if adj_p[index] < threshold]
    significant_p_nonfire = [adj_p[index] for index in nonfire_index if adj_p[index] < threshold]

    #Simulations
    num_simulations = 1000
    reject_null = np.zeros(num_simulations, dtype=bool)
    for i in range(num_simulations):
        for p_value in p_values:
            reject_null[i] = p_value < threshold
    power = np.mean(reject_null)
    print("Power:", power)
 
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
    return power, sensitivity,specificity, balanced_accuracy, f1_score, confusion_matrix,TP_index



import pandas as pd

# Assuming df is your DataFrame containing the data
data = {'student_id': [101, 53, 128, 3],
        'name': ['Ulysses', 'William', 'Henry', 'Henry'],
        'age': [13, 10, 6, 11]}

df = pd.DataFrame(data)

# Get name and age of student_id 101 as a DataFrame
student_101_df = df[df['student_id'] == 101][['name', 'age']]