############################## Diagnostics #################################
import matplotlib.pyplot as plt
import pandas as pd

from visualization import plot_power_effect
from visualization import plot_roc

### 00 - Loading the datasets
uncorrected_df = pd.read_csv(r'MultiDST\Simulated datasets\uncorrected_sim_results.csv')
bonferroni_df = pd.read_csv(r'MultiDST\Simulated datasets\bonferroni_sim_results.csv')
holm_df = pd.read_csv(r'MultiDST\Simulated datasets\holm_sim_results.csv')
sgof_df = pd.read_csv(r'MultiDST\Simulated datasets\sgof_sim_results.csv')
BH_df = pd.read_csv(r'MultiDST\Simulated datasets\bh_sim_results.csv')
BY_df = pd.read_csv(r'MultiDST\Simulated datasets\by_sim_results.csv')
storey_df =pd.read_csv(r'MultiDST\Simulated datasets\storeyQ_sim_results.csv')


### 01 - Setting up dataframes
def get_mean_power_values(df, group_by1, group_by2, num_groups):
    power_list = []
    for j in range(num_groups):
            power_list_g = []
            for i in range(num_groups):
                # Group by the first column
                grouped1 = df.groupby([group_by1])
                first_group_key = list(grouped1.groups.keys())[j]
                first_group = grouped1.get_group(first_group_key)

                # Group by the second column within the first group
                grouped2 = first_group.groupby([group_by2])
                second_group_key = list(grouped2.groups.keys())[i]
                second_group = grouped2.get_group(second_group_key)

                # Calculate the mean of the 'Power' column in the second group
                power_mean = second_group['Power'].mean()
                power_list_g.append(power_mean)
            power_list.append(power_list_g)
    return power_list

df = [bonferroni_df, holm_df, sgof_df, BH_df, BY_df, storey_df]
get_mean_power_values(df[0], 'n0', 'Effect', 3)

# Methods used
methods = ['Bonferroni', 'Holm', 'SGoF', 'BH', 'BY', 'Storey Q']

### 02 - Power over effect (under n0)

power_f = []
for i in range(len(df)):
    power_list = get_mean_power_values(df[i], 'n0', 'Effect', 3)
    power_f.append(power_list)
power_f

power_f2 = []
for i in range(3):
    nth_item = [sublist[i] for sublist in power_f]
    power_f2.append(nth_item)
power_f2


# For the n plot
effect_sizes = [0.5, 1.0, 1.5]
powers_n0 = power_f2[0]
powers_n1 = power_f2[1]
powers_n2 = power_f2[2]

titles =['n = 5', 'n = 15 ', 'n = 30']
x_labels = ['Effect size','Effect size','Effect size']
y_labels = ['Power','Power','Power']
plot_power_effect(methods, effect_sizes, powers_n0, powers_n1, powers_n2, titles=titles, x_labels=x_labels,y_labels=y_labels)


### 03 - Power over effect (under s0)
power_f = []
for i in range(len(df)):
    power_list = get_mean_power_values(df[i], 'S0', 'Effect', 2)
    power_f.append(power_list)
power_f

power_f2 = []
for i in range(3):
    nth_item = [sublist[i] for sublist in power_f]
    power_f2.append(nth_item)
power_f2

effect_sizes = [0.5, 1.0]
powers_s0 = power_f2[0]
powers_s1 = power_f2[1]

titles =['S = 0.5', 'S = 1.0']
x_labels = ['Effect size','Effect size']
y_labels = ['Power','Power']
plot_power_effect(methods, effect_sizes, powers_s0, powers_s1, titles=titles, x_labels=x_labels, y_labels=y_labels)


### 04 - Power over pi0 (under n0)
# For the n plot
effect_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]
powers_n0 = [[0.2, 0.4, 0.6, 0.8, 1.0], [0.3, 0.5, 0.7, 0.85, 0.95], [0.1, 0.3, 0.5, 0.75, 0.9],
             [0.2, 0.4, 0.6, 0.8, 1.0],[0.2, 0.4, 0.6, 0.8, 1.0],[0.2, 0.4, 0.6, 0.8, 1.0],[0.2, 0.4, 0.6, 0.8, 1.0]]
powers_n1 = [[0.2, 0.4, 0.6, 0.8, 1.0], [0.3, 0.5, 0.7, 0.85, 0.95], [0.1, 0.3, 0.5, 0.75, 0.9],
             [0.2, 0.4, 0.6, 0.8, 1.0],[0.2, 0.4, 0.6, 0.8, 1.0],[0.2, 0.4, 0.6, 0.8, 1.0],[0.2, 0.4, 0.6, 0.8, 1.0]]
powers_n2 = [[0.2, 0.4, 0.6, 0.8, 1.0], [0.3, 0.5, 0.7, 0.85, 0.95], [0.1, 0.3, 0.5, 0.75, 0.9],
             [0.2, 0.4, 0.6, 0.8, 1.0],[0.2, 0.4, 0.6, 0.8, 1.0],[0.2, 0.4, 0.6, 0.8, 1.0],[0.2, 0.4, 0.6, 0.8, 1.0]]

titles =['n = 5', 'n = 15 ', 'n = 30']
plot_power_effect(methods, effect_sizes, powers_n0, powers_n1, powers_n2, titles=titles)


### 05 - Power over pi0 (under s0)
effect_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]
powers_s0 = [[0.2, 0.4, 0.6, 0.8, 1.0], [0.3, 0.5, 0.7, 0.85, 0.95], [0.1, 0.3, 0.5, 0.75, 0.9],
             [0.2, 0.4, 0.6, 0.8, 1.0],[0.2, 0.4, 0.6, 0.8, 1.0],[0.2, 0.4, 0.6, 0.8, 1.0],[0.2, 0.4, 0.6, 0.8, 1.0]]
powers_s1 = [[0.2, 0.4, 0.6, 0.8, 1.0], [0.3, 0.5, 0.7, 0.85, 0.95], [0.1, 0.3, 0.5, 0.75, 0.9],
             [0.2, 0.4, 0.6, 0.8, 1.0],[0.2, 0.4, 0.6, 0.8, 1.0],[0.2, 0.4, 0.6, 0.8, 1.0],[0.2, 0.4, 0.6, 0.8, 1.0]]

titles =['Custom Title 1', 'Custom Title 2', 'Custom Title 3']
x_labels = ['Effect size','Effect size','Effect size']
y_labels = ['Power','Power','Power']
plot_power_effect(methods, effect_sizes, powers_s0, powers_s1, titles=titles)


### 06 - ROC curves over effect

def plot_roc(methods, tpr_list, fpr_list):
    plt.figure(figsize=(8, 6))
    for i in range(len(methods)):
        plt.plot(fpr_list[i], tpr_list[i], label=methods[i])
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example data for TPR and FPR for 7 different methods

tpr_list = [[0.2, 0.4, 0.6, 0.8, 1.0], [0.3, 0.5, 0.7, 0.85, 0.95], [0.1, 0.3, 0.5, 0.75, 0.9],
            [0.15, 0.35, 0.55, 0.75, 0.85], [0.25, 0.45, 0.65, 0.8, 0.95], [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.05, 0.1, 0.15, 0.2, 0.25]]
fpr_list = [[0.1, 0.2, 0.3, 0.4, 0.5], [0.15, 0.25, 0.35, 0.45, 0.55], [0.05, 0.15, 0.25, 0.35, 0.45],
            [0.1, 0.2, 0.3, 0.4, 0.5], [0.08, 0.18, 0.28, 0.38, 0.48], [0.05, 0.1, 0.15, 0.2, 0.25],
            [0.01, 0.05, 0.1, 0.15, 0.2]]

plot_roc(methods, tpr_list, fpr_list)
