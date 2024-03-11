############################## Diagnostics #################################
import matplotlib.pyplot as plt
import pandas as pd

from visualization import plot_power_effect
from visualization import plot_roc
from visualization import plot_radar_plots

### 00 - Loading the datasets
uncorrected_df = pd.read_csv(r'MultiDST\Simulated datasets 1\uncorrected_sim_results.csv')
bonferroni_df = pd.read_csv(r'MultiDST\Simulated datasets 1\bonferroni_sim_results.csv')
holm_df = pd.read_csv(r'MultiDST\Simulated datasets 1\holm_sim_results.csv')
sgof_df = pd.read_csv(r'MultiDST\Simulated datasets 1\sgof_sim_results.csv')
BH_df = pd.read_csv(r'MultiDST\Simulated datasets 1\bh_sim_results.csv')
BY_df = pd.read_csv(r'MultiDST\Simulated datasets 1\by_sim_results.csv')
storey_df =pd.read_csv(r'MultiDST\Simulated datasets 1\storeyQ_sim_results.csv')
all_df = pd.read_csv(r'MultiDST\Simulated datasets 1\all_df.csv')

### 01 - Setting up dataframes
def get_mean_power_values(df, group_by1, group_by2, num_groups1,num_groups2, val='Power'):
    power_list = []
    for j in range(num_groups1):
            power_list_g = []
            for i in range(num_groups2):
                # Group by the first column
                grouped1 = df.groupby([group_by1])
                first_group_key = list(grouped1.groups.keys())[j]
                first_group = grouped1.get_group(first_group_key)

                # Group by the second column within the first group
                grouped2 = first_group.groupby([group_by2])
                second_group_key = list(grouped2.groups.keys())[i]
                second_group = grouped2.get_group(second_group_key)

                # Calculate the mean of the 'Power' column in the second group
                power_mean = second_group[val].mean()
                power_list_g.append(power_mean)
            power_list.append(power_list_g)
    return power_list

df = [bonferroni_df, holm_df, sgof_df, BH_df, BY_df, storey_df]
get_mean_power_values(df[0], 'n0', 'Effect', 3,3)

# Methods used
methods = ['Bonferroni', 'Holm', 'SGoF', 'BH', 'BY', 'Storey Q']

### 02 - Power over effect (under n0)

power_f = []
for i in range(len(df)):
    power_list = get_mean_power_values(df[i], 'n0', 'Effect', 3,3)
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

titles =[r'$n = 5$', r'$n = 15$', r'$n = 30$']
x_labels = ['Effect size','Effect size','Effect size']
y_labels = ['Power','Power','Power']
plot_power_effect(methods, effect_sizes, powers_n0, powers_n1, powers_n2, titles=titles, x_labels=x_labels,y_labels=y_labels)


### 03 - Power over effect (under s0)
power_f = []
for i in range(len(df)):
    power_list = get_mean_power_values(df[i], 'S0', 'Effect', 2,3)
    power_f.append(power_list)
power_f

power_f2 = []
for i in range(3):
    nth_item = [sublist[i] for sublist in power_f]
    power_f2.append(nth_item)
power_f2

effect_sizes = [0.5, 1.0, 1.5]
powers_s0 = power_f2[0]
powers_s1 = power_f2[1]

titles =[r'$S_0$ = $S_1$ = 0.5', '$S_0$ = $S_1$ = 1.0']
x_labels = ['Effect size','Effect size']
y_labels = ['Power','Power']
plot_power_effect(methods, effect_sizes, powers_s0, powers_s1, titles=titles, x_labels=x_labels, y_labels=y_labels)


### 04 - Power over pi0 (under n0)
# For the n plot
power_f = []
for i in range(len(df)):
    power_list = get_mean_power_values(df[i], 'n0', '1-Pi0', 3,3)
    power_f.append(power_list)
power_f

power_f2 = []
for i in range(3):
    nth_item = [sublist[i] for sublist in power_f]
    power_f2.append(nth_item)
power_f2

# For the n plot
effect_sizes = [0.1, 0.25, 0.50]   # for (1-pi0) here
powers_n0 = power_f2[0]
powers_n1 = power_f2[1]
powers_n2 = power_f2[2]

titles =[r'$n = 5$', r'$n = 15$', r'$n = 30$']
x_labels = [r'$\pi_1$', r'$\pi_1$', r'$\pi_1$']
y_labels = ['Power','Power','Power']
plot_power_effect(methods, effect_sizes, powers_n0, powers_n1, powers_n2, titles=titles, x_labels=x_labels,y_labels=y_labels)


# For the n plot
effect_sizes = [0.1, 0.25, 0.50]   # for (1-pi0) here
powers_n0 = power_f2[0]
powers_n1 = power_f2[1]
powers_n2 = power_f2[2]

titles =[r'$n = 5$', r'$n = 15$', r'$n = 30$']
x_labels = [r'$\pi_1$', r'$\pi_1$', r'$\pi_1$']
y_labels = ['Power','Power','Power']
plot_power_effect(methods, effect_sizes, powers_n0, powers_n1, powers_n2, titles=titles, x_labels=x_labels,y_labels=y_labels)


### 04.5 - Power over pi0 (under n0)
# For the n plot
power_f = []
for i in range(len(df)):
    power_list = get_mean_power_values(df[i], 'n0', 'FDR', 3,3)
    power_f.append(power_list)
power_f

power_f2 = []
for i in range(3):
    nth_item = [sublist[i] for sublist in power_f]
    power_f2.append(nth_item)
power_f2

# For the n plot
effect_sizes = [0.1, 0.25, 0.50]   # for (1-pi0) here
powers_n0 = power_f2[0]
powers_n1 = power_f2[1]
powers_n2 = power_f2[2]

titles =[r'$n = 5$', r'$n = 15$', r'$n = 30$']
x_labels = [r'$\pi_1$', r'$\pi_1$', r'$\pi_1$']
y_labels = ['SD of Power','SD of Power','SD of Power']
plot_power_effect(methods, effect_sizes, powers_n0, powers_n1, powers_n2, titles=titles, x_labels=x_labels,y_labels=y_labels)



uncorrected_df
### 05 - Power over pi0 (under s0)

power_f = []
for i in range(len(df)):
    power_list = get_mean_power_values(df[i], 'S0', '1-Pi0', 2,3)
    power_f.append(power_list)
power_f

power_f2 = []
for i in range(3):
    nth_item = [sublist[i] for sublist in power_f]
    power_f2.append(nth_item)
power_f2

effect_sizes = [0.5, 1.0, 1.5]
powers_s0 = power_f2[0]
powers_s1 = power_f2[1]

titles =[r'$S_0$ = $S_1$ = 0.5', '$S_0$ = $S_1$ = 1.0']
x_labels = [r'$\pi_1$', r'$\pi_1$']
y_labels = ['Power','Power']
plot_power_effect(methods, effect_sizes, powers_s0, powers_s1, titles=titles, x_labels=x_labels, y_labels=y_labels)

### 06 - ROC curves over effect


fpr_f = []
for i in range(len(df)):
    fpr_list = get_mean_power_values(df[i], 'n0', 'Effect', 3, 3,val='FPR')
    fpr_f.append(fpr_list)
fpr_f
fpr_f2 = []
for i in range(3):
    nth_item = [sublist[i] for sublist in fpr_f]
    fpr_f2.append(nth_item)
fpr_f2


def plot_roc(methods, tpr_list, fpr_list):
    plt.figure(figsize=(8, 6))
    for i in range(len(methods)):
        plt.plot(fpr_list[i], tpr_list[i], label=methods[i])
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
    plt.xlabel('False Positive Rate (FPR)', fontname='Times New Roman')
    plt.ylabel('True Positive Rate (TPR)', fontname='Times New Roman')
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontname='Times New Roman')
    plt.legend()
    plt.grid(True)
    plt.show()


methods = ['Bonferroni', 'Holm', 'SGoF', 'BH', 'BY', 'Storey Q']
fpr_list = [[sum(fpr_f2[i][j])/len(fpr_f2[i][j]) for i in range(3)] for j in range(6)]
tpr_list = [[sum(power_f2[i][j])/len(power_f2[i][j]) for i in range(3)] for j in range(6)]

plot_roc(methods, tpr_list, fpr_list)


## 07 - Radar plot to compare metrics

import pandas as pd

bon5 = [all_df.groupby('n0')['Power bonf'].mean().iloc[0], 1-all_df.groupby('n0')['FPR bonf'].mean().iloc[0],1-all_df.groupby('n0')['FDR bonf'].mean().iloc[0],all_df.groupby('n0')['Accuracy bonf'].mean().iloc[0],all_df.groupby('n0')['F1 bonf'].mean().iloc[0]]
holm5 = [all_df.groupby('n0')['Power holm'].mean().iloc[0],1-all_df.groupby('n0')['FPR holm'].mean().iloc[0],1-all_df.groupby('n0')['FDR holm'].mean().iloc[0],all_df.groupby('n0')['Accuracy holm'].mean().iloc[0],all_df.groupby('n0')['F1 holm'].mean().iloc[0]]
sg5 = [all_df.groupby('n0')['Power sg'].mean().iloc[0], 1-all_df.groupby('n0')['FPR sg'].mean().iloc[0],1-all_df.groupby('n0')['FDR sg'].mean().iloc[0],all_df.groupby('n0')['Accuracy sg'].mean().iloc[0],all_df.groupby('n0')['F1 sg'].mean().iloc[0]]
bh5 = [all_df.groupby('n0')['Power bh'].mean().iloc[0], 1-all_df.groupby('n0')['FPR bh'].mean().iloc[0],1-all_df.groupby('n0')['FDR bh'].mean().iloc[0],all_df.groupby('n0')['Accuracy bh'].mean().iloc[0],all_df.groupby('n0')['F1 bh'].mean().iloc[0]]
by5 = [all_df.groupby('n0')['Power by'].mean().iloc[0], 1-all_df.groupby('n0')['FPR by'].mean().iloc[0],1-all_df.groupby('n0')['FDR by'].mean().iloc[0],all_df.groupby('n0')['Accuracy by'].mean().iloc[0],all_df.groupby('n0')['F1 by'].mean().iloc[0]]
Q5 = [all_df.groupby('n0')['Power Q'].mean().iloc[0], 1-all_df.groupby('n0')['FPR Q'].mean().iloc[0],1-all_df.groupby('n0')['FDR Q'].mean().iloc[0],all_df.groupby('n0')['Accuracy Q'].mean().iloc[0],all_df.groupby('n0')['F1 Q'].mean().iloc[0]]
[bon5,holm5,sg5,bh5,by5,Q5]


bon15 = [all_df.groupby('n0')['Power bonf'].mean().iloc[1], 1-all_df.groupby('n0')['FPR bonf'].mean().iloc[1],1-all_df.groupby('n0')['FDR bonf'].mean().iloc[1],all_df.groupby('n0')['Accuracy bonf'].mean().iloc[1],all_df.groupby('n0')['F1 bonf'].mean().iloc[1]]
holm15 = [all_df.groupby('n0')['Power holm'].mean().iloc[1],1-all_df.groupby('n0')['FPR holm'].mean().iloc[1],1-all_df.groupby('n0')['FDR holm'].mean().iloc[1],all_df.groupby('n0')['Accuracy holm'].mean().iloc[1],all_df.groupby('n0')['F1 holm'].mean().iloc[1]]
sg15 = [all_df.groupby('n0')['Power sg'].mean().iloc[1], 1-all_df.groupby('n0')['FPR sg'].mean().iloc[1],1-all_df.groupby('n0')['FDR sg'].mean().iloc[1],all_df.groupby('n0')['Accuracy sg'].mean().iloc[1],all_df.groupby('n0')['F1 sg'].mean().iloc[1]]
bh15 = [all_df.groupby('n0')['Power bh'].mean().iloc[1], 1-all_df.groupby('n0')['FPR bh'].mean().iloc[1],1-all_df.groupby('n0')['FDR bh'].mean().iloc[1],all_df.groupby('n0')['Accuracy bh'].mean().iloc[1],all_df.groupby('n0')['F1 bh'].mean().iloc[1]]
by15 = [all_df.groupby('n0')['Power by'].mean().iloc[1], 1-all_df.groupby('n0')['FPR by'].mean().iloc[1],1-all_df.groupby('n0')['FDR by'].mean().iloc[1],all_df.groupby('n0')['Accuracy by'].mean().iloc[1],all_df.groupby('n0')['F1 by'].mean().iloc[1]]
Q15 = [all_df.groupby('n0')['Power Q'].mean().iloc[1], 1-all_df.groupby('n0')['FPR Q'].mean().iloc[1],1-all_df.groupby('n0')['FDR Q'].mean().iloc[1],all_df.groupby('n0')['Accuracy Q'].mean().iloc[1],all_df.groupby('n0')['F1 Q'].mean().iloc[1]]
[bon15,holm15,sg15,bh15,by15,Q15]

all_df.columns
# Example usage:
methods = ['Bonferroni', 'Holm', 'SGoF', 'BH', 'BY', 'Storey Q']
criteria = ['Power\n(TPR)', 'Specificity\n(TNR)', 'Precision\n(1-FDR)', 'Accuracy', 'F1 score']
title = ''
scores_n5 = [bon5,holm5,sg5,bh5,by5,Q5]
scores_n15 = [bon15,holm15,sg15,bh15,by15,Q15]
plot_radar_plots(methods, criteria, title, scores_n5, scores_n15)






# Example usage:
import numpy as np
methods = ['Bonferroni', 'Holm', 'SGoF', 'BH', 'BY', 'Storey Q']
criteria = ['Power\n(TPR)', 'Specificity\n(TNR)', 'Precision\n(1-FDR)', 'Accuracy', 'F1 score']
title = 'Radar plot of different methods'
scores = np.random.rand(len(methods), len(criteria))  # Example scores


power_f = []
for i in range(len(df)):
    power_list = get_mean_power_values(df[i], 'S0', '1-Pi0', 2,3)
    power_f.append(power_list)
power_f

plot_radar_plots(methods, criteria, title, scores)


fpr_f = []
for i in range(len(df)):
    fpr_list = get_mean_power_values(df[i], 'n0', 'Effect', 3, 3,val='FPR')
    fpr_f.append(fpr_list)
fpr_f
fpr_f2 = []
for i in range(3):
    nth_item = [sublist[i] for sublist in fpr_f]
    fpr_f2.append(nth_item)
fpr_f2

fpr_list = [[sum(fpr_f2[i][j])/len(fpr_f2[i][j]) for i in range(3)] for j in range(6)]
tpr_list = [[sum(power_f2[i][j])/len(power_f2[i][j]) for i in range(3)] for j in range(6)]
