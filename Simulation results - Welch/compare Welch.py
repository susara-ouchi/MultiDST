############################## Diagnostics #################################
import matplotlib.pyplot as plt
import pandas as pd

from visualization import plot_power_effect
from visualization import plot_roc
from visualization import plot_radar_plots

### 00 - Loading the datasets
uncorrected_df = pd.read_csv(r'MultiDST\Simulated datasets 2\uncorrected_sim_results.csv')
bonferroni_df = pd.read_csv(r'MultiDST\Simulated datasets 2\bonferroni_sim_results.csv')
holm_df = pd.read_csv(r'MultiDST\Simulated datasets 2\holm_sim_results.csv')
sgof_df = pd.read_csv(r'MultiDST\Simulated datasets 2\sgof_sim_results.csv')
BH_df = pd.read_csv(r'MultiDST\Simulated datasets 2\bh_sim_results.csv')
BY_df = pd.read_csv(r'MultiDST\Simulated datasets 2\by_sim_results.csv')
storey_df =pd.read_csv(r'MultiDST\Simulated datasets 2\storeyQ_sim_results.csv')
all_df =pd.read_csv(r'MultiDST\Simulated datasets 2\all_df_3.csv')

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
get_mean_power_values(df[0], 'n0', 'Effect', 2,3)

# Methods used
methods = ['Bonferroni', 'Holm', 'SGoF', 'BH', 'BY', 'Storey Q']

### 02 - Power over effect (under n0)

power_f = []
for i in range(len(df)):
    power_list = get_mean_power_values(df[i], 'n0', 'Effect', 2,3)
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
plot_power_effect(methods, effect_sizes, powers_n0, powers_n1, titles=titles, x_labels=x_labels,y_labels=y_labels)


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




bon5 = [all_df.groupby('n0')['Power bonf'].mean().iloc[0], 1-all_df.groupby('n0')['FPR bonf'].mean().iloc[0],1-all_df.groupby('n0')['FDR bonf'].mean().iloc[0],all_df.groupby('n0')['Accuracy bonf'].mean().iloc[0],all_df.groupby('n0')['F1 bonf'].mean().iloc[0]]
holm5 = [all_df.groupby('n0')['Power holm'].mean().iloc[0],1-all_df.groupby('n0')['FPR holm'].mean().iloc[0],1-all_df.groupby('n0')['FDR holm'].mean().iloc[0],all_df.groupby('n0')['Accuracy holm'].mean().iloc[0],all_df.groupby('n0')['F1 holm'].mean().iloc[0]]
sg5 = [all_df.groupby('n0')['Power sg'].mean().iloc[0], 1-all_df.groupby('n0')['FPR sg'].mean().iloc[0],1-all_df.groupby('n0')['FDR sg'].mean().iloc[0],all_df.groupby('n0')['Accuracy sg'].mean().iloc[0],all_df.groupby('n0')['F1 sg'].mean().iloc[0]]
bh5 = [all_df.groupby('n0')['Power bh'].mean().iloc[0], 1-all_df.groupby('n0')['FPR bh'].mean().iloc[0],1-all_df.groupby('n0')['FDR bh'].mean().iloc[0],all_df.groupby('n0')['Accuracy bh'].mean().iloc[0],all_df.groupby('n0')['F1 bh'].mean().iloc[0]]
by5 = [all_df.groupby('n0')['Power by'].mean().iloc[0], 1-all_df.groupby('n0')['FPR by'].mean().iloc[0],1-all_df.groupby('n0')['FDR by'].mean().iloc[0],all_df.groupby('n0')['Accuracy by'].mean().iloc[0],all_df.groupby('n0')['F1 by'].mean().iloc[0]]
Q5 = [all_df.groupby('n0')['Power Q'].mean().iloc[0], 1-all_df.groupby('n0')['FPR Q'].mean().iloc[0],1-all_df.groupby('n0')['FDR Q'].mean().iloc[0],all_df.groupby('n0')['Accuracy Q'].mean().iloc[0],all_df.groupby('n0')['F1 Q'].mean().iloc[0]]
[bon5,holm5,sg5,bh5,by5,Q5]


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


################## Heatmap

all_df

### HEATMAP 01 - By s0, s1 ###

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Data

s0s0 = [all_df.groupby(['S0','S1'])['Power bonf'].mean().iloc[0],all_df.groupby(['S0','S1'])['Power holm'].mean().iloc[0],all_df.groupby(['S0','S1'])['Power sg'].mean().iloc[0],all_df.groupby(['S0','S1'])['Power bh'].mean().iloc[0],all_df.groupby(['S0','S1'])['Power by'].mean().iloc[0],all_df.groupby(['S0','S1'])['Power Q'].mean().iloc[0]]
s0s1 = [all_df.groupby(['S0','S1'])['Power bonf'].mean().iloc[1], all_df.groupby(['S0','S1'])['Power holm'].mean().iloc[1], all_df.groupby(['S0','S1'])['Power sg'].mean().iloc[1], all_df.groupby(['S0','S1'])['Power bh'].mean().iloc[1], all_df.groupby(['S0','S1'])['Power by'].mean().iloc[1], all_df.groupby(['S0','S1'])['Power Q'].mean().iloc[1]]
s1s0 = [all_df.groupby(['S0','S1'])['Power bonf'].mean().iloc[2], all_df.groupby(['S0','S1'])['Power holm'].mean().iloc[2], all_df.groupby(['S0','S1'])['Power sg'].mean().iloc[2], all_df.groupby(['S0','S1'])['Power bh'].mean().iloc[2], all_df.groupby(['S0','S1'])['Power by'].mean().iloc[2], all_df.groupby(['S0','S1'])['Power Q'].mean().iloc[2]]
s1s1 = [all_df.groupby(['S0','S1'])['Power bonf'].mean().iloc[3], all_df.groupby(['S0','S1'])['Power holm'].mean().iloc[3], all_df.groupby(['S0','S1'])['Power sg'].mean().iloc[3], all_df.groupby(['S0','S1'])['Power bh'].mean().iloc[3], all_df.groupby(['S0','S1'])['Power by'].mean().iloc[3], all_df.groupby(['S0','S1'])['Power Q'].mean().iloc[3]]


methods = ['Bonferroni', 'Holm', 'SGoF', 'BH', 'BY', 'Storey Q']
parameters = ['S0 = 1.0\nS1 = 1.0', 'S0 = 0.5\nS1 = 1.0', 'S0 = 1.0\nS1 = 0.5','S0 = 0.5\nS1 = 0.5']
num_parameters = len(parameters)
num_methods = len(methods)
data = [s1s1,s0s1,s1s0,s0s0] 

# Create heatmap
sns.heatmap(data, cmap='Blues', annot=True, fmt=".2f")

# Customize labels and title
plt.xlabel('Methods', fontname='Times New Roman')
plt.ylabel('Sample Standard deviations',fontname='Times New Roman')
#plt.title('Comparison of Power Values across combinations of standard deviations')

# Set x-axis tick labels
plt.xticks(ticks=np.arange(num_methods) + 0.5, labels=methods)

# Set y-axis tick labels
plt.yticks(ticks=np.arange(num_parameters) + 0.5, labels=parameters, rotation=0)

plt.tight_layout()
plt.show()



### HEATMAP 02 - By n0, n1 ###

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Data

n0n0 = [all_df.groupby(['n0','n1'])['Power bonf'].mean().iloc[0],all_df.groupby(['n0','n1'])['Power holm'].mean().iloc[0],all_df.groupby(['n0','n1'])['Power sg'].mean().iloc[0],all_df.groupby(['n0','n1'])['Power bh'].mean().iloc[0],all_df.groupby(['n0','n1'])['Power by'].mean().iloc[0],all_df.groupby(['n0','n1'])['Power Q'].mean().iloc[0]]
n0n1 = [all_df.groupby(['n0','n1'])['Power bonf'].mean().iloc[1], all_df.groupby(['n0','n1'])['Power holm'].mean().iloc[1], all_df.groupby(['n0','n1'])['Power sg'].mean().iloc[1], all_df.groupby(['n0','n1'])['Power bh'].mean().iloc[1], all_df.groupby(['n0','n1'])['Power by'].mean().iloc[1], all_df.groupby(['n0','n1'])['Power Q'].mean().iloc[1]]
n1n0 = [all_df.groupby(['n0','n1'])['Power bonf'].mean().iloc[2], all_df.groupby(['n0','n1'])['Power holm'].mean().iloc[2], all_df.groupby(['n0','n1'])['Power sg'].mean().iloc[2], all_df.groupby(['n0','n1'])['Power bh'].mean().iloc[2], all_df.groupby(['n0','n1'])['Power by'].mean().iloc[2], all_df.groupby(['n0','n1'])['Power Q'].mean().iloc[2]]
n1n1 = [all_df.groupby(['n0','n1'])['Power bonf'].mean().iloc[3], all_df.groupby(['n0','n1'])['Power holm'].mean().iloc[3], all_df.groupby(['n0','n1'])['Power sg'].mean().iloc[3], all_df.groupby(['n0','n1'])['Power bh'].mean().iloc[3], all_df.groupby(['n0','n1'])['Power by'].mean().iloc[3], all_df.groupby(['n0','n1'])['Power Q'].mean().iloc[3]]

methods = ['Bonferroni', 'Holm', 'SGoF', 'BH', 'BY', 'Storey Q']
parameters = ['n0 = 15\nn1 = 15', 'n0 = 5\nn1 = 15', 'n0 = 15\nn1 = 5','n0 = 5\nn1 = 5']
num_parameters = len(parameters)
num_methods = len(methods)
data = [n1n1,n0n1,n1n0,n0n0] 

# Create heatmap
sns.heatmap(data, cmap='BuPu', annot=True, fmt=".2f")

# Customize labels and title
plt.xlabel('Methods', fontname='Times New Roman')
plt.ylabel('Sample sizes',fontname='Times New Roman')
#plt.title('Comparison of Power Values across combinations of standard deviations')

# Set x-axis tick labels
plt.xticks(ticks=np.arange(num_methods) + 0.5, labels=methods)

# Set y-axis tick labels
plt.yticks(ticks=np.arange(num_parameters) + 0.5, labels=parameters, rotation=0)

plt.tight_layout()
plt.show()



# Plotting the power over effect
all_df
n0n0 = [all_df.groupby(['n0','n1','Effect'])['Power bonf'].mean().iloc[0],all_df.groupby(['n0','n1'])['Power holm'].mean().iloc[0],all_df.groupby(['n0','n1'])['Power sg'].mean().iloc[0],all_df.groupby(['n0','n1'])['Power bh'].mean().iloc[0],all_df.groupby(['n0','n1'])['Power by'].mean().iloc[0],all_df.groupby(['n0','n1'])['Power Q'].mean().iloc[0]]
n0n1 = [all_df.groupby(['n0','n1'])['Power bonf'].mean().iloc[1], all_df.groupby(['n0','n1'])['Power holm'].mean().iloc[1], all_df.groupby(['n0','n1'])['Power sg'].mean().iloc[1], all_df.groupby(['n0','n1'])['Power bh'].mean().iloc[1], all_df.groupby(['n0','n1'])['Power by'].mean().iloc[1], all_df.groupby(['n0','n1'])['Power Q'].mean().iloc[1]]
n1n0 = [all_df.groupby(['n0','n1'])['Power bonf'].mean().iloc[2], all_df.groupby(['n0','n1'])['Power holm'].mean().iloc[2], all_df.groupby(['n0','n1'])['Power sg'].mean().iloc[2], all_df.groupby(['n0','n1'])['Power bh'].mean().iloc[2], all_df.groupby(['n0','n1'])['Power by'].mean().iloc[2], all_df.groupby(['n0','n1'])['Power Q'].mean().iloc[2]]
n1n1 = [all_df.groupby(['n0','n1'])['Power bonf'].mean().iloc[3], all_df.groupby(['n0','n1'])['Power holm'].mean().iloc[3], all_df.groupby(['n0','n1'])['Power sg'].mean().iloc[3], all_df.groupby(['n0','n1'])['Power bh'].mean().iloc[3], all_df.groupby(['n0','n1'])['Power by'].mean().iloc[3], all_df.groupby(['n0','n1'])['Power Q'].mean().iloc[3]]

all_df['n0'][:12]
all_df['n1'][:12]

list1 = ['Power bonf','Power holm','Power sg','Power bh','Power by','Power Q']
[all_df.groupby(['n0','n1','Effect'])[list[0]].mean().iloc[:3].to_list(),
 all_df.groupby(['n0','n1','Effect'])[list[1]].mean().iloc[:3].to_list()]

data = [[all_df.groupby(['n0', 'n1', 'Effect'])[column].mean().iloc[:3].tolist() for column in list],
        [all_df.groupby(['n0', 'n1', 'Effect'])[column].mean().iloc[3:6].tolist() for column in list],
        [all_df.groupby(['n0', 'n1', 'Effect'])[column].mean().iloc[6:9].tolist() for column in list],
        [all_df.groupby(['n0', 'n1', 'Effect'])[column].mean().iloc[9:12].tolist() for column in list]]



import matplotlib.pyplot as plt

def plot_power_effect(methods, effect_sizes, powers_s0, powers_s1, powers_s2=None, titles=None, x_labels=None, y_labels=None):
    num_plots = 4 if powers_s2 is not None else 3
    plt.figure(figsize=(8, 8))

    # Define colors and markers
    colors = ['black', 'red', 'purple', 'tomato', 'mediumseagreen', 'navy', 'magenta']
    markers = ['o', 's', '^', 'v', 'D', '*', 'X']

    # Plot for s = 0.5 / n = 5
    plt.subplot(2, 2, 1)
    for i in range(len(methods)):
        plt.plot(effect_sizes, powers_s0[i], label=methods[i], color=colors[i % len(colors)], marker=markers[i % len(markers)])
    plt.xlabel(x_labels[0] if x_labels else 'Effect Size', fontname='Times New Roman')
    plt.ylabel(y_labels[0] if y_labels else 'Power', fontname='Times New Roman')
    plt.title(titles[0] if titles else 'Power vs. Effect Size (S0 = 0.5)', fontname='Times New Roman', fontsize=15)
    plt.ylim(0, 1.0)  # Set y-axis limits
    plt.legend(loc='upper left', prop={'family': 'Times New Roman'}, fontsize=10)
    plt.grid(True)

    # Plot for S = 1.0 / n = 15
    plt.subplot(2, 2, 2)
    for i in range(len(methods)):
        plt.plot(effect_sizes, powers_s1[i], label=methods[i], color=colors[i % len(colors)], marker=markers[i % len(markers)])
    plt.xlabel(x_labels[1] if x_labels else 'Effect Size', fontname='Times New Roman')
    plt.ylabel(y_labels[1] if y_labels else 'Power', fontname='Times New Roman')
    plt.title(titles[1] if titles else 'Power vs. Effect Size (S1 = 1.0)', fontname='Times New Roman', fontsize=15)
    plt.ylim(0, 1.0)  # Set y-axis limits
    plt.legend(loc='upper left', prop={'family': 'Times New Roman'}, fontsize=10)
    plt.grid(True)

    # Plot for S = 1.5 / n = 30
    plt.subplot(2, 2, 3)
    for i in range(len(methods)):
        plt.plot(effect_sizes, powers_s2[i], label=methods[i], color=colors[i % len(colors)], marker=markers[i % len(markers)])
    plt.xlabel(x_labels[2] if x_labels else 'Effect Size', fontname='Times New Roman')
    plt.ylabel(y_labels[2] if y_labels else 'Power', fontname='Times New Roman')
    plt.title(titles[2] if titles else 'Power vs. Effect Size (S2 = 1.5)', fontname='Times New Roman', fontsize=15)
    plt.ylim(0, 1.0)  # Set y-axis limits
    plt.legend(loc='upper left', prop={'family': 'Times New Roman'}, fontsize=10)
    plt.grid(True)

    # Plot for additional data, if available
    if num_plots == 4:
        plt.subplot(2, 2, 4)
        # Add your plotting logic for the fourth subplot here

    plt.tight_layout()
    plt.show()

# Example usage:
methods = ['Method A', 'Method B', 'Method C']
effect_sizes = [0.1, 0.2, 0.3, 0.4]
powers_s0 = [[0.2, 0.3, 0.4, 0.5], [0.3, 0.4, 0.5, 0.6], [0.4, 0.5, 0.6, 0.7],[0.4, 0.5, 0.6, 0.7]]
powers_s1 = [[0.5, 0.6, 0.7, 0.8], [0.6, 0.7, 0.8, 0.9], [0.7, 0.8, 0.9, 0.95],[0.4, 0.5, 0.6, 0.7]]
powers_s2 = [[0.7, 0.8, 0.9, 0.95], [0.8, 0.85, 0.9, 0.95], [0.9, 0.92, 0.94, 0.96],[0.4, 0.5, 0.6, 0.7]]
titles = ['Power vs. Effect Size (S0 = 0.5)', 'Power vs. Effect Size (S1 = 1.0)', 'Power vs. Effect Size (S2 = 1.5)']
x_labels = ['Effect Size (S0 = 0.5)', 'Effect Size (S1 = 1.0)', 'Effect Size (S2 = 1.5)']
y_labels = ['Power', 'Power', 'Power']

plot_power_effect(methods, effect_sizes, powers_s0, powers_s1, powers_s2, titles, x_labels, y_labels)
