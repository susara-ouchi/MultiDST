### This is the full appication to real world problem

from utils.weighting import weighted_p_list
from MultiDST.bonferroni import bonferroni
from MultiDST.holm import holm
from MultiDST.sgof import sgof_test
from MultiDST.BH import bh_method
from MultiDST.qval import q_value
from MultiDST.BY import BY_method

from utils.visualization import draw_histogram
from utils.visualization import sig_index_plot
from utils.visualization import draw_p_bar_chart
from utils.visualization import plot_heatmap 

import pandas as pd


# Opening the file in read mode
with open('MultiDST/MultiDST - Real Dataset/experimental_pvalues.txt', 'r') as file:
    # Read the entire content of the file
    content = file.read()

# Split the content into lines
lines = content.splitlines()

# Split each line into columns (assuming a space delimiter)
table_data = [line.split() for line in lines]
colnames = lines[0].split()

# Create a DataFrame using pandas
df = pd.DataFrame(table_data[1:], columns=colnames)

P_MPRA = df['p.value.MPRA']
P_STARR = df['p.value.STARR']

###################################### For the full dataset #################################

## plot 01 - Histograms
import matplotlib.pyplot as plt
p_valuesMPRA = P_MPRA.values.astype(float)
p_valuesSTARR = P_STARR.values.astype(float)

# Create subplots with 1 row and 2 columns
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].hist(p_valuesMPRA, bins=50, color='skyblue', edgecolor='navy')
axs[0].set_title('Histogram of MPRA p-values', fontsize=15)
axs[0].set_xlabel('Values', fontsize=15)
axs[0].set_ylabel('Frequency', fontsize=15)
axs[1].hist(p_valuesSTARR, bins=50, color='skyblue', edgecolor='navy', alpha=0.7)
axs[1].set_title('Histogram of STARR p-values', fontsize=15)
axs[1].set_xlabel('Values', fontsize=15)
axs[1].set_ylabel('Frequency', fontsize=15)
plt.tight_layout()
plt.show()

## Plot -2 - Barplot of type
T_CRE = sum(df['Type']=="CRE")
T_Random = sum(df['Type']=="Random")
Total = T_CRE + T_Random

draw_p_bar_chart(['CRE','Random'], [100*T_CRE/Total,100*T_Random/Total], title='Bar chart of Type' , xlabel='Type of element', ylabel='Percentage', border_color='grey')


import matplotlib.pyplot as plt
import seaborn as sns

sizes = [40, 60]  # Percentages for the two classes
labels = ['CRE', 'Random']
colors = ['#2B547E', '#90EE90']
plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title('Type of Element', fontsize=16)
plt.axis('equal')
plt.show()



p_values = p_valuesSTARR
###################### Try 01 - Applying the methods #################################

# 0 - Uncorrected
sig_index = [index for index,p in enumerate(p_values) if p < 0.05]
len(sig_index)
sig_index_plot(p_values, sig_index)

# 1 - Bonferroni
bonf_results = bonferroni(p_values,alpha=0.05, weights = False)
bonf_p, sig_bonf_p = bonf_results[0], bonf_results[1]
sig_index_plot(p_values, sig_bonf_p, pt=1, color = 'blue')

# 2 - Holm
holm_results = holm(p_values,alpha=0.05, weights = False)
holm_p, sig_holm_p = holm_results[0], holm_results[1]
sig_index_plot(p_values, sig_holm_p, pt=2, color = 'green')

# 3 - SGoF
sgof_results = sgof_test(p_values,alpha=0.05, weights = False)
sgof_p, sig_sgof_p = sgof_results[0], sgof_results[1]
sig_index_plot(p_values, sig_sgof_p, pt=2, color = 'green')

# 4 - BH
bh_results = bh_method(p_values,alpha=0.05, weights = False)
bh_p, sig_bh_p = bh_results[0], bh_results[1]
sig_index_plot(p_values, sig_bh_p,pt=2, color = 'green')

# 5 - BY
by_results = BY_method(p_values,alpha=0.05, weights = False)
by_p, sig_by_p = by_results[0], by_results[1]
sig_index_plot(p_values, sig_by_p, pt=2, color = 'green')

# 6 - Qval
q_results = q_value(p_values,alpha=0.05, weights = False)
q, sig_q = q_results[0], q_results[1]
sig_index_plot(p_values, sig_q,pt=2, color = 'green')

print("Uncorrected: ",len(sig_index),
      "\nBonferroni:",len(sig_bonf_p),
      "\nHolm",len(sig_holm_p),
      "\nSGoF",len(sig_sgof_p),
      "\nBH",len(sig_bh_p),
      "\nBY",len(sig_by_p),
      "\nQ-value:",len(sig_q))

methods = ['Bonferroni', 'Holm', 'SGoF', 'BH', 'BY', 'Q value']
sig_indices = [sig_bonf_p, sig_holm_p, sig_sgof_p, sig_bh_p, sig_by_p, sig_q]
plot_heatmap(methods, sig_indices)


############################# Shortlisting p values - Remove Bonferroni ########################

p_values2 = [p_values[i] for i,val in enumerate(p_values) if i not in sig_bonf_p]
p_values = p_values2

draw_histogram(p_values, bins=50, color='skyblue', edgecolor='navy', title='Histogram of MPRA p_values', xlabel='Values', ylabel='Frequency')

###################### Try 02 -  Applying the methods #################################

# 0 - Uncorrected
sig_index = [index for index,p in enumerate(p_values) if p < 0.05]
len(sig_index)

# 1 - Bonferroni
bonf_results = bonferroni(p_values,alpha=0.05, weights = False)
bonf_p, sig_bonf_p = bonf_results[0], bonf_results[1]

# 2 - Holm
holm_results = holm(p_values,alpha=0.05, weights = False)
holm_p, sig_holm_p = holm_results[0], holm_results[1]

# 3 - SGoF
sgof_results = sgof_test(p_values,alpha=0.05, weights = False)
sgof_p, sig_sgof_p = sgof_results[0], sgof_results[1]

# 4 - BH
bh_results = bh_method(p_values,alpha=0.05, weights = False)
bh_p, sig_bh_p = bh_results[0], bh_results[1]

# 5 - BY
by_results = BY_method(p_values,alpha=0.05, weights = False)
by_p, sig_by_p = by_results[0], by_results[1]

# 6 - Qval
q_results = q_value(p_values,alpha=0.05, weights = False)
q, sig_q = q_results[0], q_results[1]

print("Uncorrected: ",len(sig_index),
      "\nBonferroni:",len(sig_bonf_p),
      "\nHolm",len(sig_holm_p),
      "\nSGoF",len(sig_sgof_p),
      "\nBH",len(sig_bh_p),
      "\nBY",len(sig_by_p),
      "\nQ-value:",len(sig_q))

methods = ['Bonferroni', 'Holm', 'SGoF', 'BH', 'BY', 'Q value']
sig_indices = [sig_bonf_p, sig_holm_p, sig_sgof_p, sig_bh_p, sig_by_p, sig_q]
plot_heatmap(methods, sig_indices, title="second")

############################# Iter 3 - Shortlisting p values ########################

p_values3 = [p_values[i] for i,val in enumerate(p_values) if i not in sig_q]
p_values = p_values3

draw_histogram(p_values, bins=50, color='skyblue', edgecolor='navy', title='Histogram of MPRA p_values', xlabel='Values', ylabel='Frequency')
methods = ['Bonferroni', 'Holm', 'SGoF', 'BH', 'BY', 'Q value']


###################### Try 02 -  Applying the methods #################################

# 0 - Uncorrected
sig_index = [index for index,p in enumerate(p_values) if p < 0.05]
len(sig_index)

# 1 - Bonferroni
bonf_results = bonferroni(p_values,alpha=0.05, weights = False)
bonf_p, sig_bonf_p = bonf_results[0], bonf_results[1]

# 2 - Holm
holm_results = holm(p_values,alpha=0.05, weights = False)
holm_p, sig_holm_p = holm_results[0], holm_results[1]

# 3 - SGoF
sgof_results = sgof_test(p_values,alpha=0.05, weights = False)
sgof_p, sig_sgof_p = sgof_results[0], sgof_results[1]

# 4 - BH
bh_results = bh_method(p_values,alpha=0.05, weights = False)
bh_p, sig_bh_p = bh_results[0], bh_results[1]

# 5 - BY
by_results = BY_method(p_values,alpha=0.05, weights = False)
by_p, sig_by_p = by_results[0], by_results[1]

# 6 - Qval
q_results = q_value(p_values,alpha=0.05, weights = False)
q, sig_q = q_results[0], q_results[1]

print("Uncorrected: ",len(sig_index),
      "\nBonferroni:",len(sig_bonf_p),
      "\nHolm",len(sig_holm_p),
      "\nSGoF",len(sig_sgof_p),
      "\nBH",len(sig_bh_p),
      "\nBY",len(sig_by_p),
      "\nQ-value:",len(sig_q))

sig_indices = [sig_bonf_p, sig_holm_p, sig_sgof_p, sig_bh_p, sig_by_p, sig_q]
plot_heatmap(methods, sig_indices, title="third")

############################# Iter 4 - Shortlisting p values ########################

p_values4 = [p_values[i] for i,val in enumerate(p_values) if i not in sig_sgof_p]
p_values = p_values4

draw_histogram(p_values, bins=50, color='skyblue', edgecolor='navy', title='Histogram of MPRA p_values', xlabel='Values', ylabel='Frequency')

###################### Try 02 -  Applying the methods #################################

# 0 - Uncorrected
sig_index = [index for index,p in enumerate(p_values) if p < 0.05]
len(sig_index)

# 1 - Bonferroni
bonf_results = bonferroni(p_values,alpha=0.05, weights = False)
bonf_p, sig_bonf_p = bonf_results[0], bonf_results[1]

# 2 - Holm
holm_results = holm(p_values,alpha=0.05, weights = False)
holm_p, sig_holm_p = holm_results[0], holm_results[1]

# 3 - SGoF
sgof_results = sgof_test(p_values,alpha=0.05, weights = False)
sgof_p, sig_sgof_p = sgof_results[0], sgof_results[1]

# 4 - BH
bh_results = bh_method(p_values,alpha=0.05, weights = False)
bh_p, sig_bh_p = bh_results[0], bh_results[1]

# 5 - BY
by_results = BY_method(p_values,alpha=0.05, weights = False)
by_p, sig_by_p = by_results[0], by_results[1]

# 6 - Qval
q_results = q_value(p_values,alpha=0.05, weights = False)
q, sig_q = q_results[0], q_results[1]

print("Uncorrected: ",len(sig_index),
      "\nBonferroni:",len(sig_bonf_p),
      "\nHolm",len(sig_holm_p),
      "\nSGoF",len(sig_sgof_p),
      "\nBH",len(sig_bh_p),
      "\nBY",len(sig_by_p),
      "\nQ-value:",len(sig_q))

sig_indices = [sig_bonf_p, sig_holm_p, sig_sgof_p, sig_bh_p, sig_by_p, sig_q]
plot_heatmap(methods, sig_indices, title="fourth")

############################# Iter 5 - Shortlisting p values ########################

p_values5 = [p_values[i] for i,val in enumerate(p_values) if i not in sig_sgof_p]
p_values = p_values5

draw_histogram(p_values, bins=50, color='skyblue', edgecolor='navy', title='Histogram of MPRA p_values', xlabel='Values', ylabel='Frequency')

###################### Try 02 -  Applying the methods #################################

# 0 - Uncorrected
sig_index = [index for index,p in enumerate(p_values) if p < 0.05]
len(sig_index)

# 1 - Bonferroni
bonf_results = bonferroni(p_values,alpha=0.05, weights = False)
bonf_p, sig_bonf_p = bonf_results[0], bonf_results[1]

# 2 - Holm
holm_results = holm(p_values,alpha=0.05, weights = False)
holm_p, sig_holm_p = holm_results[0], holm_results[1]

# 3 - SGoF
sgof_results = sgof_test(p_values,alpha=0.05, weights = False)
sgof_p, sig_sgof_p = sgof_results[0], sgof_results[1]

# 4 - BH
bh_results = bh_method(p_values,alpha=0.05, weights = False)
bh_p, sig_bh_p = bh_results[0], bh_results[1]

# 5 - BY
by_results = BY_method(p_values,alpha=0.05, weights = False)
by_p, sig_by_p = by_results[0], by_results[1]

# 6 - Qval
q_results = q_value(p_values,alpha=0.05, weights = False)
q, sig_q = q_results[0], q_results[1]

print("Uncorrected: ",len(sig_index),
      "\nBonferroni:",len(sig_bonf_p),
      "\nHolm",len(sig_holm_p),
      "\nSGoF",len(sig_sgof_p),
      "\nBH",len(sig_bh_p),
      "\nBY",len(sig_by_p),
      "\nQ-value:",len(sig_q))

sig_indices = [sig_bonf_p, sig_holm_p, sig_sgof_p, sig_bh_p, sig_by_p, sig_q]
plot_heatmap(methods, sig_indices, title="fifth")

# Now that all the methods have shortlisted the p-values, take a look at the uncorrected ones
[p_values5[i] for i in sig_index]

#As we can see, most of the values are close to the cut off of 0.05, hence the methods have done their bit








import matplotlib.pyplot as plt

# Data
data_labels = ['Available', 'Missing']
data_values = [90, 10]

# Plot
fig, ax = plt.subplots()
bars = ax.barh(data_labels, data_values, color=['lightblue', 'lightgreen'])

# Add data labels
for bar in bars:
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height()/2, f'{width}%', ha='left', va='center')

# Customize plot
ax.set_xlim(0, 100)
ax.set_xlabel('Percentage')
ax.set_title('Availability of Dataset')

# Show plot
plt.show()
