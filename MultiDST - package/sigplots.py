
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

import pandas as pd
import numpy as np


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

# Convert scientific notation strings to floats
P_MPRA = [float(val) for val in P_MPRA]
P_STARR = [float(val) for val in P_STARR]

p_values = P_STARR

# 1 - Bonferroni
bonf_results = bonferroni(p_values,alpha=0.05, weights = False)
bonf_p, sig_bonf_p = bonf_results[0], bonf_results[1]
#sig_index_plot(p_values, sig_bonf_p, pt=1, color = 'blue')

# 2 - Holm
holm_results = holm(p_values,alpha=0.05, weights = False)
holm_p, sig_holm_p = holm_results[0], holm_results[1]
# sig_index_plot(p_values, sig_holm_p, pt=2, color = 'green')

# 3 - SGoF
sgof_results = sgof_test(p_values,alpha=0.05, weights = False)
sgof_p, sig_sgof_p = sgof_results[0], sgof_results[1]
# sig_index_plot(p_values, sig_sgof_p, pt=2, color = 'green')

# 4 - BH
bh_results = bh_method(p_values,alpha=0.05, weights = False)
bh_p, sig_bh_p = bh_results[0], bh_results[1]
# sig_index_plot(p_values, sig_bh_p,pt=2, color = 'green')

# 5 - BY
by_results = BY_method(p_values,alpha=0.05, weights = False)
by_p, sig_by_p = by_results[0], by_results[1]
# sig_index_plot(p_values, sig_by_p, pt=2, color = 'green')

# 6 - Qval
q_results = q_value(p_values,alpha=0.05, weights = False)
q, sig_q = q_results[0], q_results[1]
# sig_index_plot(p_values, sig_q,pt=2, color = 'green')


### Weighted MPRA
random_ind = df.loc[df['Type'] == "Random"].index

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def plot_heatmap(methods, sig_indices):
    # Create a matrix to represent the selected points for each method
    max_index = max(max(indices) for indices in sig_indices)
    matrix = np.zeros((len(methods), max_index))

    # Fill the matrix with 1 where a method selected a point
    for i, indices in enumerate(sig_indices):
        for idx in indices:
            matrix[i, idx - 1] = 1  # Subtract 1 to align with 0-based indexing

    # Plot the heatmap
    plt.figure(figsize=(12, 6))
    plt.imshow(matrix, cmap='Blues', aspect='auto', interpolation='nearest')

    # Customize the plot
    plt.xlabel('Points', fontname='Times New Roman')
    plt.ylabel('Methods', fontname='Times New Roman')
    plt.title('Selected Points by Methods', fontname='Times New Roman')
    plt.yticks(np.arange(len(methods)), methods)

    # Add a legend box
    legend_box = Rectangle((0, 0), 1, 1, linewidth=1, edgecolor='none', facecolor='midnightblue', label='Significant')
    plt.legend(handles=[legend_box], loc='upper right')

    plt.show()

# Example usage
methods = ['Bonferroni', 'Holm', 'SGoF', 'BH', 'BY', 'Q value']
sig_indices = [sig_bonf_p, sig_holm_p, sig_sgof_p, sig_bh_p, sig_by_p, sig_q]
plot_heatmap(methods, sig_indices)

methods = ['Bonferroni', 'Holm', 'SGoF', 'BH', 'BY', 'Q value', 'Random']
sig_indices = [sig_bonf_p, sig_holm_p, sig_sgof_p, sig_bh_p, sig_by_p, sig_q, random_ind]
plot_heatmap(methods, sig_indices)
