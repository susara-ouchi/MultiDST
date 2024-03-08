def draw_histogram(data, bins=10, color='skyblue', edgecolor='black', title='Histogram', xlabel='Values', ylabel='Frequency'):
    import matplotlib.pyplot as plt
    import numpy as np
    """
    Draw a histogram using user-defined function.

    Parameters:
        data (array-like): Input data for the histogram.
        bins (int or array, optional): Number of bins or bin edges. Default is 10.
        color (str, optional): Color of the bars. Default is 'skyblue'.
        edgecolor (str, optional): Color of the bar edges. Default is 'black'.
        title (str, optional): Title of the chart. Default is 'Histogram'.
        xlabel (str, optional): Label for the x-axis. Default is 'Values'.
        ylabel (str, optional): Label for the y-axis. Default is 'Frequency'.
    """
    plt.hist(data, bins=bins, color=color, edgecolor=edgecolor)

    # Customize the chart
    plt.title(title, fontname='Times New Roman')
    plt.xlabel(xlabel,fontname='Times New Roman')
    plt.ylabel(ylabel,fontname='Times New Roman')

    # Show the chart
    plt.show()

################################################################################

def sig_index_plot_1(p_values, sig_index,non_sig_index):

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # Sample data - replace this with your actual data
    sig_indices = sig_index
    non_sig_indices = non_sig_index
    methods = ['Method 1', 'Method 2', 'Method 3', 'Method 4', 'Method 5', 'Method 6']  # List of methods

    # Plotting
    plt.figure(figsize=(10, 6))

    # Define marker size
    marker_size = 4

    # Plot group A (significant)
    for i, method in enumerate(methods):
        plt.scatter([index for index in sig_indices], 
                    [i] * len([index for index in sig_indices]), 
                    color='green', s=marker_size)

    # Plot group B (non-significant)
    for i, method in enumerate(methods):
        plt.scatter([index for index in non_sig_indices], 
                    [i] * len([index for index in non_sig_indices]), 
                    color='red', s=marker_size)

    plt.yticks(range(len(methods)), methods)  # Set y-axis ticks to method names
    plt.xlabel('Observation Index')
    plt.ylabel('Methods')
    plt.title('Observations by Correction Method and Group')

    # Create custom legend handles with colors
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=4),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=4),
    ]

    # Display the legend with custom handles and labels
    plt.legend(legend_handles, ['Significant', 'Non-significant'], loc='upper right')
    plt.grid(False)
    plt.show()


##########################################################################################
    
def sig_index_plot(p_values, sig_index):
    p_index = [i for i,p in enumerate(p_values)]

    import matplotlib.pyplot as plt

    # Create lists for x and y values
    x_values = list(range(1, len(p_values)+1))
    y_values = [2 if i in sig_index else 1 for i in p_index]  # Alternate between 1 and 2 for y-values

    # Plotting
    plt.figure(figsize=(10, 6))

    # Plot the line connecting all points
    plt.plot(x_values, y_values, color='green', marker='o', linestyle='-')

    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.title('Line Plot of significance over p value indices')
    plt.grid(True)
    plt.show()

p_values = [0.1,0.2,0.05,0.6,0.1,0.1,0.05,0.04,0.006,0.7,0.4]
sig_index = [0,2,3,5]

sig_index_plot(p_values, sig_index)

#############################################################################################

def group_line_plot(df_select, g_var,var1,var2): 
    import pandas as pd
    import matplotlib.pyplot as plt

    # Grouping the DataFrame by 'pi0'
    grouped = df_select.groupby(g_var)

    # Plotting each group separately
    for name, group in grouped:
        plt.plot(group[var1], group[var2], label=f'{g_var} = {name}')

    # Adding labels, title, and legend
    plt.xlabel(var1)
    plt.ylabel(var2)
    plt.title(f'Line Plot of {var2} over {var1}')
    plt.grid(True)
    plt.legend()
    plt.show()
###########################################################################################################

def draw_bar_chart(categories, values, title='Bar Chart', xlabel='Categories', ylabel='Values'):
    import matplotlib.pyplot as plt
    """
    Draw a bar chart using user-defined function.

    Parameters:
        categories (list): List of category labels.
        values (list): List of corresponding values for each category.
        title (str): Title of the chart (default is 'Bar Chart').
        xlabel (str): Label for the x-axis (default is 'Categories').
        ylabel (str): Label for the y-axis (default is 'Values').
    """
    colors = [plt.cm.viridis(120),plt.cm.viridis(50)]
    plt.bar(categories, values, color=colors)
    plt.title(title,fontname='Times New Roman')
    plt.xlabel(xlabel,fontname='Times New Roman')
    #plt.yticks(range(0, 110,10)) #for percentage
    plt.ylabel(ylabel,fontname='Times New Roman')
    plt.show()


##############################################################################

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

# Example data for TPR and FPR for 7 different methods
methods = ['Method 1', 'Method 2', 'Method 3', 'Method 4', 'Method 5', 'Method 6', 'Method 7']
tpr_list = [[0.2, 0.4, 0.6, 0.8, 1.0], [0.3, 0.5, 0.7, 0.85, 0.95], [0.1, 0.3, 0.5, 0.75, 0.9],
            [0.15, 0.35, 0.55, 0.75, 0.85], [0.25, 0.45, 0.65, 0.8, 0.95], [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.05, 0.1, 0.15, 0.2, 0.25]]
fpr_list = [[0.1, 0.2, 0.3, 0.4, 0.5], [0.15, 0.25, 0.35, 0.45, 0.55], [0.05, 0.15, 0.25, 0.35, 0.45],
            [0.1, 0.2, 0.3, 0.4, 0.5], [0.08, 0.18, 0.28, 0.38, 0.48], [0.05, 0.1, 0.15, 0.2, 0.25],
            [0.01, 0.05, 0.1, 0.15, 0.2]]

plot_roc(methods, tpr_list, fpr_list)


#########################################################################################

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
methods = ['Bonferroni', 'Holm', 'SGoF', 'BH', 'BY', 'Storey Q']

def plot_power_effect(methods, effect_sizes, powers_s0, powers_s1, powers_s2=None, titles=None, x_labels=None, y_labels=None):
    num_plots = 2 if powers_s2 is None else 3
    plt.figure(figsize=(5*num_plots, 5))

    # Define colors and markers
    colors = ['black', 'red', 'purple', 'tomato', 'mediumseagreen', 'navy', 'magenta']
    markers = ['o', 's', '^', 'v', 'D', '*','X']

    # Plot for s = 0.5 / n = 5
    plt.subplot(1, num_plots, 1)
    for i in range(len(methods)):
        plt.plot(effect_sizes, powers_s0[i], label=methods[i], color=colors[i % len(colors)], marker=markers[i % len(markers)])
    plt.xlabel(x_labels[0] if x_labels else 'Effect Size', fontname='Times New Roman')
    plt.ylabel(y_labels[0] if y_labels else 'Power', fontname='Times New Roman')
    plt.title(titles[0] if titles else 'Power vs. Effect Size (S0 = 0.5)', fontname='Times New Roman')
    plt.ylim(0, 1.0)  # Set y-axis limits
    plt.legend(loc='upper left', prop={'family': 'Times New Roman'})
    plt.grid(True)

    # Plot for S = 1.0 / n = 15
    plt.subplot(1, num_plots, 2)
    for i in range(len(methods)):
        plt.plot(effect_sizes, powers_s1[i], label=methods[i], color=colors[i % len(colors)], marker=markers[i % len(markers)])
    plt.xlabel(x_labels[1] if x_labels else 'Effect Size', fontname='Times New Roman')
    plt.ylabel(y_labels[1] if y_labels else 'Power', fontname='Times New Roman')
    plt.title(titles[1] if titles else 'Power vs. Effect Size (S1 = 1.0)', fontname='Times New Roman')
    plt.ylim(0, 1.0)  # Set y-axis limits
    plt.grid(True)

    # Plot for n = 30 
    if num_plots == 3:
        plt.subplot(1, num_plots, 3)
        for i in range(len(methods)):
            plt.plot(effect_sizes, powers_s2[i], label=methods[i], color=colors[i % len(colors)], marker=markers[i % len(markers)])
        plt.xlabel(x_labels[2] if x_labels else 'Effect Size', fontname='Times New Roman')
        plt.ylabel(y_labels[2] if y_labels else 'Power', fontname='Times New Roman')
        plt.title(titles[2] if titles else 'Power vs. Effect Size (S2 = 1.5)', fontname='Times New Roman')
        plt.ylim(0, 1.0)  # Set y-axis limits
        plt.grid(True)

    plt.tight_layout()
    plt.show()




# Example data for effect sizes and powers for 7 different methods
methods = ['Method 1', 'Method 2', 'Method 3', 'Method 4', 'Method 5', 'Method 6', 'Method 7']
effect_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]
powers_s0 = [[0.2, 0.4, 0.6, 0.8, 1.0], [0.3, 0.5, 0.7, 0.85, 0.95], [0.1, 0.3, 0.5, 0.75, 0.9],
             [0.2, 0.4, 0.6, 0.8, 1.0],[0.2, 0.4, 0.6, 0.8, 1.0],[0.2, 0.4, 0.6, 0.8, 1.0],[0.2, 0.4, 0.6, 0.8, 1.0]]
powers_s1 = [[0.2, 0.4, 0.6, 0.8, 1.0], [0.3, 0.5, 0.7, 0.85, 0.95], [0.1, 0.3, 0.5, 0.75, 0.9],
             [0.2, 0.4, 0.6, 0.8, 1.0],[0.2, 0.4, 0.6, 0.8, 1.0],[0.2, 0.4, 0.6, 0.8, 1.0],[0.2, 0.4, 0.6, 0.8, 1.0]]

titles =['Custom Title 1', 'Custom Title 2', 'Custom Title 3']
x_labels = ['Effect size','Effect size','Effect size']
y_labels = ['Power','Power','Power']
plot_power_effect(methods, effect_sizes, powers_s0, powers_s1, titles=titles)


# For the n plot
methods = ['Method 1', 'Method 2', 'Method 3', 'Method 4', 'Method 5', 'Method 6', 'Method 7']
effect_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]
powers_n0 = [[0.2, 0.4, 0.6, 0.8, 1.0], [0.3, 0.5, 0.7, 0.85, 0.95], [0.1, 0.3, 0.5, 0.75, 0.9],
             [0.2, 0.4, 0.6, 0.8, 1.0],[0.2, 0.4, 0.6, 0.8, 1.0],[0.2, 0.4, 0.6, 0.8, 1.0],[0.2, 0.4, 0.6, 0.8, 1.0]]
powers_n1 = [[0.2, 0.4, 0.6, 0.8, 1.0], [0.3, 0.5, 0.7, 0.85, 0.95], [0.1, 0.3, 0.5, 0.75, 0.9],
             [0.2, 0.4, 0.6, 0.8, 1.0],[0.2, 0.4, 0.6, 0.8, 1.0],[0.2, 0.4, 0.6, 0.8, 1.0],[0.2, 0.4, 0.6, 0.8, 1.0]]
powers_n2 = [[0.2, 0.4, 0.6, 0.8, 1.0], [0.3, 0.5, 0.7, 0.85, 0.95], [0.1, 0.3, 0.5, 0.75, 0.9],
             [0.2, 0.4, 0.6, 0.8, 1.0],[0.2, 0.4, 0.6, 0.8, 1.0],[0.2, 0.4, 0.6, 0.8, 1.0],[0.2, 0.4, 0.6, 0.8, 1.0]]

titles =['Custom Title 1', 'Custom Title 2', 'Custom Title 3']
plot_power_effect(methods, effect_sizes, powers_n0, powers_n1, powers_n2, titles=titles)

###############################################################################################################