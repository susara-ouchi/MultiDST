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
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

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