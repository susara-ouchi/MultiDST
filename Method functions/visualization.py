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