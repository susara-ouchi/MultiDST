import matplotlib.pyplot as plt

def draw_p_bar_chart(categories, values, title='Bar Chart', xlabel='Categories', ylabel='Values', border_color='grey'):
    """
    Draw a bar chart using user-defined function.

    Parameters:
        categories (list): List of category labels.
        values (list): List of corresponding values for each category.
        title (str): Title of the chart (default is 'Bar Chart').
        xlabel (str): Label for the x-axis (default is 'Categories').
        ylabel (str): Label for the y-axis (default is 'Values').
        border_color (str): Color of the borders (default is 'grey').
    """
    colors = ['lightblue', 'lightgreen']  # Oceanic blue and light green
    bars = plt.bar(categories, values, color=colors)
    
    # Add borders to bars
    for bar in bars:
        bar.set_edgecolor(border_color)
    
    plt.title(title, fontname='Times New Roman', fontsize=15)
    plt.xlabel(xlabel, fontname='Times New Roman', fontsize=15)
    plt.ylabel(ylabel, fontname='Times New Roman', fontsize=15)
    plt.xticks(fontname='Times New Roman', fontsize=15)  # Set font for category labels
    
    # Adding percentage labels on bars
    total = sum(values)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height / total * 100:.2f}%', ha='center', va='bottom', fontname='Times New Roman')
    
    plt.ylim(0, 100)
    plt.yticks(range(0, 101, 10))

    plt.show()

# Example usage:
categories = ['Available', 'Missing']
values = [90, 10]
draw_p_bar_chart(categories, values, title='Data Availability', xlabel='Data Status', ylabel='Percentage')
