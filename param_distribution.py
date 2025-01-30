import json
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib

# Set dpi and font style
dpi = 300
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = 'Times New Roman'
matplotlib.rcParams["axes.edgecolor"] = "black"
matplotlib.rcParams["axes.linewidth"] = 1
matplotlib.rcParams['font.weight'] = 'bold'
matplotlib.rcParams['axes.labelweight'] = 'bold'
matplotlib.rcParams['axes.titleweight'] = 'bold'
matplotlib.rcParams['axes.linewidth'] = 1.5
matplotlib.rcParams['xtick.labelsize'] = 'x-large'
matplotlib.rcParams['ytick.labelsize'] = 'x-large'

matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.it'] = 'STIXGeneral:italic'
matplotlib.rcParams['mathtext.bf'] = 'STIXGeneral:italic:bold'

# Set base font size
base_font_size = 22  # Base font size, other font sizes will be adjusted based on this

# Set scale factor (e.g., 1.5 means font size is increased by 1.5 times)
scale_factor = 1.5

# Adjust font sizes using the scale factor
plt_dict = {
    'label.size': base_font_size,  # Label font size
    'title.size': base_font_size * scale_factor,  # Title font size
    'legend.size': base_font_size * 0.8,  # Legend font size (reduced)
    'xtick.size': base_font_size * 0.8,  # x-axis tick font size
    'ytick.size': base_font_size * 0.8,  # y-axis tick font size
    'legend.loc': 'upper right'  # Legend position
}

# File path list
files = ['./distribution/cifar10_mask_statistics.json',
         './distribution/cifar100_mask_statistics.json',
         './distribution/emnist_mask_statistics.json',
         './distribution/svhn_mask_statistics.json']

# files = ['./distribution/mnist_mask_statistics.json',
#          './distribution/fmnist_mask_statistics.json',
#          './distribution/medmnistA_mask_statistics.json',
#          './distribution/medmnistC_mask_statistics.json']

# Subplot titles
titles = ['CIFAR10', 'CIFAR100', 'EMNIST', 'SVHN']

# titles = ['MNIST', 'FMNIST', 'MedMNISTA', 'MedMNISTC']

# Divisors for each file (scaling factors)
divisors = [44, 93, 61, 36]  # CIFAR10/100 EMNIST SVHN
# divisors = [6, 30, 30, 30]  # MNIST FMNIST MedmnistA MedmnistC

# Set up the figure for 1 row and 4 columns of subplots
fig, axes = plt.subplots(1, 4, figsize=(28, 6), dpi=dpi)

# Prepare the charts
for i, file in enumerate(files):
    # Read the JSON file
    with open(file, 'r') as f:
        data = json.load(f)

    # Prepare statistics data
    layers = ['conv1', 'conv2', 'fc1', 'classifier']
    false_counts = {layer: {'weight': [], 'bias': []} for layer in layers}

    # Count the number of False values for each layer, excluding data where the total false count exceeds 1000
    for entry in data:
        total_false = sum(entry[layer]['weight']['False'] + entry[layer]['bias']['False'] for layer in layers)

        if total_false <= 1000:  # Exclude data where the total false count is greater than 1000
            for layer in layers:
                false_counts[layer]['weight'].append(entry[layer]['weight']['False'])
                false_counts[layer]['bias'].append(entry[layer]['bias']['False'])

    # Smoothing operation: use a moving average with a window size of 5
    smoothed_false_counts = {layer: {'combined': []} for layer in layers}
    window_size = 3  # Can adjust the window size

    for layer in layers:
        # Calculate combined data for weight and bias
        combined_data = [false_counts[layer]['weight'][i] + false_counts[layer]['bias'][i]
                         for i in range(len(false_counts[layer]['weight']))]
        # Smoothing operation
        smoothed_false_counts[layer]['combined'] = pd.Series(combined_data).rolling(window=window_size).mean().tolist()

    # Get the corresponding subplot
    ax = axes[i]

    # Plot the chart for each file
    for layer in layers:
        # Get the divisor for the current file
        divisor = divisors[i]

        # Calculate the ratio: divide by the divisor
        ratio_data = [val / divisor for val in smoothed_false_counts[layer]['combined']]

        # Plot the ratio chart
        ax.plot(range(len(ratio_data)), ratio_data, label=f'{layer}', linewidth=2)

    # Set up the chart
    ax.set_title(titles[i], fontsize=plt_dict['title.size'], fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=plt_dict['label.size'], fontweight='bold')
    ax.set_ylabel('Personalized Parameters Proportion', fontsize=plt_dict['label.size'], fontweight='bold')

    # Set grid lines
    ax.grid(True)  # Major grid lines
    # ax.minorticks_on()  # Enable minor ticks
    # ax.grid(True)  # Minor grid lines

    # Set x-axis and y-axis tick font sizes
    ax.tick_params(axis='x', labelsize=plt_dict['xtick.size'])
    ax.tick_params(axis='y', labelsize=plt_dict['ytick.size'])

    # Custom x-axis ticks
    custom_ticks = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]  # Custom tick positions
    custom_labels = ['0', '50', '100', '150', '200', '250', '300', '350', '400', '450']  # Custom tick labels

    # Set x-axis tick positions
    ax.set_xticks(custom_ticks)

    # Set x-axis tick labels
    ax.set_xticklabels(custom_labels)

    # Dynamically choose legend position
    ax.legend(fontsize=plt_dict['legend.size'], loc='best', title='Layers', title_fontsize=plt_dict['legend.size'],
              frameon=True, markerscale=1.5)

# Adjust the spacing between subplots
plt.subplots_adjust(wspace=0.3)  # Increase the space between subplots, can adjust the value of wspace

# Adjust layout and save the chart as a PDF
plt.tight_layout()
plt.savefig('./figures/param_distribution_1.pdf', format='pdf', dpi=dpi)

# Show the chart
plt.show()