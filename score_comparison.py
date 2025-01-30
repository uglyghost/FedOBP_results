import os
import re
import matplotlib.pyplot as plt
import numpy as np
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
    'xtick.size': base_font_size * 0.7,  # x-axis tick font size
    'ytick.size': base_font_size * 0.8,  # y-axis tick font size
    'legend.loc': 'lower left'  # Legend position
}

# Set method folder path
base_folder = "./scores/"  # Change to the actual folder path
dataset_names = ["cifar10", "cifar100"]  # Add multiple dataset names

# Custom method names (LaTeX format)
custom_method_names = [
    r"$I_{\text{F}}(\cdot)$",
    r"$I_{\text{G}}(\cdot)$",
    r"$I_{\text{O}}(\cdot)$"
    # Add more custom LaTeX formatted method names
]

# Store x and y values for each method
methods_data = {dataset_name: {} for dataset_name in dataset_names}

# Iterate through each dataset
for dataset_name in dataset_names:
    base_folder_path = os.path.join(base_folder, dataset_name)

    # Iterate through method folders
    for method_name in os.listdir(base_folder_path):
        method_folder = os.path.join(base_folder_path, method_name)

        if os.path.isdir(method_folder):
            # Initialize x and y lists for the current method
            x_vals = []
            y_vals = []

            # Iterate through each result folder
            for result_folder in os.listdir(method_folder):
                result_folder_path = os.path.join(method_folder, result_folder)

                if os.path.isdir(result_folder_path):
                    # Use regex to extract the last number from the folder name (x coordinate)
                    match = re.search(r'_(\d+\.\d+)$', result_folder)
                    if match:
                        x_val = float(match.group(1))
                        x_vals.append(x_val)

                        # Look for and read the main.log file
                        log_file_path = os.path.join(result_folder_path, 'main.log')
                        if os.path.exists(log_file_path):
                            with open(log_file_path, 'r') as log_file:
                                # Read each line and extract accuracy
                                for line in log_file:
                                    if "before fine-tuning" in line:
                                        match = re.search(r'before fine-tuning: (\d+\.\d+)%', line)
                                        if match:
                                            y_val = float(match.group(1))
                                            y_vals.append(y_val)
                                            break

            # Store the current method's data
            methods_data[dataset_name][method_name] = (x_vals, y_vals)

# Create main figure and subplots
fig, axs = plt.subplots(1, 4, figsize=(28, 6), dpi=dpi)

index = 0
# Define colors and markers
colors = ['blue', 'orange', 'red']  # Use blue, orange, and red
markers = ['o', 's', '^']  # Circle, square, triangle markers

# Plot for each dataset
for dataset_idx, dataset_name in enumerate(dataset_names):
    # Get the method data for the current dataset
    dataset_methods = methods_data[dataset_name]

    # Plot the curve from 0 to 1.0
    ax1 = axs[dataset_idx + index]
    for idx, (method_name, (x_vals, y_vals)) in enumerate(dataset_methods.items()):
        color = colors[idx % len(colors)]  # Get the color for the current curve
        ax1.plot(x_vals, y_vals, marker=markers[idx % len(markers)], color=color, label=f'{custom_method_names[idx]}')

        # Find the highest point and highlight it
        max_index = np.argmax(y_vals)
        max_x = x_vals[max_index]
        max_y = y_vals[max_index]
        ax1.plot(max_x, max_y, 'o', markersize=10, color=color)  # Mark the highest point with the current curve color
        ax1.plot(max_x, max_y, 'o', markersize=15, color=color, alpha=0.5)  # Highlight the highest point with larger size

    # Set x-axis to have denser ticks
    x_ticks = np.linspace(0, 1, 11)  # Create x ticks from 0 to 1
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels([f'{tick:.1f}' for tick in x_ticks], fontsize=plt_dict['xtick.size'])

    # Set y-axis tick label font size
    ax1.set_yticklabels([f'{tick}' for tick in ax1.get_yticks()], fontsize=plt_dict['ytick.size'])

    # Set title and labels
    ax1.set_title(f'{dataset_name.upper()} (0.0 to 1.0)', fontsize=plt_dict['title.size'], fontweight='bold')
    ax1.set_xlabel('Quantile', fontsize=plt_dict['label.size'], fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=plt_dict['label.size'], fontweight='bold')
    ax1.legend(fontsize=plt_dict['legend.size'], loc=plt_dict['legend.loc'], title='Scores', title_fontsize=plt_dict['legend.size'], frameon=True, markerscale=1.5)
    ax1.grid(True)

    # Plot the curve from 0.99 to 1.0
    ax2 = axs[dataset_idx + index + 1]
    for idx, (method_name, (x_vals, y_vals)) in enumerate(dataset_methods.items()):
        # Filter x values between 0.99 and 1.0
        filtered_x_vals = [x for x in x_vals if 0.99 <= x <= 1.0]
        filtered_y_vals = [y_vals[i] for i in range(len(x_vals)) if 0.99 <= x_vals[i] <= 1.0]

        color = colors[idx % len(colors)]  # Get the color for the current curve
        ax2.plot(filtered_x_vals, filtered_y_vals, marker=markers[idx % len(markers)], color=color, label=f'{custom_method_names[idx]}', linestyle='--')

        # Find the highest point and highlight it
        if filtered_y_vals:  # Ensure there is data
            max_index = np.argmax(filtered_y_vals)
            max_x = filtered_x_vals[max_index]
            max_y = filtered_y_vals[max_index]
            ax2.plot(max_x, max_y, 'o', markersize=10, color=color)  # Mark the highest point with the current curve color
            ax2.plot(max_x, max_y, 'o', markersize=15, color=color, alpha=0.5)  # Highlight the highest point with larger size

    # Set x-axis to have denser ticks
    x_ticks = np.linspace(0.99, 1.0, 11)  # Create x ticks from 0.99 to 1.0
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels([f'{tick:.3f}' for tick in x_ticks], fontsize=plt_dict['xtick.size'])

    # Set y-axis tick label font size
    ax2.set_yticklabels([f'{tick}' for tick in ax2.get_yticks()], fontsize=plt_dict['ytick.size'])

    # Set title and labels
    ax2.set_title(f'{dataset_name.upper()} (0.99 to 1.0)', fontsize=plt_dict['title.size'], fontweight='bold')
    ax2.set_xlabel('Quantile', fontsize=plt_dict['label.size'], fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=plt_dict['label.size'], fontweight='bold')
    ax2.legend(fontsize=plt_dict['legend.size'], loc=plt_dict['legend.loc'], title='Scores', title_fontsize=plt_dict['legend.size'], frameon=True, markerscale=1.5)
    ax2.grid(True)

    index += 1

# Adjust spacing between subplots
plt.tight_layout()

# Save the image
output_path = f'./figures/score_comparison.pdf'  # Set output path as a PDF file
plt.savefig(output_path, dpi=dpi, format='pdf')

# Show the image
plt.show()