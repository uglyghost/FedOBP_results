import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import matplotlib
import re

def plot_ablation_experiment_results(root_folder, custom_legend_order=None):
    """
    Plot line charts of ablation experiment results, arranged in a layout of one row with four columns.

    :param root_folder: Path to the main folder containing multiple dataset folders
    :param custom_legend_order: Custom legend order list, e.g., ["EMA w/o CLS", "CLS w/o EMA"]
    """
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

    # Set base font size
    base_font_size = 22  # Base font size, other font sizes will be adjusted based on this
    scale_factor = 1.3  # Set scale factor
    plt.rcParams['axes.titlesize'] = base_font_size * scale_factor  # Title font size
    plt.rcParams['axes.labelsize'] = base_font_size  # Label font size
    plt.rcParams['legend.fontsize'] = base_font_size * 0.8  # Legend font size (reduced)
    plt.rcParams['xtick.labelsize'] = base_font_size * 0.8  # x-axis tick font size
    plt.rcParams['ytick.labelsize'] = base_font_size * 0.8  # y-axis tick font size

    # Use regular expressions to process folder names and map labels
    label_mapping = [
        # (r"Ablation_GlobalNorm_EMAscore\(0.5\)_ALL_IG_.*", "GlobalNorm (EMA w/o CLS)"),
        # (r"Ablation_GlobalNorm_EMAscore\(0.5\)_CLS_IG_.*", "GlobalNorm (EMA & CLS)"),
        (r"Ablation_GlobalNorm_NoEMAscore_ALL_IG_.*", "GlobalNorm"),
        (r"Ablation_GlobalNorm_NoEMAscore_CLS_IG_.*", "GlobalNorm (CLS)"),
        (r"Ablation_LayerNorm_NoEMAscore_ALL_IG_.*", "LayerNorm"),
        (r"Ablation_NoNorm_NoEMAscore_ALL_IG_.*", "NoNorm")
    ]

    # # CIFAR
    # y_limits = {
    #     "cifar10": (60, 90),
    #     "cifar100": (10, 50),
    #     "emnist": (87, 97),
    #     "svhn": (77, 97)
    # }

    # MNIST
    y_limits = {
        "mnist": (95, 100),
        "fmnist": (80, 100),
        "medmnista": (50, 75),
        "medmnistc": (40, 75)
    }

    # Get all subfolders (datasets)
    dataset_folders = [f for f in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, f))]

    # Ensure the order of subplots matches the y-axis limits
    # dataset_order = ["cifar10", "cifar100", "emnist", "svhn"]
    dataset_order = ["mnist", "fmnist", "medmnistA", "medmnistC"]
    dataset_folders = [folder for folder in dataset_order if folder in dataset_folders]

    # Create 1x4 subplots
    n_cols = 4  # Maximum of 4 subplots per row
    n_rows = (len(dataset_folders) + n_cols - 1) // n_cols  # Calculate number of rows
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(28, 6 * n_rows), dpi=dpi)
    axes = axes.flatten()  # Flatten axes to a 1D array

    # Plot line charts for each dataset
    for i, dataset_folder in enumerate(dataset_folders):
        ax = axes[i]
        ax.set_title(dataset_folder.upper())  # Dataset name as title
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")

        # Get all experiment folders under the dataset
        experiment_folders = [f for f in os.listdir(os.path.join(root_folder, dataset_folder)) if
                              os.path.isdir(os.path.join(root_folder, dataset_folder, f))]

        # Store lines and labels for each experiment
        lines = []
        labels = []

        # Plot line charts for each experiment
        for exp_folder in experiment_folders:
            # Match label using regex
            label = None
            for pattern, exp_label in label_mapping:
                if re.match(pattern, exp_folder):
                    label = exp_label
                    break

            # Read the corresponding CSV file and extract the accuracy_test_before column
            csv_path = os.path.join(root_folder, dataset_folder, exp_folder, "metrics.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                accuracy_data = df["accuracy_test_before"].values

                # Use Savitzky-Golay filter for smoothing
                smoothed_data = savgol_filter(accuracy_data, window_length=20, polyorder=2)

                # Plot line chart
                line, = ax.plot(smoothed_data, label=label)
                lines.append(line)
                labels.append(label)

        # Custom legend order
        if custom_legend_order:
            # Arrange according to custom order
            ordered_labels = [label for label in custom_legend_order if label in labels]
            ordered_lines = [lines[labels.index(label)] for label in ordered_labels]
            ax.legend(ordered_lines, ordered_labels, loc="lower right")
        else:
            ax.legend(loc="lower right")  # Default legend position is lower right

        ax.grid(True)

        # Set y-axis limits
        dataset_name = dataset_folder.lower()  # Convert folder name to lowercase to match the dictionary
        if dataset_name in y_limits:
            ax.set_ylim(y_limits[dataset_name])  # Set y-axis limits
        else:
            print(f"Warning: No y-limit found for dataset '{dataset_name}'")  # Warn if no y-limit found

    # Hide any extra subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Adjust the layout of the chart
    plt.tight_layout()

    # Save the image
    output_path = f'./figures/ablation_mnist4datasets.pdf'  # Set output path as a PDF file
    plt.savefig(output_path, dpi=dpi, format='pdf')
    plt.show()


# Example usage
root_folder = "./results/ablation/mnist4datasets"

# Custom legend order, for example:
custom_legend_order = ["NoNorm",
                       "LayerNorm",
                       "GlobalNorm",
                       # "GlobalNorm (EMA & CLS)",
                       # "GlobalNorm (CLS w/o EMA)",
                       "GlobalNorm (CLS)"
                       ]

plot_ablation_experiment_results(root_folder, custom_legend_order)