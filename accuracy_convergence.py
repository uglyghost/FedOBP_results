import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter  # Import smoothing function
import matplotlib  # Import matplotlib to set parameters


def find_csv_file(dataset_folder):
    """
    Recursively search for CSV files in the dataset folder.

    :param dataset_folder: Path to the dataset folder
    :return: Path to the CSV file, or None if not found
    """
    for root, dirs, files in os.walk(dataset_folder):
        for file in files:
            if file.endswith(".csv"):
                return os.path.join(root, file)
    return None


def load_data(root_folder, algorithm, dataset_folder):
    """
    Load CSV data for the specified algorithm and dataset.

    :param root_folder: Path to the main folder
    :param algorithm: Algorithm name
    :param dataset_folder: Dataset folder name
    :return: Data from the accuracy_test_before column, or None if data not found
    """
    dataset_path = os.path.join(root_folder, algorithm, dataset_folder)
    csv_file = find_csv_file(dataset_path)

    if csv_file:
        df = pd.read_csv(csv_file)
        return df['accuracy_test_before'].values
    else:
        print(f"Warning: No CSV file found in {dataset_folder}")
        return None


def plot_algorithm_per_dataset(root_folder, algorithms=None, datasets=None, y_limits=None, legend_names=None):
    """
    Plot line charts for different algorithms on different datasets, with each dataset in a separate subplot.

    :param root_folder: Path to the main folder containing algorithm and dataset folders
    :param algorithms: List of algorithms to plot (optional, if empty, all algorithms will be plotted)
    :param datasets: List of datasets to plot (optional, if empty, all datasets will be plotted)
    :param y_limits: Dictionary of y-axis limits (optional, format: {dataset_name: (y_min, y_max)})
    :param legend_names: Dictionary of custom legend names (optional, format: {algorithm_name: display_name})
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
    plt.rcParams['legend.loc'] = 'upper right'  # Legend position

    # Get all algorithm folders
    algorithm_folders = [f for f in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, f))]

    # Filter if algorithms are specified
    if algorithms:
        algorithm_folders = [alg for alg in algorithm_folders if alg in algorithms]

    # Get all dataset folders and remove duplicates
    dataset_folders = set()  # Use a set to remove duplicates
    for alg_folder in algorithm_folders:
        dataset_folders.update([f for f in os.listdir(os.path.join(root_folder, alg_folder)) if
                                 os.path.isdir(os.path.join(root_folder, alg_folder, f))])

    # Filter if datasets are specified
    if datasets:
        dataset_folders = [ds for ds in datasets if ds in dataset_folders]  # Filter according to specified order
    else:
        dataset_folders = list(dataset_folders)  # Convert back to list

    # Create 1x4 subplots
    n_cols = min(4, len(dataset_folders))  # Maximum of 4 subplots per row
    n_rows = (len(dataset_folders) + n_cols - 1) // n_cols  # Calculate number of rows
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(25, 6 * n_rows))
    axes = axes.flatten()  # Flatten axes to a 1D array

    # Store legend labels
    legend_labels = []

    # Plot line charts for each dataset
    for i, dataset_folder in enumerate(dataset_folders):
        ax = axes[i]
        ax.set_title(dataset_folder.upper())  # Dataset name as title
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")

        # Plot line charts for each algorithm in this dataset
        for alg_folder in algorithms:  # Plot in the specified order of algorithms
            if alg_folder in algorithm_folders:  # Ensure the algorithm exists
                test_before = load_data(root_folder, alg_folder, dataset_folder)

                if test_before is not None:
                    # Use Savitzky-Golay filter for smoothing
                    smoothed_data = savgol_filter(test_before, window_length=20, polyorder=2)

                    # Check if it is FedOBP and set bold style
                    if alg_folder == "FedOBP":
                        line, = ax.plot(smoothed_data, label=legend_names.get(alg_folder, alg_folder), linewidth=3,
                                        color='red')  # Bold and set color
                    else:
                        line, = ax.plot(smoothed_data, label=legend_names.get(alg_folder, alg_folder))  # Use custom name

                    legend_labels.append(line)

        ax.legend().set_visible(False)  # Hide legend for each subplot

        # Set y-axis limits
        if y_limits and dataset_folder in y_limits:
            ax.set_ylim(y_limits[dataset_folder])

        ax.grid(True)

    # Create global legend and adjust position
    fig.legend(handles=legend_labels, labels=[legend_names.get(alg, alg) for alg in algorithms],
               loc='lower center', bbox_to_anchor=(0.5, 0.03), ncol=12, fontsize=base_font_size * 0.8)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0.1, 1, 1])  # Adjust layout to leave space for the legend

    # Save the image
    output_path = f'./figures/convergence_analysis_mnist_05.pdf'  # Set output path as a PDF file
    plt.savefig(output_path, dpi=dpi, format='pdf')
    plt.show()


# Call the function, passing the root folder path and custom algorithm or dataset lists
root_folder = "accuracy/mnist4datasets/alpha=0.5"
# root_folder = "accuracy/cifar4datasets/alpha=0.5"
# algorithms = ["local", "fedavg", "fedper", "apfl", "lgfedavg", "fedrep", "fedrod", "pfedfda", "flute", "feddpa", "floco", "FedOBP"]  # Optional, customize algorithms to plot
algorithms = ["local", "fedavg", "fedper", "apfl", "lgfedavg", "fedrep", "pfedfda", "flute", "feddpa", "floco", "FedOBP"]  # Optional, customize algorithms to plot
# datasets = ["cifar10", "cifar100", "emnist", "svhn"]  # Optional, customize datasets to plot
datasets = ["mnist", "fmnist", "medmnistA", "medmnistC"]  # Optional, customize datasets to plot

# # alpha=0.1
# y_limits = {
#     "cifar10": (45, 95),
#     "cifar100": (10, 50),
#     "emnist": (40, 100),
#     "svhn": (50, 100)
# }

# y_limits = {
#     "mnist": (85, 100),
#     "fmnist": (60, 100),
#     "medmnistA": (0, 75),
#     "medmnistC": (0, 75)
# }

# alpha=0.5
# y_limits = {
#     "cifar10": (40, 75),
#     "cifar100": (10, 30),
#     "emnist": (60, 90),
#     "svhn": (75, 95)
# }

y_limits = {
    "mnist": (70, 100),
    "fmnist": (70, 95),
    "medmnistA": (0, 48),
    "medmnistC": (0, 48)
}

# Custom legend names
legend_names = {
    "local": "Local-Only",
    "fedavg": "FedAvg",
    "fedper": "FedPer",
    "apfl": "APFL",
    "lgfedavg": "LG-FedAvg",
    "fedrep": "FedRep",
    # "fedrod": "FedRoD",
    "pfedfda": "pFedFDA",
    "flute": "FLUTE",
    "feddpa": "FedDPA",
    "floco": "FLOCO",
    "FedOBP": "FedOBP"
}

plot_algorithm_per_dataset(root_folder, algorithms=algorithms, datasets=datasets, y_limits=y_limits, legend_names=legend_names)