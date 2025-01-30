import os
import pandas as pd
import matplotlib.pyplot as plt

# Root directory path
# root_dir = 'D:/IEEETOC/data_1114/vgg11-compare/a=0.5'
# root_dir = 'D:/IEEETOC/alpha/out_a=1'
root_dir = 'D:/IEEETOC/motivition3'

# List all algorithm folders
algorithm_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]

# Create a dictionary to store "test_before" data for the same dataset names
dataset_dict = {}

for algorithm in algorithm_folders:
    algorithm_dir = os.path.join(root_dir, algorithm)

    # Get all CSV files in the current algorithm folder
    csv_files = [f for f in os.listdir(algorithm_dir) if f.endswith('.csv')]

    for csv_file in csv_files:
        csv_path = os.path.join(algorithm_dir, csv_file)

        # Read the CSV file
        df = pd.read_csv(csv_path)

        # Get the dataset name (assuming the CSV filename is the dataset name)
        dataset_name = os.path.splitext(csv_file)[0]

        # Truncate the dataset name at the first underscore
        truncated_name = dataset_name.split('_')[0]

        # Check if the CSV file contains a 'test_before' column
        if 'test_before' in df.columns:
            # Calculate the moving average (MA), window size of 3
            ma = df['test_before'].rolling(window=1).mean()

            if truncated_name not in dataset_dict:
                dataset_dict[truncated_name] = {}
            dataset_dict[truncated_name][algorithm] = ma
        else:
            print(f"Warning: CSV does not contain 'test_before' column.")

# Plot the figures
fig, axes = plt.subplots(2, 5, figsize=(30, 10))
axes = axes.flatten()

# Ensure the number of plots does not exceed 10
for i, (dataset_name, data) in enumerate(dataset_dict.items()):
    if i >= 10:
        break

    ax = axes[i]

    for algorithm, values in data.items():
        ax.plot(values, label=algorithm)

    # Add grid with custom style
    ax.grid(True, linestyle='--', alpha=0.7, color='gray', linewidth=0.5)

    # Set title with truncated dataset name
    ax.set_title(f'{dataset_name}', fontsize=16)
    ax.set_xlabel('Epochs', fontsize=16)
    ax.set_ylabel('Accaracy', fontsize=16)
    ax.legend(fontsize=16)

    # Set tick font size
    ax.tick_params(axis='both', which='major', labelsize=16)

# Adjust layout for better spacing and readability
plt.tight_layout()

# Save the complete figure as one EPS file
plt.savefig('motivation3.eps', format='eps', bbox_inches='tight', transparent=True)

plt.show()