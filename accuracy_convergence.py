# Experiment Plot (Convergence Analysis)

import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter  # 导入平滑处理函数
import matplotlib  # 导入matplotlib以设置参数


def find_csv_file(dataset_folder):
    """
    在数据集文件夹下递归查找csv文件。

    :param dataset_folder: 数据集文件夹路径
    :return: csv文件的路径，如果没有找到则返回None
    """
    for root, dirs, files in os.walk(dataset_folder):
        for file in files:
            if file.endswith(".csv"):
                return os.path.join(root, file)
    return None


def load_data(root_folder, algorithm, dataset_folder):
    """
    加载指定算法和数据集的CSV数据。

    :param root_folder: 主文件夹路径
    :param algorithm: 算法名称
    :param dataset_folder: 数据集文件夹名称
    :return: accuracy_test_before 列的数据，如果没有找到数据则返回 None
    """
    dataset_path = os.path.join(root_folder, algorithm, dataset_folder)
    csv_file = find_csv_file(dataset_path)

    if csv_file:
        df = pd.read_csv(csv_file)
        return df['accuracy_test_before'].values
    else:
        print(f"警告: 在 {dataset_folder} 中没有找到csv文件")
        return None


def plot_algorithm_per_dataset(root_folder, algorithms=None, datasets=None, y_limits=None, legend_names=None):
    """
    绘制不同算法在不同数据集上的折线图，每个数据集一张子图。

    :param root_folder: 主文件夹路径，包含算法和数据集文件夹
    :param algorithms: 想要绘制的算法列表 (可选，如果为空则绘制所有算法)
    :param datasets: 想要绘制的数据集列表 (可选，如果为空则绘制所有数据集)
    :param y_limits: y轴范围字典 (可选，格式为 {dataset_name: (y_min, y_max)})
    :param legend_names: 自定义图例名称字典 (可选，格式为 {algorithm_name: display_name})
    """
    # 设置dpi和字体样式
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

    # 设置基本字体大小
    base_font_size = 22  # 基础字体大小，其他字体大小将基于此进行调整
    scale_factor = 1.3  # 设置比例因子
    plt.rcParams['axes.titlesize'] = base_font_size * scale_factor  # 标题字体大小
    plt.rcParams['axes.labelsize'] = base_font_size  # 字标签大小
    plt.rcParams['legend.fontsize'] = base_font_size * 0.8  # 图例字体大小（缩小）
    plt.rcParams['xtick.labelsize'] = base_font_size * 0.8  # x轴刻度字体大小
    plt.rcParams['ytick.labelsize'] = base_font_size * 0.8  # y轴刻度字体大小
    plt.rcParams['legend.loc'] = 'upper right'  # 图例位置

    # 获取所有的算法文件夹
    algorithm_folders = [f for f in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, f))]

    # 如果指定了算法，则过滤
    if algorithms:
        algorithm_folders = [alg for alg in algorithm_folders if alg in algorithms]

    # 获取所有的数据集文件夹并去重
    dataset_folders = set()  # 使用集合去重
    for alg_folder in algorithm_folders:
        dataset_folders.update([f for f in os.listdir(os.path.join(root_folder, alg_folder)) if
                                 os.path.isdir(os.path.join(root_folder, alg_folder, f))])

    # 如果指定了数据集，则过滤
    if datasets:
        dataset_folders = [ds for ds in datasets if ds in dataset_folders]  # 按照指定顺序过滤
    else:
        dataset_folders = list(dataset_folders)  # 转换回列表

    # 创建 1x4 的子图
    n_cols = min(4, len(dataset_folders))  # 每行最多4个子图
    n_rows = (len(dataset_folders) + n_cols - 1) // n_cols  # 计算行数
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(25, 6 * n_rows))
    axes = axes.flatten()  # 将axes展平为一维数组

    # 存储图例标签
    legend_labels = []

    # 绘制每个数据集的折线图
    # 绘制每个数据集的折线图
    for i, dataset_folder in enumerate(dataset_folders):
        ax = axes[i]
        ax.set_title(dataset_folder.upper())  # 数据集名称为title
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")

        # 绘制每个算法在该数据集下的折线图
        for alg_folder in algorithms:  # 按照指定的算法顺序绘制
            if alg_folder in algorithm_folders:  # 确保算法存在
                test_before = load_data(root_folder, alg_folder, dataset_folder)

                if test_before is not None:
                    # 使用Savitzky-Golay滤波器进行平滑处理
                    smoothed_data = savgol_filter(test_before, window_length=20, polyorder=2)

                    # 判断是否为 FedOBP，设置加粗样式
                    if alg_folder == "FedOBP":
                        line, = ax.plot(smoothed_data, label=legend_names.get(alg_folder, alg_folder), linewidth=3,
                                        color='red')  # 加粗并设置颜色
                    else:
                        line, = ax.plot(smoothed_data, label=legend_names.get(alg_folder, alg_folder))  # 使用自定义名称

                    legend_labels.append(line)

        ax.legend().set_visible(False)  # 隐藏每个子图的图例

        # 设置y轴范围
        if y_limits and dataset_folder in y_limits:
            ax.set_ylim(y_limits[dataset_folder])

        ax.grid(True)

        # 创建全局图例，调整位置
    fig.legend(handles=legend_labels, labels=[legend_names.get(alg, alg) for alg in algorithms],
               loc='lower center', bbox_to_anchor=(0.5, 0.03), ncol=12, fontsize=base_font_size * 0.8)

    # 隐藏未使用的子图
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0.1, 1, 1])  # 调整布局以留出空间给图例

    # 保存图像
    output_path = f'./figures/convergence_analysis_mnist_05.pdf'  # 设置输出路径为PDF文件
    plt.savefig(output_path, dpi=dpi, format='pdf')
    plt.show()


# 调用函数，传入根文件夹路径和自定义的算法或数据集列表
root_folder = "accuracy/mnist4datasets/alpha=0.5"
# root_folder = "accuracy/cifar4datasets/alpha=0.5"
# algorithms = ["local", "fedavg", "fedper", "apfl", "lgfedavg", "fedrep", "fedrod", "pfedfda", "flute", "feddpa", "floco", "FedOBP"]  # 可选，自定义要绘制的算法
algorithms = ["local", "fedavg", "fedper", "apfl", "lgfedavg", "fedrep", "pfedfda", "flute", "feddpa", "floco", "FedOBP"]  # 可选，自定义要绘制的算法
# datasets = ["cifar10", "cifar100", "emnist", "svhn"]  # 可选，自定义要绘制的数据集
datasets = ["mnist", "fmnist", "medmnistA", "medmnistC"]  # 可选，自定义要绘制的数据集

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

# 自定义图例名称
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