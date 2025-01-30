import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import matplotlib
import re

def plot_ablation_experiment_results(root_folder, custom_legend_order=None):
    """
    绘制消融实验结果的折线图，按照数据集数量一行四列的布局绘制。

    :param root_folder: 主文件夹路径，包含多个数据集文件夹
    :param custom_legend_order: 自定义的图例顺序列表，例如 ["EMA w/o CLS", "CLS w/o EMA"]
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

    # 使用正则表达式处理文件夹名并映射标签
    label_mapping = [
        (r"Ablation_GlobalNorm_EMAscore\(0.5\)_ALL_IG_.*", "GlobalNorm (EMA w/o CLS)"),
        (r"Ablation_GlobalNorm_EMAscore\(0.5\)_CLS_IG_.*", "GlobalNorm (EMA & CLS)"),
        (r"Ablation_GlobalNorm_NoEMAscore_ALL_IG_.*", "GlobalNorm (NA)"),
        (r"Ablation_GlobalNorm_NoEMAscore_CLS_IG_.*", "GlobalNorm (CLS w/o EMA)"),
        (r"Ablation_LayerNorm_NoEMAscore_ALL_IG_.*", "LayerNorm"),
        (r"Ablation_NoNorm_NoEMAscore_ALL_IG_.*", "NoNorm")
    ]

    # # CIFAR系列数据集坐标轴
    # # y轴坐标范围设置
    # y_limits = {
    #     "cifar10": (60, 90),
    #     "cifar100": (10, 50),
    #     "emnist": (87, 97),
    #     "svhn": (77, 97)
    # }

    # MNIST系列数据集坐标轴
    # y轴坐标范围设置
    y_limits = {
        "mnist": (95, 100),
        "fmnist": (80, 100),
        "medmnista": (50, 75),
        "medmnistc": (40, 75)
    }

    # 获取所有的子文件夹（数据集）
    dataset_folders = [f for f in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, f))]

    # 确保子图顺序与y轴范围一致
    dataset_order = ["mnist", "fmnist", "medmnistA", "medmnistC"]
    dataset_folders = [folder for folder in dataset_order if folder in dataset_folders]

    # 创建 1x4 的子图
    n_cols = 4  # 每行最多4个子图
    n_rows = (len(dataset_folders) + n_cols - 1) // n_cols  # 计算行数
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(28, 6 * n_rows), dpi=dpi)
    axes = axes.flatten()  # 将axes展平为一维数组

    # 绘制每个数据集的折线图
    for i, dataset_folder in enumerate(dataset_folders):
        ax = axes[i]
        ax.set_title(dataset_folder.upper())  # 数据集名称为title
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")

        # 获取数据集下的所有实验文件夹
        experiment_folders = [f for f in os.listdir(os.path.join(root_folder, dataset_folder)) if
                              os.path.isdir(os.path.join(root_folder, dataset_folder, f))]

        # 保存每个实验的线条和标签
        lines = []
        labels = []

        # 绘制每个实验的折线图
        for exp_folder in experiment_folders:
            # 根据正则匹配标签
            label = None
            for pattern, exp_label in label_mapping:
                if re.match(pattern, exp_folder):
                    label = exp_label
                    break

            # 读取对应的CSV文件并提取 accuracy_test_before 列
            csv_path = os.path.join(root_folder, dataset_folder, exp_folder, "metrics.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                accuracy_data = df["accuracy_test_before"].values

                # 使用Savitzky-Golay滤波器进行平滑处理
                smoothed_data = savgol_filter(accuracy_data, window_length=20, polyorder=2)

                # 绘制折线图
                line, = ax.plot(smoothed_data, label=label)
                lines.append(line)
                labels.append(label)

        # 自定义图例顺序
        if custom_legend_order:
            # 按照自定义顺序排列
            ordered_labels = [label for label in custom_legend_order if label in labels]
            ordered_lines = [lines[labels.index(label)] for label in ordered_labels]
            ax.legend(ordered_lines, ordered_labels, loc="lower right")
        else:
            ax.legend(loc="lower right")  # 默认图例位置为右下角

        ax.grid(True)

        # 设置y轴范围
        dataset_name = dataset_folder.lower()  # 将文件夹名转换为小写以匹配字典
        if dataset_name in y_limits:
            ax.set_ylim(y_limits[dataset_name])  # 设置y轴范围
        else:
            print(f"Warning: No y-limit found for dataset '{dataset_name}'")  # 提示未找到y轴范围

    # 隐藏多余的子图
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # 调整图表布局
    plt.tight_layout()

    # 保存图像
    output_path = f'./figures/ablation_mnist4datasets.pdf'  # 设置输出路径为PDF文件
    plt.savefig(output_path, dpi=dpi, format='pdf')
    plt.show()


# 示例用法
root_folder = "./results/ablation/mnist4datasets"

# 自定义图例顺序，例如：
custom_legend_order = ["NoNorm",
                       "LayerNorm",
                       "GlobalNorm (NA)",
                       "GlobalNorm (EMA & CLS)",
                       "GlobalNorm (CLS w/o EMA)",
                       "GlobalNorm (EMA w/o CLS)"]

plot_ablation_experiment_results(root_folder, custom_legend_order)