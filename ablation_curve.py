import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import re


def plot_ablation_experiment_results(root_folder):
    """
    绘制消融实验结果的折线图，按照数据集数量一行四列的布局绘制。

    :param root_folder: 主文件夹路径，包含多个数据集文件夹
    """
    # 使用正则表达式处理文件夹名并映射标签
    label_mapping = [
        (r"Ablation_GlobalNorm_EMAscore\(0.5\)_ALL_IG_.*", "EMA w/o CLS"),
        (r"Ablation_GlobalNorm_EMAscore\(0.5\)_CLS_IG_.*", "EMA & CLS"),
        (r"Ablation_GlobalNorm_NoEMAscore_ALL_IG_.*", "NA"),
        (r"Ablation_GlobalNorm_NoEMAscore_CLS_IG_.*", "CLS w/o EMA"),
        (r"Ablation_LayerNorm_NoEMAscore_ALL_IG_.*", "LayerNorm"),
        (r"Ablation_NoNorm_NoEMAscore_ALL_IG_.*", "NoNorm")
    ]

    # 获取所有的子文件夹（数据集）
    dataset_folders = [f for f in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, f))]

    # 创建 1x4 的子图
    n_cols = 4  # 每行最多4个子图
    n_rows = (len(dataset_folders) + n_cols - 1) // n_cols  # 计算行数
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(25, 6 * n_rows))
    axes = axes.flatten()  # 将axes展平为一维数组

    legend_labels = []  # 存储图例标签

    # 绘制每个数据集的折线图
    for i, dataset_folder in enumerate(dataset_folders):
        ax = axes[i]
        ax.set_title(dataset_folder.upper())  # 数据集名称为title
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")

        # 获取数据集下的所有实验文件夹
        experiment_folders = [f for f in os.listdir(os.path.join(root_folder, dataset_folder)) if
                              os.path.isdir(os.path.join(root_folder, dataset_folder, f))]

        # 绘制每个实验的折线图
        for exp_folder in experiment_folders:
            # 根据正则匹配标签
            label = None
            for pattern, exp_label in label_mapping:
                if re.match(pattern, exp_folder):
                    label = exp_label
                    break

            # 读取对应的CSV文件并提取 accuracy_test_before 列
            csv_path = os.path.join(root_folder, dataset_folder, exp_folder, "results.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                accuracy_data = df["accuracy_test_before"].values

                # 使用Savitzky-Golay滤波器进行平滑处理
                smoothed_data = savgol_filter(accuracy_data, window_length=20, polyorder=2)

                # 绘制折线图
                ax.plot(smoothed_data, label=label)

        ax.legend(loc="upper right")  # 添加图例
        ax.grid(True)

    # 调整图表布局
    plt.tight_layout()
    plt.show()


# 示例用法
root_folder = "/path/to/your/folder"
plot_ablation_experiment_results(root_folder)
