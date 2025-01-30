import os
import re
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

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

matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.it'] = 'STIXGeneral:italic'
matplotlib.rcParams['mathtext.bf'] = 'STIXGeneral:italic:bold'

# 设置基本字体大小
base_font_size = 22  # 基础字体大小，其他字体大小将基于此进行调整

# 设置比例因子（例如：1.5表示字体大小放大1.5倍）
scale_factor = 1.5

# 使用比例因子调整字体大小
plt_dict = {
    'label.size': base_font_size,  # 字标签大小
    'title.size': base_font_size * scale_factor,  # 标题字体大小
    'legend.size': base_font_size * 0.8,  # 图例字体大小（缩小）
    'xtick.size': base_font_size * 0.7,  # x轴刻度字体大小
    'ytick.size': base_font_size * 0.8,  # y轴刻度字体大小
    'legend.loc': 'lower left'  # 图例位置
}

# 设置方法文件夹路径
base_folder = "./scores/"  # 修改为实际文件夹路径
dataset_names = ["cifar10", "cifar100"]  # 添加多个数据集名称

# 自定义方法名称（LaTeX 格式）
custom_method_names = [
    r"$I_{\text{F}}(\cdot)$",
    r"$I_{\text{G}}(\cdot)$",
    r"$I_{\text{O}}(\cdot)$"
    # 添加更多自定义的 LaTeX 格式方法名称
]

# 用来存储每个方法的x和y值
methods_data = {dataset_name: {} for dataset_name in dataset_names}

# 遍历每个数据集
for dataset_name in dataset_names:
    base_folder_path = os.path.join(base_folder, dataset_name)

    # 遍历方法文件夹
    for method_name in os.listdir(base_folder_path):
        method_folder = os.path.join(base_folder_path, method_name)

        if os.path.isdir(method_folder):
            # 为当前方法初始化x和y列表
            x_vals = []
            y_vals = []

            # 遍历每个结果文件夹
            for result_folder in os.listdir(method_folder):
                result_folder_path = os.path.join(method_folder, result_folder)

                if os.path.isdir(result_folder_path):
                    # 使用正则表达式提取文件夹名中的最后一个数字（x坐标）
                    match = re.search(r'_(\d+\.\d+)$', result_folder)
                    if match:
                        x_val = float(match.group(1))
                        x_vals.append(x_val)

                        # 寻找并读取main.log文件
                        log_file_path = os.path.join(result_folder_path, 'main.log')
                        if os.path.exists(log_file_path):
                            with open(log_file_path, 'r') as log_file:
                                # 读取每一行并提取精度
                                for line in log_file:
                                    if "before fine-tuning" in line:
                                        match = re.search(r'before fine-tuning: (\d+\.\d+)%', line)
                                        if match:
                                            y_val = float(match.group(1))
                                            y_vals.append(y_val)
                                            break

            # 存储当前方法的数据
            methods_data[dataset_name][method_name] = (x_vals, y_vals)

# 创建主图和子图
fig, axs = plt.subplots(1, 4, figsize=(28, 6), dpi=dpi)

index = 0
# 绘制每个数据集的图
for dataset_idx, dataset_name in enumerate(dataset_names):
    # 获取当前数据集的方法数据
    dataset_methods = methods_data[dataset_name]

    # 绘制0到1.0的曲线
    ax1 = axs[dataset_idx+index]
    for idx, (method_name, (x_vals, y_vals)) in enumerate(dataset_methods.items()):
        ax1.plot(x_vals, y_vals, marker='o', label=f'{custom_method_names[idx]} (0.0 to 1.0)')

        # 找到最高点并突出显示
        max_index = np.argmax(y_vals)
        max_x = x_vals[max_index]
        max_y = y_vals[max_index]
        ax1.plot(max_x, max_y, 'ro', markersize=10)  # 用红色圆点标记最高点并加大
        ax1.plot(max_x, max_y, 'o', markersize=15, color='yellow', alpha=0.5)  # 用黄色圆点加大高亮显示

    # 设置x轴为更密集的刻度
    x_ticks = np.linspace(0, 1, 11)  # 创建0到1的x坐标刻度
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels([f'{tick:.1f}' for tick in x_ticks], fontsize=plt_dict['xtick.size'])

    # 设置y轴标签字体大小
    ax1.set_yticklabels([f'{tick}' for tick in ax1.get_yticks()], fontsize=plt_dict['ytick.size'])

    # 设置标题和标签
    ax1.set_title(f'{dataset_name.upper()} (0.0 to 1.0)', fontsize=plt_dict['title.size'], fontweight='bold')
    ax1.set_xlabel('Quantile', fontsize=plt_dict['label.size'], fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=plt_dict['label.size'], fontweight='bold')
    ax1.legend(fontsize=plt_dict['legend.size'], loc=plt_dict['legend.loc'], title='Scores', title_fontsize=plt_dict['legend.size'], frameon=True, markerscale=1.5)
    ax1.grid(True)

    # 绘制0.99到1.0的曲线
    ax2 = axs[dataset_idx+index+1]
    for idx, (method_name, (x_vals, y_vals)) in enumerate(dataset_methods.items()):
        # 筛选x坐标在0.99到1.0之间的部分
        filtered_x_vals = [x for x in x_vals if 0.99 <= x <= 1.0]
        filtered_y_vals = [y_vals[i] for i in range(len(x_vals)) if 0.99 <= x_vals[i] <= 1.0]

        ax2.plot(filtered_x_vals, filtered_y_vals, marker='o', label=f'{custom_method_names[idx]} (0.99 to 1.0)', linestyle='--')

        # 找到最高点并突出显示
        if filtered_y_vals:  # 确保有数据
            max_index = np.argmax(filtered_y_vals)
            max_x = filtered_x_vals[max_index]
            max_y = filtered_y_vals[max_index]
            ax2.plot(max_x, max_y, 'ro', markersize=10)  # 用红色圆点标记最高点并加大
            ax2.plot(max_x, max_y, 'o', markersize=15, color='yellow', alpha=0.5)  # 用黄色圆点加大高亮显示

    # 设置x轴为更密集的刻度
    x_ticks = np.linspace(0.99, 1.0, 11)  # 创建0.99到1.0的x坐标刻度
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels([f'{tick:.3f}' for tick in x_ticks], fontsize=plt_dict['xtick.size'])

    # 设置y轴标签字体大小
    ax2.set_yticklabels([f'{tick}' for tick in ax2.get_yticks()], fontsize=plt_dict['ytick.size'])

    # 设置标题和标签
    ax2.set_title(f'{dataset_name.upper()} (0.99 to 1.0)', fontsize=plt_dict['title.size'], fontweight='bold')
    ax2.set_xlabel('Quantile', fontsize=plt_dict['label.size'], fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=plt_dict['label.size'], fontweight='bold')
    ax2.legend(fontsize=plt_dict['legend.size'], loc=plt_dict['legend.loc'], title='Scores', title_fontsize=plt_dict['legend.size'], frameon=True, markerscale=1.5)
    ax2.grid(True)

    index += 1

# 调整子图间距
plt.tight_layout()

# 保存图像
output_path = f'./combined_{dataset_names[0]}_{dataset_names[1]}.pdf'  # 设置输出路径为PDF文件
plt.savefig(output_path, dpi=dpi, format='pdf')

# 显示图像
plt.show()