# Experiment Plot (Personalized Parameter Distribution)

import json
import matplotlib.pyplot as plt
import pandas as pd
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
    'xtick.size': base_font_size * 0.8,  # x轴刻度字体大小
    'ytick.size': base_font_size * 0.8,  # y轴刻度字体大小
    'legend.loc': 'upper right'  # 图例位置
}

# 文件路径列表
files = ['./distribution/cifar10_mask_statistics.json',
         './distribution/cifar100_mask_statistics.json',
         './distribution/emnist_mask_statistics.json',
         './distribution/svhn_mask_statistics.json']

# files = ['./distribution/mnist_mask_statistics.json',
#          './distribution/fmnist_mask_statistics.json',
#          './distribution/medmnistA_mask_statistics.json',
#          './distribution/medmnistC_mask_statistics.json']

# 子图标题
titles = ['CIFAR10', 'CIFAR100', 'EMNIST', 'SHVN']

# titles = ['MNIST', 'FMNIST', 'MedMNISTA', 'MedMNISTC']

# 每个文件对应的除数（比例因子）
divisors = [44, 93, 61, 36]  # CIFAR10/100 EMNIST SHVN
# divisors = [6, 30, 30, 30]  # MNIST FMNIST MedmnistA MedmnistC

# 设置图形为1行4列的子图
fig, axes = plt.subplots(1, 4, figsize=(28, 6), dpi=dpi)

# 准备图表
for i, file in enumerate(files):
    # 读取JSON文件
    with open(file, 'r') as f:
        data = json.load(f)

    # 准备统计数据
    layers = ['conv1', 'conv2', 'fc1', 'classifier']
    false_counts = {layer: {'weight': [], 'bias': []} for layer in layers}

    # 统计每一层的False数量，排除false总和超过1000的数据
    for entry in data:
        total_false = sum(entry[layer]['weight']['False'] + entry[layer]['bias']['False'] for layer in layers)

        if total_false <= 1000:  # 排除false总和大于1000的数据
            for layer in layers:
                false_counts[layer]['weight'].append(entry[layer]['weight']['False'])
                false_counts[layer]['bias'].append(entry[layer]['bias']['False'])

    # 平滑操作：使用滑动平均窗口大小为5
    smoothed_false_counts = {layer: {'combined': []} for layer in layers}
    window_size = 3  # 可以调整窗口大小

    for layer in layers:
        # 计算weight和bias的合并数据
        combined_data = [false_counts[layer]['weight'][i] + false_counts[layer]['bias'][i]
                         for i in range(len(false_counts[layer]['weight']))]
        # 平滑操作
        smoothed_false_counts[layer]['combined'] = pd.Series(combined_data).rolling(window=window_size).mean().tolist()

    # 获取对应的子图
    ax = axes[i]

    # 绘制每个文件的图表
    for layer in layers:
        # 获取当前文件的除数
        divisor = divisors[i]

        # 计算比例：除以除数
        ratio_data = [val / divisor for val in smoothed_false_counts[layer]['combined']]

        # 绘制比例图
        ax.plot(range(len(ratio_data)), ratio_data, label=f'{layer}', linewidth=2)

    # 设置图表
    ax.set_title(titles[i], fontsize=plt_dict['title.size'], fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=plt_dict['label.size'], fontweight='bold')
    ax.set_ylabel('Personalized Parameters Proportion', fontsize=plt_dict['label.size'], fontweight='bold')

    # 设置网格线
    ax.grid(True)  # Major grid lines
    # ax.minorticks_on()  # Enable minor ticks
    # ax.grid(True)  # Minor grid lines

    # 设置x轴和y轴刻度字体大小
    ax.tick_params(axis='x', labelsize=plt_dict['xtick.size'])
    ax.tick_params(axis='y', labelsize=plt_dict['ytick.size'])

    # 自定义 x 轴刻度
    custom_ticks = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]  # 自定义刻度位置
    custom_labels = ['0', '50', '100', '150', '200', '250', '300', '350', '400', '450']  # 自定义刻度标签

    # 设置x轴刻度位置
    ax.set_xticks(custom_ticks)

    # 设置x轴刻度标签
    ax.set_xticklabels(custom_labels)

    # 动态选择图例位置
    ax.legend(fontsize=plt_dict['legend.size'], loc='best', title='Layers', title_fontsize=plt_dict['legend.size'],
              frameon=True, markerscale=1.5)

# 调整子图之间的间距
plt.subplots_adjust(wspace=0.3)  # 增加子图之间的间距，可以调整wspace的值

# 调整布局后保存图表为PDF
plt.tight_layout()
plt.savefig('./figures/param_distribution_1.pdf', format='pdf', dpi=dpi)

# 显示图表
plt.show()