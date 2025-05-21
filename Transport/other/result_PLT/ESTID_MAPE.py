import matplotlib.pyplot as plt
import numpy as np

# 数据集名称
datasets = ['PEMSO3', 'PEMSO4', 'PEMSO7', 'PEMSO8']

# 方法名称
methods = ['VAR', 'LSTM', 'DCRNN', 'STGCN', 'GWNet', 'GMAN', 'STNorm', 'STID', 'ESTIM']

# 不同数据集上的MAE值
MAE_values = {
    'VAR': [26.14, 23.51, 37.06, 31.02],
    'LSTM': [26.13, 23.81, 23.54, 22.07],
    'DCRNN': [22.67, 19.71, 21.20, 21.31],
    'STGCN': [22.59, 19.63, 21.71, 15.26],
    'GWNet': [20.91, 18.97, 20.25, 15.98],
    'GMAN': [21.74, 18.83, 20.43, 15.91],
    'STNorm': [20.51, 18.96, 20.52, 15.54],
    'STID': [20.68, 18.29, 19.54, 14.20],
    'ESTIM': [20.30, 18.13, 19.21, 13.61]
}

# 创建画布
plt.figure(figsize=(10, 6))

# 绘制每个方法的折线图
for i, method in enumerate(methods):
    if method == 'ESTIM':
        plt.plot(datasets, MAE_values[method], marker='o', label=method, color='red', linewidth=2)
    else:
        color = plt.cm.tab10(i / len(methods))  # 使用tab10色图来生成不同颜色
        plt.plot(datasets, MAE_values[method], marker='o', label=method, color=color, alpha=0.7)

# 添加标签和标题
plt.xlabel('Datasets')
plt.ylabel('MAE')
plt.title('MAE for Different Methods on Different Datasets')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# 不同数据集上的RMSE值
RMSE_values = {
    'VAR': [39.92, 36.39, 55.73, 38.97],
    'LSTM': [40.18, 36.62, 38.20, 33.10],
    'DCRNN': [36.85, 31.43, 34.43, 32.10],
    'STGCN': [36.66, 31.32, 34.53, 22.48],
    'GWNet': [33.12, 30.32, 33.32, 25.37],
    'GMAN': [33.23, 30.93, 33.30, 25.44],
    'STNorm': [34.12, 30.98, 34.85, 26.01],
    'STID': [33.23, 29.82, 32.85, 23.49],
    'ESTIM': [32.92, 29.43, 31.52, 22.52]
}

# 创建画布
plt.figure(figsize=(10, 6))

# 绘制每个方法的折线图
for i, method in enumerate(methods):
    if method == 'ESTIM':
        plt.plot(datasets, RMSE_values[method], marker='s', linestyle='--', label=method, color='red', linewidth=2)
    else:
        color = plt.cm.tab10(i / len(methods))  # 使用tab10色图来生成不同颜色
        plt.plot(datasets, RMSE_values[method], marker='s', linestyle='--', label=method, color=color, alpha=0.7)

# 添加标签和标题
plt.xlabel('Datasets')
plt.ylabel('RMSE')
plt.title('RMSE for Different Methods on Different Datasets')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# 不同数据集上的MAPE值
MAPE_values = {
    'VAR': [18.92, 17.85, 19.93, 14.04],
    'LSTM': [18.31, 18.12, 9.96, 17.04],
    'DCRNN': [15.97, 13.54, 9.06, 17.47],
    'STGCN': [15.01, 13.82, 9.25, 9.96],
    'GWNet': [12.88, 14.26, 8.63, 10.43],
    'GMAN': [11.85, 13.21, 8.69, 10.90],
    'STNorm': [13.71, 12.69, 8.77, 10.03],
    'STID': [13.34, 12.49, 8.25, 9.28],
    'ESTIM': [12.97, 12.12, 7.92, 8.91]
}

# 创建画布
plt.figure(figsize=(10, 6))

# 绘制每个方法的折线图
for i, method in enumerate(methods):
    if method == 'ESTIM':
        plt.plot(datasets, MAPE_values[method], marker='^', linestyle='-.', label=method, color='red', linewidth=2)
    else:
        color = plt.cm.tab10(i / len(methods))  # 使用tab10色图来生成不同颜色
        plt.plot(datasets, MAPE_values[method], marker='^', linestyle='-.', label=method, color=color, alpha=0.7)

# 添加标签和标题
plt.xlabel('Datasets')
plt.ylabel('MAPE (%)')
plt.title('MAPE for Different Methods on Different Datasets')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
