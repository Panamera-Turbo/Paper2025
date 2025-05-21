import matplotlib.pyplot as plt

# 数据集名称
datasets = ['PEMSO3', 'PEMSO4', 'PEMSO7', 'PEMSO8']

# 不同方法在各数据集上的平均时间花费
VAR = [30.20, 14.73, 189.37, 7.65]
LSTM = [15.95, 7.78, 25.73, 4.56]
DCRNN = [194.99, 95.12, 510.53, 57.17]
STGCN = [84.38, 41.16, 198.13, 25.31]
GWNet = [57.15, 27.88, 170.61, 29.72]
STID = [10.74, 5.24, 14.32, 4.46]
GMAN = [220.09, 107.31, 827.77, 71.04]
STNorm = [37.31, 18.20, 74.12, 32.45]
ESTIM = [11.24, 5.43, 15.0, 4.65]

# 画图
plt.figure(figsize=(10, 6))

plt.plot(datasets, VAR, marker='o', label='VAR')
plt.plot(datasets, LSTM, marker='o', label='LSTM')
plt.plot(datasets, DCRNN, marker='o', label='DCRNN')
plt.plot(datasets, STGCN, marker='o', label='STGCN')
plt.plot(datasets, GWNet, marker='o', label='GWNet')
plt.plot(datasets, STID, marker='o', label='STID')
plt.plot(datasets, GMAN, marker='o', label='GMAN')
plt.plot(datasets, STNorm, marker='o', label='STNorm')
plt.plot(datasets, ESTIM, marker='o', linestyle='--', color='red', label='ESTIM')  # 突出显示ESTIM方法，使用红色虚线

plt.xlabel('Datasets')
plt.ylabel('Average Time (seconds)')
plt.title('Average Time Cost per Epoch for Different Methods')
plt.xticks(rotation=45)
plt.ylim(0, 100)  # 设置Y轴范围为0到100
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
