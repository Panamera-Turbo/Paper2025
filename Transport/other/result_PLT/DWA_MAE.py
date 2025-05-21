import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 创建数据
data = {
    "Configuration": ["Only DWA", "Only DWA", "Only DWA", "Only DWA",
                      "No TPE and DWA (STID)", "No TPE and DWA (STID)", "No TPE and DWA (STID)", "No TPE and DWA (STID)"],
    "Dataset": ["PEMS03 Avg.", "PEMS04 Avg.", "PEMS07 Avg.", "PEMS08 Avg."] * 2,
    "MAE": [20.30, 18.13, 19.21, 13.61,
            20.71, 18.38, 19.64, 14.29],
    "RMSE": [32.92, 29.43, 31.52, 22.52,
             33.56, 29.95, 33.08, 23.63],
    "MAPE": [12.97, 12.12, 7.92, 8.91,
             13.42, 12.57, 8.31, 9.35]
}

# 转换为DataFrame
df = pd.DataFrame(data)

# 绘图函数
def plot_metric(metric):
    fig, ax = plt.subplots(figsize=(12, 7))
    for config in df['Configuration'].unique():
        subset = df[df['Configuration'] == config]
        ax.plot(subset['Dataset'], subset[metric], marker='o', label=config)

    # 设置图例和标签
    ax.set_xlabel("Dataset", fontsize=14)
    ax.set_ylabel(metric, fontsize=14)
    ax.set_title(f"{metric} Performance Comparison", fontsize=16)
    ax.legend(title="Configuration")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 绘制MAE
plot_metric("MAE")

# 绘制RMSE
plot_metric("RMSE")

# 绘制MAPE
plot_metric("MAPE")
