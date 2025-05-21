import matplotlib.pyplot as plt
import pandas as pd

# 创建数据
data = {
    "Configuration": ["Only DWA", "Only DWA", "Only DWA", "Only DWA",
                      "TPE Only", "TPE Only", "TPE Only", "TPE Only",
                      "No TPE and DWA (STID)", "No TPE and DWA (STID)", "No TPE and DWA (STID)", "No TPE and DWA (STID)",
                      "Full Model (ESTIM)", "Full Model (ESTIM)", "Full Model (ESTIM)", "Full Model (ESTIM)"],
    "Dataset": ["PEMS03 Avg.", "PEMS04 Avg.", "PEMS07 Avg.", "PEMS08 Avg.",
                "PEMS03 Avg.", "PEMS04 Avg.", "PEMS07 Avg.", "PEMS08 Avg.",
                "PEMS03 Avg.", "PEMS04 Avg.", "PEMS07 Avg.", "PEMS08 Avg.",
                "PEMS03 Avg.", "PEMS04 Avg.", "PEMS07 Avg.", "PEMS08 Avg."],
    "RMSE": [32.92, 29.43, 31.52, 22.52,
             33.33, 29.82, 32.85, 23.49,
             33.56, 29.95, 33.08, 23.63,
             32.67, 29.25, 31.36, 22.35]
}

# 转换为DataFrame
df = pd.DataFrame(data)

# 绘图
fig, ax = plt.subplots(figsize=(12, 7))

# 设置颜色
colors = {
    "Full Model (ESTIM)": "gray",
    "No TPE and DWA (STID)": "blue",
    "TPE Only": "red",
    "Only DWA": "gray"
}

# 绘制每个配置的RMSE
for config in df['Configuration'].unique():
    subset = df[df['Configuration'] == config]
    ax.plot(subset['Dataset'], subset['RMSE'], marker='o', label=config, color=colors[config])

# 设置y轴范围
min_rmse = df['RMSE'].min()
max_rmse = df['RMSE'].max()
ax.set_ylim([min_rmse - 1, max_rmse + 1])

ax.set_xlabel("Dataset", fontsize=14)
ax.set_ylabel("RMSE", fontsize=14)
ax.set_title("RMSE Performance Comparison", fontsize=16)
ax.legend(title="Configuration", fontsize=12)
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
