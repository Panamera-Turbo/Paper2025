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
    "MAPE": [12.97, 12.12, 7.92, 8.91,
             13.34, 12.49, 8.25, 9.28,
             13.42, 12.57, 8.31, 9.35,
             12.84, 12.02, 7.85, 8.83]
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

# 绘制每个配置的MAPE
for config in df['Configuration'].unique():
    subset = df[df['Configuration'] == config]
    ax.plot(subset['Dataset'], subset['MAPE'], marker='o', label=config, color=colors[config])

# 设置y轴范围
min_mape = df['MAPE'].min()
max_mape = df['MAPE'].max()
ax.set_ylim([min_mape - 1, max_mape + 1])

ax.set_xlabel("Dataset", fontsize=14)
ax.set_ylabel("MAPE", fontsize=14)
ax.set_title("MAPE Performance Comparison", fontsize=16)
ax.legend(title="Configuration", fontsize=12)
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
