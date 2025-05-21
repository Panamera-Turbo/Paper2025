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
    "MAE": [20.30, 18.13, 19.21, 13.61,
            20.68, 18.29, 19.54, 14.20,
            20.71, 18.38, 19.64, 14.29,
            20.17, 17.98, 19.08, 13.50]
}

# 转换为DataFrame
df = pd.DataFrame(data)

# 绘图
fig, ax = plt.subplots(figsize=(12, 7))

# 设置颜色
colors = {
    "Full Model (ESTIM)": "gray",
    "No TPE and DWA (STID)": "blue",
    "TPE Only": "red",  # 突出显示
    "Only DWA": "gray"
}

# 绘制每个配置的MAE
for config in df['Configuration'].unique():
    subset = df[df['Configuration'] == config]
    ax.plot(subset['Dataset'], subset['MAE'], marker='o', label=config, color=colors[config])

# 设置y轴范围
min_mae = df['MAE'].min()
max_mae = df['MAE'].max()
ax.set_ylim([min_mae - 1, max_mae + 1])  # 为了更清楚地显示变化，调整y轴范围

ax.set_xlabel("Dataset", fontsize=14)
ax.set_ylabel("MAE", fontsize=14)
ax.set_title("MAE Performance Comparison", fontsize=16)
ax.legend(title="Configuration", fontsize=12)
plt.xticks(rotation=45)  # 旋转X轴标签以提高可读性
plt.grid(True)  # 启用网格
plt.tight_layout()  # 自动调整布局
plt.show()
