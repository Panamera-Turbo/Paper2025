import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.dates as mdates

# 加载npz文件
data = np.load('PEMS04.npz', allow_pickle=True)
data_array = data['data']

# 设置特定日期的时间索引和数据提取
start_date = pd.Timestamp('2018-01-26')
end_date = start_date + pd.Timedelta(days=1)
start_index = (start_date - pd.Timestamp('2018-01-01')).total_seconds() // (5 * 60)
end_index = start_index + 24 * 12  # 每5分钟记录一次，一天有12*24=288个时间点
one_day_range = pd.date_range(start_date, periods=24*12, freq='5T')
one_day_flow_sensor_301 = data_array[int(start_index):int(end_index), 300, 0]

# 设置特定日期的时间索引和数据提取（2018-01-27）
start_date_20180127 = pd.Timestamp('2018-01-27')
end_date_20180127 = start_date_20180127 + pd.Timedelta(days=1)
start_index_20180127 = (start_date_20180127 - pd.Timestamp('2018-01-01')).total_seconds() // (5 * 60)
end_index_20180127 = start_index_20180127 + 24 * 12  # 每5分钟记录一次，一天有12*24=288个时间点
one_day_range_20180127 = pd.date_range(start_date_20180127, periods=24*12, freq='5T')
one_day_flow_sensor_301_20180127 = data_array[int(start_index_20180127):int(end_index_20180127), 300, 0]

# 创建时间标签，用于绘图时仅显示小时和分钟
hourly_time_labels = [time.strftime('%H:%M') for time in one_day_range if time.minute == 0]  # 只保留每小时的时间标签

# 创建图形和轴对象
fig, ax = plt.subplots(figsize=(15, 5))

# 2018-1-26的数据绘制
ax.plot(hourly_time_labels, one_day_flow_sensor_301[::12], label='Sensor 301 (2018-01-26)', color='blue')  # 每小时一个数据点

# 2018-1-27的数据绘制
ax.plot(hourly_time_labels, one_day_flow_sensor_301_20180127[::12], label='Sensor 301 (2018-01-27)', color='red')  # 每小时一个数据点

# 设置标题和轴标签
ax.set_title('Traffic Flow Data for Sensor 301 on 2018-01-26 and 2018-01-27 (Hourly)')
ax.set_xlabel('Time of Day')
ax.set_ylabel('Flow')

# 图例
ax.legend()

# 显示图表
plt.xticks(rotation=45)  # 旋转标签以免重叠
plt.tight_layout()  # 调整布局以防止标签被截断
plt.show()
