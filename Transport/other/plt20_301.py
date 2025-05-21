import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.dates as mdates

# 加载npz文件
data = np.load('PEMS04.npz', allow_pickle=True)
data_array = data['data']

# 设置时间范围
start_date = pd.Timestamp('2018-01-26')
end_date = pd.Timestamp('2018-01-28')

# 计算索引位置
start_index = (start_date - pd.Timestamp('2018-01-01')).total_seconds() // (5 * 60)
end_index = (end_date - pd.Timestamp('2018-01-01')).total_seconds() // (5 * 60) + 1

# 创建时间序列
selected_dates = pd.date_range(start=start_date, periods=end_index-start_index, freq='5T')

# 提取数据
flow_sensor_20 = data_array[int(start_index):int(end_index), 20 - 1, 0]  # Sensor 20的数据
flow_sensor_301 = data_array[int(start_index):int(end_index), 300, 0]    # Sensor 301的数据

# 绘图
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(selected_dates, flow_sensor_20, label='Sensor 20', color='orange')
ax.plot(selected_dates, flow_sensor_301, label='Sensor 301', color='blue')
ax.legend()
ax.set_title('Traffic Flow Data for Sensor 20 and Sensor 301 (2018-01-26 to 2018-01-28)')
ax.set_xlabel('Date and Time')
ax.set_ylabel('Flow')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.xaxis.set_major_locator(mdates.DayLocator())
plt.gcf().autofmt_xdate()
plt.show()
