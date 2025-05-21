import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 数据集信息，确保每个字典都有'time_steps'键
datasets_info = {
    'PEMS03': {
        'start_date': '2018-09-01', 'end_date': '2018-11-30', 'num_sensors': 358, 'file': 'PEMS03.npz', 
        'time_steps': 26208, 'data_types': ['Traffic Flow']
    },
    'PEMS04': {
        'start_date': '2018-01-01', 'end_date': '2018-02-28', 'num_sensors': 307, 'file': 'PEMS04.npz', 
        'time_steps': 16992, 'data_types': ['Traffic Flow', 'Traffic Speed', 'Traffic Occupancy']
    },
    'PEMS07': {
        'start_date': '2017-05-01', 'end_date': '2017-08-31', 'num_sensors': 883, 'file': 'PEMS07.npz', 
        'time_steps': 28224, 'data_types': ['Traffic Flow']
    },
    'PEMS08': {
        'start_date': '2016-07-01', 'end_date': '2016-08-31', 'num_sensors': 170, 'file': 'PEMS08.npz', 
        'time_steps': 17856, 'data_types': ['Traffic Flow', 'Traffic Speed', 'Traffic Occupancy']
    }
}

# 加载数据集
def load_data(dataset):
    return np.load(dataset['file'], allow_pickle=True)['data']

# 创建绘图函数
def plot_traffic_data(dataset, start_date, duration, sensor_ids, data_type):
    try:
        dataset_info = datasets_info[dataset]
        data_array = load_data(dataset_info)
        
        # 验证数据类型是否有效
        if data_type not in dataset_info['data_types']:
            raise ValueError(f"{data_type} is not available in the selected dataset.")
        
        # 检查传入的sensor_ids是否在有效范围内
        max_sensor_id = dataset_info['num_sensors'] - 1
        if not all(0 <= sensor_id <= max_sensor_id for sensor_id in sensor_ids):
            raise ValueError(f"Sensor ID must be between 0 and {max_sensor_id}")

        # 检查日期和时间是否有效
        start_date_pd = pd.Timestamp(start_date)
        if start_date_pd < pd.Timestamp(dataset_info['start_date']) or start_date_pd > pd.Timestamp(dataset_info['end_date']):
            raise ValueError(f"Start date must be between {dataset_info['start_date']} and {dataset_info['end_date']}")

        # 确保持续时间在数据集的时间步数范围内
        start_index = int((start_date_pd - pd.Timestamp(dataset_info['start_date'])).total_seconds() // (5 * 60))
        end_index = start_index + duration * 12  # 每5分钟记录一次，所以乘以12
        if end_index > dataset_info['time_steps']:
            raise ValueError("The combination of start date and duration is out of the dataset's range.")

        time_range = pd.date_range(start_date, periods=duration * 12, freq='5T')
        fig, ax = plt.subplots(figsize=(10, 6))

        data_type_index = {'Traffic Flow': 0, 'Traffic Speed': 1, 'Traffic Occupancy': 2}.get(data_type, 0)
        for sensor_id in sensor_ids:
            data_values = data_array[int(start_index):int(end_index), sensor_id, data_type_index]
            ax.plot(time_range, data_values, label=f'Sensor {sensor_id} - {data_type}')

        ax.set_title(f'{data_type} Data for {dataset}')
        ax.set_xlabel('Time')
        ax.set_ylabel(data_type)
        ax.legend()
        plt.show()
    except Exception as e:
        messagebox.showerror('Error', str(e))

# 创建主窗口
root = tk.Tk()
root.title('PEMS Data Analysis')
root.geometry('450x500')  # 设置窗口大小

# 创建数据集选择部件
dataset_label = ttk.Label(root, text='Select Dataset:')
dataset_label.grid(row=0, column=0, padx=10, pady=10)
dataset_combobox = ttk.Combobox(root, values=list(datasets_info.keys()))
dataset_combobox.grid(row=0, column=1, padx=10, pady=10)
dataset_combobox.current(0)

# 创建日期选择部件
start_date_label = ttk.Label(root, text='Start Date (YYYY-MM-DD):')
start_date_label.grid(row=1, column=0, padx=10, pady=10)
start_date_entry = ttk.Entry(root)
start_date_entry.grid(row=1, column=1, padx=10, pady=10)

# 创建持续时间选择部件
duration_label = ttk.Label(root, text='Duration (hours):')
duration_label.grid(row=2, column=0, padx=10, pady=10)
duration_entry = ttk.Entry(root)
duration_entry.grid(row=2, column=1, padx=10, pady=10)

# 创建选择检测器部件
sensor_label = ttk.Label(root, text='Sensor IDs (comma-separated):')
sensor_label.grid(row=3, column=0, padx=10, pady=10)
sensor_entry = ttk.Entry(root)
sensor_entry.grid(row=3, column=1, padx=10, pady=10)

# 创建数据类型选择部件
data_type_label = ttk.Label(root, text='Select Data Type:')
data_type_label.grid(row=4, column=0, padx=10, pady=10)
data_type_combobox = ttk.Combobox(root, values=['Traffic Flow', 'Traffic Speed', 'Traffic Occupancy'])
data_type_combobox.grid(row=4, column=1, padx=10, pady=10)
data_type_combobox.current(0)  # 默认选中第一个选项

# 创建绘图按钮
plot_button = ttk.Button(root, text='Plot', command=lambda: plot_traffic_data(
    dataset_combobox.get(),
    start_date_entry.get(),
    int(duration_entry.get()),
    [int(x.strip()) for x in sensor_entry.get().split(',')],
    data_type_combobox.get()
))
plot_button.grid(row=5, column=0, columnspan=2, padx=10, pady=20)

# 运行主循环
root.mainloop()
