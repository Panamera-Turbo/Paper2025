import numpy as np

# 加载数据
data = np.load('PEMS07.npz', allow_pickle=True)

# 提取'data'键关联的数据
data_array = data['data']

# 显示数据的形状和前几行
shape = data_array.shape
first_few_rows = data_array[:5]

print("Shape of the data:", shape)
print("First few rows of the data:", first_few_rows)
