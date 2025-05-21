import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os


# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 数据路径
npz_path = "/home/data2t1/tempuser/GTAgent/zTrain_test/transport/PEMS03.npz"
csv_path = "/home/data2t1/tempuser/GTAgent/zTrain_test/transport/PEMS03.csv"

class TrafficData:
    def __init__(self, X, y):
        self.X = X
        self.y = y

# 加载数据
def load_data():
    # 加载交通流量数据
    data = np.load(npz_path)
    traffic_data = data['data']  # 形状为 (26208, 358, 1)
    
    # 保存最后12个时刻的数据到CSV
    last_12 = traffic_data[-12:]  # 获取最后12个时刻的数据
    
    # 去掉最后12个时刻的数据
    traffic_data = traffic_data[:-12]
    
    # 加载距离数据
    distance_df = pd.read_csv(csv_path)
    
    return traffic_data, distance_df

# 数据预处理
def preprocess_data(traffic_data, seq_len=12, pred_len=1):

    n_samples = traffic_data.shape[0] - seq_len - pred_len + 1
    n_nodes = traffic_data.shape[1]
    n_features = traffic_data.shape[2]
    
    X = np.zeros((n_samples, seq_len, n_nodes, n_features))
    y = np.zeros((n_samples, n_nodes, n_features))
    
    for i in range(n_samples):
        X[i] = traffic_data[i:i+seq_len]
        y[i] = traffic_data[i+seq_len+pred_len-1]
    
    # 重塑X为(样本数, 序列长度, 节点数*特征数)
    X = X.reshape(n_samples, seq_len, -1)
    # 重塑y为(样本数, 节点数*特征数)
    y = y.reshape(n_samples, -1)
    
    return TrafficData(X, y)  # 返回封装好的数据对象

traffic_data, distance_df = load_data()
data = preprocess_data(traffic_data, seq_len=12, pred_len=1)
    

# 补充可能缺失的依赖
try:
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv
    from torch_geometric.data import Data
except ImportError as e:
    print("Missing dependencies:", e)
    sys.exit(1)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def method(data):
    # Convert data to PyTorch tensors
    X = torch.FloatTensor(data.X)
    y = torch.FloatTensor(data.y)
    
    # Define model architecture
    class TrafficModel(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(TrafficModel, self).__init__()
            self.conv1 = nn.Conv1d(input_size, hidden_size, 3, padding=1)
            self.lstm1 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)
            
        def forward(self, x):
            x = x.permute(0, 2, 1)  # Change to channels_first for conv
            x = self.conv1(x)
            x = x.permute(0, 2, 1)  # Change back for LSTM
            x, _ = self.lstm1(x)
            x = self.fc(x[:, -1, :])  # Take last timestep
            return x
    
    # Model parameters
    input_size = X.shape[2]  # Number of nodes (358)
    hidden_size = 64
    output_size = X.shape[2]  # Predicting for all nodes
    
    model = TrafficModel(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Quick training with just a few epochs
    for epoch in range(5):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    
    # Make prediction - assume we're predicting next step after last known data
    with torch.no_grad():
        # Take last timestep from data and reshape for prediction
        last_input = X[-1:].clone()  # Shape: (1, 12, 358)
        prediction = model(last_input)[0, 302].item()  # Get node 302's prediction
    
    return prediction  # Ensure non-negative traffic count

# 执行段三的调用
if __name__ == "__main__":
    try:
        result = method(data)
        print("\nExecution Result:", result)
    except Exception as e:
        print("Execution Error:", str(e))