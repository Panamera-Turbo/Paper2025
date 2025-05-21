import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
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
    # 1. 计算样本数量和数据维度
    n_samples = traffic_data.shape[0] - seq_len - pred_len + 1
    n_nodes = traffic_data.shape[1]
    n_features = traffic_data.shape[2]
    
    # 2. 初始化输入输出容器
    X = np.zeros((n_samples, seq_len, n_nodes, n_features))
    y = np.zeros((n_samples, n_nodes, n_features))
    
    # 3. 定义异常值处理函数（基于IQR四分位距）
    def remove_outliers(data):
        q1 = np.percentile(data, 25)  # 第一四分位数
        q3 = np.percentile(data, 75)  # 第三四分位数
        iqr = q3 - q1  # 四分位距
        lower_bound = q1 - 1.5 * iqr  # 下限
        upper_bound = q3 + 1.5 * iqr  # 上限
        
        # 将超出范围的值裁剪到边界
        return np.clip(data, lower_bound, upper_bound)
    
    # 4. 对每个节点的时序数据分别处理异常值
    for node in range(n_nodes):
        traffic_data[:, node, 0] = remove_outliers(traffic_data[:, node, 0])
    
    # 5. 按节点进行Min-Max归一化（保留极值用于后续反归一化）
    min_vals = np.min(traffic_data, axis=0)  # 每个节点的最小值
    max_vals = np.max(traffic_data, axis=0)  # 每个节点的最大值
    
    # 处理最大值等于最小值的情况（避免除零错误）
    max_vals[max_vals == min_vals] = 1
    traffic_data = (traffic_data - min_vals) / (max_vals - min_vals)
    
    # 6. 构建时间序列样本
    for i in range(n_samples):
        X[i] = traffic_data[i:i+seq_len]  # 输入序列
        y[i] = traffic_data[i+seq_len+pred_len-1]  # 预测目标
    
    # 7. 调整形状适应模型输入
    X = X.reshape(n_samples, seq_len, -1)  # 展平节点和特征维度
    y = y.reshape(n_samples, -1)
    
    # 8. 封装数据并保存归一化参数
    traffic_data_obj = TrafficData(X, y)
    traffic_data_obj.min_vals = min_vals  # 存储最小值（用于预测结果反归一化）
    traffic_data_obj.max_vals = max_vals  # 存储最大值
    
    return traffic_data_obj, min_vals, max_vals

# 补充可能缺失的依赖
try:
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv
    from torch_geometric.data import Data
except ImportError as e:
    print("Missing dependencies:", e)
    sys.exit(1)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def method(data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Prepare data
    X = torch.tensor(data.X, dtype=torch.float32)
    y = torch.tensor(data.y, dtype=torch.float32)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Model parameters
    input_size = X.shape[2]  # number of nodes
    hidden_size = 64
    output_size = input_size
    num_layers = 2
    
    # Initialize model
    model = LSTMModel(input_size, hidden_size, output_size, num_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    num_epochs = 20
    for epoch in range(num_epochs):
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Predict for node 123 (0-based index 122)
    with torch.no_grad():
        last_sequence = torch.tensor(data.X[-1:], dtype=torch.float32).to(device)
        prediction = model(last_sequence)
        node_123_prediction = prediction[0, 301].item()
    
    return node_123_prediction

# 主程序执行
if __name__ == "__main__":
    # 加载并预处理数据
    traffic_data, distance_df = load_data()
    data, min_vals, max_vals = preprocess_data(traffic_data, seq_len=12, pred_len=1)
    
    try:
        result = method(data)
        node_min = min_vals[301, 0]
        node_max = max_vals[301, 0]
        print("Execution Result:", result * (node_max - node_min) + node_min)
    except Exception as e:
        print("Execution Error:", str(e))