# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# # 设置随机种子以确保结果可复现
# torch.manual_seed(42)
# np.random.seed(42)

# # 数据路径
# npz_path = "/home/data2t1/wangrongzheng/GTAgent/zTrain_test/transport/PEMS03.npz"
# csv_path = "/home/data2t1/wangrongzheng/GTAgent/zTrain_test/transport/PEMS03.csv"

# class TrafficData:
#     def __init__(self, X, y):
#         self.X = X
#         self.y = y

# # 加载数据
# def load_data():
#     # 加载交通流量数据
#     data = np.load(npz_path)
#     traffic_data = data['data']  # 形状为 (26208, 358, 1)
    
#     # 加载距离数据
#     distance_df = pd.read_csv(csv_path)
    
#     return traffic_data, distance_df

# # 数据预处理
# def preprocess_data(traffic_data, seq_len=12, pred_len=1):
#     """
#     将原始数据处理成序列预测格式
#     seq_len: 输入序列长度（过去的时间步）
#     pred_len: 预测序列长度（未来的时间步）
#     """
#     n_samples = traffic_data.shape[0] - seq_len - pred_len + 1
#     n_nodes = traffic_data.shape[1]
#     n_features = traffic_data.shape[2]
    
#     X = np.zeros((n_samples, seq_len, n_nodes, n_features))
#     y = np.zeros((n_samples, n_nodes, n_features))
    
#     for i in range(n_samples):
#         X[i] = traffic_data[i:i+seq_len]
#         y[i] = traffic_data[i+seq_len+pred_len-1]
    
#     # 重塑X为(样本数, 序列长度, 节点数*特征数)
#     X = X.reshape(n_samples, seq_len, -1)
#     # 重塑y为(样本数, 节点数*特征数)
#     y = y.reshape(n_samples, -1)
    
#     return TrafficData(X, y)  # 返回封装好的数据对象

# traffic_data, distance_df = load_data()
# data = preprocess_data(traffic_data, seq_len=12, pred_len=1)


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
npz_path = "/home/data2t1/wangrongzheng/GTAgent/zTrain_test/transport/PEMS03.npz"
csv_path = "/home/data2t1/wangrongzheng/GTAgent/zTrain_test/transport/PEMS03.csv"
output_csv_path = "/home/data2t1/wangrongzheng/GTAgent/zTrain_test/transport/PEMS03_last12.csv"  # 新增：保存最后12个时刻的CSV路径

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
    last_12 = traffic_data[-13:]  # 获取最后12个时刻的数据
    # 将3D数组转换为2D (12, 358)并保存为CSV
    last_12_2d = last_12.reshape(13, -1)
    pd.DataFrame(last_12_2d).to_csv(output_csv_path, index=False, header=False)
    print(f"已保存最后12个时刻的数据到: {output_csv_path}")
    
    # 去掉最后12个时刻的数据
    traffic_data = traffic_data[:-12]
    
    # 加载距离数据
    distance_df = pd.read_csv(csv_path)
    
    return traffic_data, distance_df

# 数据预处理
def preprocess_data(traffic_data, seq_len=12, pred_len=1):
    """
    将原始数据处理成序列预测格式
    seq_len: 输入序列长度（过去的时间步）
    pred_len: 预测序列长度（未来的时间步）
    """
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



import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

class LSTMTrafficPredictor:
    def __init__(self, input_size=358, hidden_size=64, output_size=358, num_layers=2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LSTMNetwork(input_size, hidden_size, output_size, num_layers).to(self.device)
        self.scaler = MinMaxScaler()
        
    def fit(self, X, y, epochs=50, batch_size=32, lr=0.001):
        # 数据预处理
        X = self.scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        y = self.scaler.transform(y)
        
        # 转换为PyTorch张量
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # 训练模型
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            for i in range(0, len(X), batch_size):
                batch_X = X_tensor[i:i+batch_size]
                batch_y = y_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
    def predict(self, X):
        # 数据预处理
        X = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # 预测
        with torch.no_grad():
            predictions = self.model(X_tensor)
        
        # 反归一化
        predictions = self.scaler.inverse_transform(predictions.cpu().numpy())
        return predictions

class LSTMNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMNetwork, self).__init__()
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
    # 初始化预测器
    predictor = LSTMTrafficPredictor()
    
    # 训练模型
    predictor.fit(data.X, data.y)
    
    # 获取最后一个样本作为输入进行预测
    last_sample = data.X[-1:].reshape(1, 12, 358)
    
    # 预测下一时刻所有节点的流量
    predictions = predictor.predict(last_sample)
    
    # 返回节点302的预测值
    return predictions[0, 1].item()


next_traffic_flow = method(data)
print(next_traffic_flow)

