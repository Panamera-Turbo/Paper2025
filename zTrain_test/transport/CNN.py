import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 设置随机种子以保证结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 加载PEMS03数据集
npz_path = "/home/data2t1/wangrongzheng/GTAgent/zTrain_test/transport/PEMS03.npz"
csv_path = "/home/data2t1/wangrongzheng/GTAgent/zTrain_test/transport/PEMS03.csv"

# 加载数据
data = np.load(npz_path)['data']  # (26208, 358, 1)
print(f"数据形状: {data.shape}")

# 加载距离信息
distance_df = pd.read_csv(csv_path)
print(f"距离数据形状: {distance_df.shape}")
print(distance_df.head())

# 定义参数
input_length = 12  # 使用12个时间步预测下一个时间步
batch_size = 64
epochs = 30
learning_rate = 0.001
test_ratio = 0.2
val_ratio = 0.2

# 数据预处理
class TrafficDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.data) - self.seq_len
    
    def __getitem__(self, index):
        x = self.data[index:index + self.seq_len]
        y = self.data[index + self.seq_len]
        return torch.FloatTensor(x), torch.FloatTensor(y)

# 数据标准化
data = data.reshape(data.shape[0], -1)  # 将(26208, 358, 1)转为(26208, 358)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
data_scaled = data_scaled.reshape(-1, 358, 1)  # 转回(26208, 358, 1)

# 划分训练集、验证集和测试集
train_end = int(len(data_scaled) * (1 - test_ratio - val_ratio))
val_end = int(len(data_scaled) * (1 - test_ratio))
train_data = data_scaled[:train_end]
val_data = data_scaled[train_end:val_end]
test_data = data_scaled[val_end:]

print(f"训练集大小: {train_data.shape}")
print(f"验证集大小: {val_data.shape}")
print(f"测试集大小: {test_data.shape}")

# 创建数据加载器
train_dataset = TrafficDataset(train_data, input_length)
val_dataset = TrafficDataset(val_data, input_length)
test_dataset = TrafficDataset(test_data, input_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 构建CNN模型
class CNNModel(nn.Module):
    def __init__(self, input_channels=1, input_length=12, num_nodes=358):
        super(CNNModel, self).__init__()
        
        # CNN层
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1))
        
        # 全连接层
        self.fc1 = nn.Linear(64 * input_length * num_nodes, 512)
        self.fc2 = nn.Linear(512, num_nodes)
        
        # 激活函数
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # 输入形状: [batch_size, input_length, num_nodes, 1]
        # 转换为: [batch_size, 1, input_length, num_nodes]
        x = x.permute(0, 3, 1, 2)
        
        # CNN层
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        # 展平
        x = x.reshape(x.size(0), -1)
        
        # 全连接层
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x.unsqueeze(-1)  # 输出形状: [batch_size, num_nodes, 1]

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
model = CNNModel().to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # 验证
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_cnn_model.pth')
        
        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{epochs}] - "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Time: {epoch_time:.2f}s")
        
    return train_losses, val_losses

# 开始训练
train_losses, val_losses = train_model(model, train_loader, val_loader, 
                                       criterion, optimizer, epochs, device)

# 加载最佳模型
model.load_state_dict(torch.load('best_cnn_model.pth'))

# 在测试集上评估模型
def evaluate_model(model, test_loader, device, scaler):
    model.eval()
    predictions = []
    actual_values = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # 转回CPU并转为numpy
            outputs = outputs.cpu().numpy()
            targets = targets.cpu().numpy()
            
            predictions.append(outputs)
            actual_values.append(targets)
    
    # 合并结果
    predictions = np.vstack(predictions)
    actual_values = np.vstack(actual_values)
    
    # 将预测值和真实值转回原始尺度
    predictions_reshaped = predictions.reshape(predictions.shape[0], -1)
    actual_values_reshaped = actual_values.reshape(actual_values.shape[0], -1)
    
    predictions_original = scaler.inverse_transform(predictions_reshaped)
    actual_values_original = scaler.inverse_transform(actual_values_reshaped)
    
    # 计算评估指标
    mae = mean_absolute_error(actual_values_original, predictions_original)
    rmse = np.sqrt(mean_squared_error(actual_values_original, predictions_original))
    
    return mae, rmse, predictions_original, actual_values_original

# 评估模型
mae, rmse, predictions, actual_values = evaluate_model(model, test_loader, device, scaler)
print(f"测试集上的MAE: {mae:.4f}")
print(f"测试集上的RMSE: {rmse:.4f}")

# 可视化训练过程
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# 可视化预测结果（随机选择一个节点）
node_idx = np.random.randint(0, 358)
plt.subplot(1, 2, 2)
time_steps = np.arange(len(predictions))
plt.plot(time_steps, predictions[:, node_idx], label='Predictions')
plt.plot(time_steps, actual_values[:, node_idx], label='Actual')
plt.xlabel('Time Steps')
plt.ylabel('Traffic Flow')
plt.title(f'Predictions vs Actual for Node {node_idx}')
plt.legend()

plt.tight_layout()
plt.savefig('traffic_prediction_results.png')
plt.show()

print("预测完成！结果已保存为traffic_prediction_results.png")
