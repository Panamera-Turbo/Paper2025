import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# 1. 加载数据
def load_data(csv_path, npz_path=None):
    """
    加载交通流量数据
    """
    # 读取 CSV 文件
    traffic_data = pd.read_csv(csv_path)
    print("CSV 数据加载成功，形状为：", traffic_data.shape)
    
    # 如果有时间戳列，可以将其转换为时间序列索引
    if 'timestamp' in traffic_data.columns:
        traffic_data['timestamp'] = pd.to_datetime(traffic_data['timestamp'])
        traffic_data.set_index('timestamp', inplace=True)
    
    # 读取 NPZ 文件（如果提供）
    adj_matrix = None
    if npz_path:
        npz_data = np.load(npz_path)
        print("NPZ 数据加载成功，包含以下键值：", list(npz_data.keys()))
        if 'adj_matrix' in npz_data:
            adj_matrix = npz_data['adj_matrix']
            print("邻接矩阵形状为：", adj_matrix.shape)
    
    return traffic_data, adj_matrix

# 2. 数据预处理
def preprocess_data(data, seq_length=12, pred_length=1):
    """
    数据预处理：归一化、时间序列转换
    """
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(len(data_scaled) - seq_length - pred_length + 1):
        X.append(data_scaled[i:i+seq_length])
        y.append(data_scaled[i+seq_length:i+seq_length+pred_length])
    X, y = np.array(X), np.array(y)
    print("数据预处理完成，X 形状为：", X.shape, "y 形状为：", y.shape)
    return X, y, scaler

# 3. 使用 KNN 模型进行时序预测
def train_and_predict_knn(X_train, y_train, X_test, y_test, scaler, features, n_neighbors=5):
    """
    使用 KNN 模型进行训练和预测
    """
    # 将数据从 3D 转换为 2D，KNN 不支持 3D 数据
    X_train_2d = X_train.reshape(X_train.shape[0], -1)  # (样本数, seq_length * 特征数)
    X_test_2d = X_test.reshape(X_test.shape[0], -1)
    y_train_2d = y_train.reshape(y_train.shape[0], -1)  # (样本数, pred_length)
    y_test_2d = y_test.reshape(y_test.shape[0], -1)
    
    # 创建 KNN 模型
    knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn.fit(X_train_2d, y_train_2d)  # 训练模型
    
    # 使用模型进行预测
    y_pred_2d = knn.predict(X_test_2d)
    
    # 将预测值和真实值的形状调整为与原始数据一致
    n_features = features.shape[1]  # 原始数据的特征数量
    y_pred_full = np.zeros((y_pred_2d.shape[0], n_features))
    y_true_full = np.zeros((y_test_2d.shape[0], n_features))

    # 假设目标值是原始数据的第一个特征（根据数据集调整索引）
    y_pred_full[:, 0] = y_pred_2d[:, 0]
    y_true_full[:, 0] = y_test_2d[:, 0]

    # 将预测值和真实值反归一化
    y_test_rescaled = scaler.inverse_transform(y_true_full)[:, 0]  # 提取目标值的反归一化结果
    y_pred_rescaled = scaler.inverse_transform(y_pred_full)[:, 0]  # 提取目标值的反归一化结果

    return y_test_rescaled, y_pred_rescaled

# 4. 主函数
def main():
    # 文件路径
    csv_path = "/home/data2t1/wangrongzheng/GTAgent/zTrain_test/transport/PEMS03.csv"  # 替换为你的 CSV 文件路径
    npz_path = "/home/data2t1/wangrongzheng/GTAgent/zTrain_test/transport/PEMS03.npz"  # 替换为你的 NPZ 文件路径（如果有）
    
    # 加载数据
    traffic_data, adj_matrix = load_data(csv_path, npz_path)
    
    # 提取特征和目标（假设所有列都是流量数据）
    features = traffic_data.values
    
    # 数据预处理
    seq_length = 12  # 使用过去 12 个时间步的数据
    pred_length = 1  # 预测下一个时间步
    X, y, scaler = preprocess_data(features, seq_length, pred_length)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("训练集和测试集划分完成")
    
    # 使用 KNN 模型进行训练和预测
    y_test_rescaled, y_pred_rescaled = train_and_predict_knn(X_train, y_train, X_test, y_test, scaler, features)

    # 打印结果
    print("真实值（部分）：", y_test_rescaled[:5])
    print("预测值（部分）：", y_pred_rescaled[:5])

    # 计算误差
    mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
    print(f"均方误差 (MSE): {mse:.4f}")

if __name__ == "__main__":
    main()
