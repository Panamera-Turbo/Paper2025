import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx, negative_sampling
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

# 1. 读取数据并创建NetworkX图
def read_graph_from_txt(file_path):
    G = nx.Graph()
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # 跳过注释行(以#开头的行)
    edge_list = []
    for line in lines:
        if not line.startswith('#'):
            source, target = map(int, line.strip().split())
            edge_list.append((source, target))
    
    G.add_edges_from(edge_list)
    return G

# 2. 定义GNN模型
class LinkPredictionGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(LinkPredictionGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        
    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    
    def decode(self, z, edge_index):
        row, col = edge_index
        return (z[row] * z[col]).sum(dim=1)
    
    def decode_all(self, z):
        prob_adj = z @ z.t()
        return prob_adj

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        return z

# 主函数
def main():
    # 读取图数据
    file_path = "Web/Link_create/web-of-chrome-subgraph.txt"  # 请将此处更改为您的文件路径
    G = read_graph_from_txt(file_path)
    
    print(f"图读取完成. 节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()}")
    
    # 为networkx图添加节点特征（如果没有特征，使用节点度数作为特征）
    for node in G.nodes():
        G.nodes[node]['feat'] = [G.degree(node)]
    
    # 将NetworkX图转换为PyTorch Geometric数据格式
    data = from_networkx(G)
    
    # 如果没有节点特征，创建一个单位矩阵作为节点特征
    if not hasattr(data, 'x') or data.x is None:
        data.x = torch.eye(data.num_nodes)
    
    # 确保特征是浮点型
    data.x = data.x.float()
    
    # 划分训练集和测试集
    edge_index = data.edge_index
    num_edges = edge_index.size(1)
    
    # 随机选择80%的边作为训练集
    perm = torch.randperm(num_edges)
    train_edges_idx = perm[:int(0.8 * num_edges)]
    test_edges_idx = perm[int(0.8 * num_edges):]
    
    train_edge_index = edge_index[:, train_edges_idx]
    test_edge_index = edge_index[:, test_edges_idx]
    
    # 为测试集生成负样本（不存在的边）
    test_neg_edge_index = negative_sampling(
        edge_index=train_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=test_edges_idx.size(0))
    
    # 初始化模型
    device = torch.device('cpu')
    model = LinkPredictionGNN(data.x.size(1), 64, 32).to(device)
    data = data.to(device)
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # 训练模型
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        
        # 编码节点
        z = model.encode(data.x, train_edge_index)
        
        # 为训练生成正样本和负样本
        pos_edge_index = train_edge_index
        neg_edge_index = negative_sampling(
            edge_index=train_edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=train_edge_index.size(1))
        
        pos_prob = model.decode(z, pos_edge_index)
        neg_prob = model.decode(z, neg_edge_index)
        
        # 使用BCEWithLogitsLoss，内部会应用sigmoid
        pos_loss = F.binary_cross_entropy_with_logits(pos_prob, torch.ones_like(pos_prob))
        neg_loss = F.binary_cross_entropy_with_logits(neg_prob, torch.zeros_like(neg_prob))
        loss = pos_loss + neg_loss
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch: {epoch+1:02d}, Loss: {loss:.4f}')
    
    # 评估模型
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, train_edge_index)
        
        # 测试集
        pos_prob = model.decode(z, test_edge_index).sigmoid()
        neg_prob = model.decode(z, test_neg_edge_index).sigmoid()
        
        # 计算预测结果
        y_pred = torch.cat([pos_prob, neg_prob]).cpu().numpy()
        y_true = torch.cat([torch.ones(pos_prob.size(0)), torch.zeros(neg_prob.size(0))]).cpu().numpy()
        
        # 计算准确率和AUC
        y_pred_binary = (y_pred > 0.5).astype(float)
        accuracy = accuracy_score(y_true, y_pred_binary)
        auc = roc_auc_score(y_true, y_pred)
        
        print(f'准确率: {accuracy:.4f}')
        print(f'AUC: {auc:.4f}')

if __name__ == "__main__":
    main()
