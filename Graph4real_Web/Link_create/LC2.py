import numpy as np
import networkx as nx
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges, negative_sampling
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score, accuracy_score

# 读取原始数据集
def load_web_of_chrome_data(file_path):
    edges = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#'):  # 跳过注释行
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                src, dst = int(parts[0]), int(parts[1])
                edges.append((src, dst))
    return edges

# 采样子图
def sample_subgraph(edges, target_nodes=3000, seed=42):
    G = nx.DiGraph()
    G.add_edges_from(edges)
    
    # 确保我们从一个连通分量开始
    largest_cc = max(nx.weakly_connected_components(G), key=len)
    
    # 从连通分量中选择初始节点集
    if len(largest_cc) > target_nodes:
        # 随机选择一个种子节点
        random.seed(seed)
        seed_node = random.choice(list(largest_cc))
        
        # 使用BFS扩展节点集
        sampled_nodes = set([seed_node])
        frontier = [seed_node]
        
        while len(sampled_nodes) < target_nodes and frontier:
            current = frontier.pop(0)
            neighbors = list(G.successors(current)) + list(G.predecessors(current))
            random.shuffle(neighbors)
            
            for neighbor in neighbors:
                if neighbor not in sampled_nodes:
                    sampled_nodes.add(neighbor)
                    frontier.append(neighbor)
                    if len(sampled_nodes) >= target_nodes:
                        break
        
        # 获取子图
        subgraph = G.subgraph(sampled_nodes).copy()
    else:
        # 如果最大连通分量小于目标节点数，就直接使用它
        subgraph = G.subgraph(largest_cc).copy()
    
    # 重映射节点ID为连续整数
    mapping = {old_id: new_id for new_id, old_id in enumerate(subgraph.nodes())}
    nx.relabel_nodes(subgraph, mapping, copy=False)
    
    return subgraph, mapping

# 主函数
def prepare_link_prediction_dataset(file_path, output_path, target_nodes=3000):
    # 加载数据
    edges = load_web_of_chrome_data(file_path)
    print(f"原始数据集有 {len(set([e[0] for e in edges] + [e[1] for e in edges]))} 个节点和 {len(edges)} 条边")
    
    # 采样子图
    subgraph, mapping = sample_subgraph(edges, target_nodes)
    print(f"采样子图有 {subgraph.number_of_nodes()} 个节点和 {subgraph.number_of_edges()} 条边")
    
    # 保存子图到文件
    with open(output_path, 'w') as f:
        f.write("# Web of Chrome 子图数据集用于链接预测\n")
        f.write(f"# 节点数: {subgraph.number_of_nodes()}\n")
        f.write(f"# 边数: {subgraph.number_of_edges()}\n")
        f.write("# FromNodeID ToNodeID\n")
        for u, v in subgraph.edges():
            f.write(f"{u} {v}\n")
    
    return subgraph


# 将NetworkX图转换为PyTorch Geometric数据格式
def convert_to_pyg_format(G):
    # 创建边索引
    edge_index = torch.tensor([[u, v] for u, v in G.edges()], dtype=torch.long).t()
    
    # 创建简单的节点特征（这里使用节点度作为特征）
    node_degrees = dict(G.degree())
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    
    x = torch.zeros((G.number_of_nodes(), 3), dtype=torch.float)
    for node in G.nodes():
        x[node, 0] = node_degrees[node]
        x[node, 1] = in_degrees[node]
        x[node, 2] = out_degrees[node]
    
    data = Data(x=x, edge_index=edge_index)
    return data

# 定义GNN模型
class GCNLinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNLinkPredictor, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.classifier = nn.Sequential(
            nn.Linear(out_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )
    
    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    
    def decode(self, z, edge_index):
        src, dst = edge_index
        return self.classifier(torch.cat([z[src], z[dst]], dim=1))
    
    def forward(self, x, edge_index, pos_edge_index, neg_edge_index):
        z = self.encode(x, edge_index)
        
        pos_pred = self.decode(z, pos_edge_index)
        neg_pred = self.decode(z, neg_edge_index)
        
        return pos_pred, neg_pred

# 训练和评估链接预测模型
def train_and_evaluate(data, epochs=100):
    # 准备训练数据
    data = train_test_split_edges(data, val_ratio=0.1, test_ratio=0.2)
    
    # 初始化模型
    model = GCNLinkPredictor(in_channels=data.x.size(1), 
                            hidden_channels=64, 
                            out_channels=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()
    
    # 训练模型
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # 构造训练边索引
        edge_index = data.train_pos_edge_index
        
        # 采样负边
        neg_edge_index = negative_sampling(
            edge_index=edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=edge_index.size(1)
        )
        
        # 前向传播
        pos_pred, neg_pred = model(data.x, edge_index, edge_index, neg_edge_index)
        
        # 计算损失
        pos_loss = criterion(pos_pred, torch.ones_like(pos_pred))
        neg_loss = criterion(neg_pred, torch.zeros_like(neg_pred))
        loss = pos_loss + neg_loss
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 验证
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                # 在验证集上评估
                val_auc, val_acc = evaluate(model, data, data.val_pos_edge_index, data.val_neg_edge_index)
                print(f'Epoch: {epoch+1:03d}, Loss: {loss.item():.4f}, Val AUC: {val_auc:.4f}, Val Acc: {val_acc:.4f}')
    
    # 在测试集上评估最终模型
    model.eval()
    with torch.no_grad():
        test_auc, test_acc = evaluate(model, data, data.test_pos_edge_index, data.test_neg_edge_index)
        print(f'最终测试结果 - AUC: {test_auc:.4f}, Accuracy: {test_acc:.4f}')
    
    return model, test_auc, test_acc

# 评估函数
def evaluate(model, data, pos_edge_index, neg_edge_index):
    # 获取节点嵌入
    z = model.encode(data.x, data.train_pos_edge_index)
    
    # 预测正边
    pos_pred = model.decode(z, pos_edge_index)
    pos_pred = torch.sigmoid(pos_pred)
    
    # 预测负边
    neg_pred = model.decode(z, neg_edge_index)
    neg_pred = torch.sigmoid(neg_pred)
    
    # 计算指标
    pred = torch.cat([pos_pred, neg_pred], dim=0).cpu().numpy()
    true = torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))], dim=0).cpu().numpy()
    
    auc = roc_auc_score(true, pred)
    acc = accuracy_score(true, pred > 0.5)
    
    return auc, acc


def main():
    # 数据集路径（需要根据实际情况修改）
    input_file = "Web/Link_create/web-Google_node.txt"
    output_file = "Web/Link_create/web-of-chrome-subgraph.txt"
    
    # 准备数据集
    subgraph = prepare_link_prediction_dataset(input_file, output_file)
    
    # 转换为PyG格式
    data = convert_to_pyg_format(subgraph)
    print(f"PyG数据: {data}")
    
    # 训练和评估模型
    model, test_auc, test_acc = train_and_evaluate(data, epochs=100)
    
    print("\n链接预测任务结果:")
    print(f"测试集AUC: {test_auc:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")
    
    # 输出结果报告
    with open("Web/Link_create/link_prediction_results.txt", 'w') as f:
        f.write("Web of Chrome子图链接预测结果\n")
        f.write(f"子图节点数: {subgraph.number_of_nodes()}\n")
        f.write(f"子图边数: {subgraph.number_of_edges()}\n")
        f.write(f"测试集AUC: {test_auc:.4f}\n")
        f.write(f"测试集准确率: {test_acc:.4f}\n")

if __name__ == "__main__":
    main()
