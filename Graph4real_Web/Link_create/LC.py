import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.utils import train_test_split_edges

# 读取数据集
def load_data(file_path):
    # 读取边列表
    df = pd.read_csv(file_path, sep='\t', header=None, names=['source', 'target'])
    return df

# 采样子图
def sample_subgraph(edge_df, target_nodes=3000, density_factor=2.0):
    """
    采样一个节点数量约为target_nodes的子图，并确保边密度较高
    density_factor: 控制边密度的系数，值越大，采样的边越多
    """
    # 构建完整图
    G = nx.from_pandas_edgelist(edge_df, 'source', 'target', create_using=nx.DiGraph())
    
    # 获取所有节点
    all_nodes = list(G.nodes())
    
    # 策略：选择度数较高的节点为起点，然后进行BFS
    degrees = dict(G.degree())
    sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
    
    # 从高度数节点开始
    start_node = sorted_nodes[0][0]
    
    # 使用BFS获取连通子图
    subgraph_nodes = set()
    queue = [start_node]
    visited = set(queue)
    
    while queue and len(subgraph_nodes) < target_nodes:
        current = queue.pop(0)
        subgraph_nodes.add(current)
        
        # 获取邻居
        neighbors = list(G.neighbors(current))
        # 优先添加度数高的邻居
        neighbors.sort(key=lambda x: degrees.get(x, 0), reverse=True)
        
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    # 如果BFS不足以找到足够的节点，随机添加高度数节点
    if len(subgraph_nodes) < target_nodes:
        remaining_nodes = [n for n, _ in sorted_nodes if n not in subgraph_nodes]
        additional_nodes = min(target_nodes - len(subgraph_nodes), len(remaining_nodes))
        subgraph_nodes.update(remaining_nodes[:additional_nodes])
    
    # 获取子图
    subgraph = G.subgraph(list(subgraph_nodes))
    
    # 确保边密度足够高 - 可以额外添加一些边
    subgraph_nodes_list = list(subgraph_nodes)
    
    # 为了增加密度，连接一些高度数节点
    high_degree_nodes = subgraph_nodes_list[:int(len(subgraph_nodes_list) * 0.1)]
    additional_edges = []
    
    # 将高度数节点相互连接
    for i in range(len(high_degree_nodes)):
        for j in range(i+1, len(high_degree_nodes)):
            if np.random.random() < density_factor * 0.1:  # 控制添加边的概率
                additional_edges.append((high_degree_nodes[i], high_degree_nodes[j]))
    
    # 重新构建具有额外边的子图
    H = nx.DiGraph()
    H.add_nodes_from(subgraph_nodes)
    H.add_edges_from(subgraph.edges())
    H.add_edges_from(additional_edges)
    
    # 转换为无向图以便于后续处理
    H_undirected = H.to_undirected()
    
    # 获取最大连通分量
    largest_cc = max(nx.connected_components(H_undirected), key=len)
    if len(largest_cc) < target_nodes * 0.8:  # 如果最大连通分量太小
        print(f"警告: 最大连通分量只有 {len(largest_cc)} 节点")
    
    final_subgraph = H.subgraph(largest_cc)
    
    # 重新映射节点ID为连续整数
    mapping = {old_id: new_id for new_id, old_id in enumerate(final_subgraph.nodes())}
    final_subgraph = nx.relabel_nodes(final_subgraph, mapping)
    
    # 转换为边列表
    edge_list = list(final_subgraph.edges())
    edge_df_new = pd.DataFrame(edge_list, columns=['source', 'target'])
    
    return edge_df_new, final_subgraph

# 假设数据文件路径
file_path = "Web/Link_create/web-Google_node.txt"  # 请替换为实际文件路径

# 读取数据并采样子图
try:
    edge_df = load_data(file_path)
    subgraph_df, G_sub = sample_subgraph(edge_df, target_nodes=3000, density_factor=2.5)
    
    print(f"原始图: {len(edge_df['source'].unique())} 节点, {len(edge_df)} 边")
    print(f"采样子图: {G_sub.number_of_nodes()} 节点, {G_sub.number_of_edges()} 边")
    print(f"平均度数: {2 * G_sub.number_of_edges() / G_sub.number_of_nodes():.2f}")
    
    # 保存采样的子图用于后续实验
    subgraph_df.to_csv("Web/Link_create/web_chrome_sampled.csv", index=False)
    
except Exception as e:
    print(f"读取数据或采样子图时出错: {e}")


# 将采样的子图转换为PyG数据格式
def create_pyg_data(edge_df):
    # 获取所有的节点ID
    all_nodes = set(edge_df['source'].unique()) | set(edge_df['target'].unique())
    node_mapping = {node: i for i, node in enumerate(all_nodes)}
    
    # 将边列表转换为tensor
    edge_index = torch.tensor([
        [node_mapping[src] for src in edge_df['source']],
        [node_mapping[tgt] for tgt in edge_df['target']]
    ], dtype=torch.long)
    
    # 创建PyG数据对象
    data = Data(edge_index=edge_index, num_nodes=len(all_nodes))
    return data

# 定义GNN模型 (Graph Convolutional Network)
class LinkPredictionGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(LinkPredictionGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        
    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x
    
    def decode(self, z, edge_index):
        # 计算边连接的节点的内积
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
    
    def decode_all(self, z):
        # 计算所有节点对之间的内积
        prob_adj = z @ z.t()
        return prob_adj
    
    def forward(self, x, pos_edge_index, neg_edge_index=None):
        z = self.encode(x, pos_edge_index)
        
        pos_pred = self.decode(z, pos_edge_index)
        
        if neg_edge_index is not None:
            neg_pred = self.decode(z, neg_edge_index)
            return pos_pred, neg_pred
        
        return pos_pred

# 准备训练和评估链接预测模型
def train_eval_link_prediction():
    # 加载采样的子图数据
    subgraph_df = pd.read_csv("Web/Link_create/web_chrome_sampled.csv")
    
    # 创建PyG数据
    data = create_pyg_data(subgraph_df)
    
    # 随机特征，如果没有节点特征
    data.x = torch.ones((data.num_nodes, 1), dtype=torch.float)  # 使用简单的常量特征
    
    # 分割数据为训练、验证和测试集
    data = train_test_split_edges(data, val_ratio=0.1, test_ratio=0.2)
    
    # 初始化模型
    model = LinkPredictionGNN(in_channels=1, hidden_channels=64, out_channels=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # 训练模型
    def train():
        model.train()
        optimizer.zero_grad()
        
        # 使用训练边作为正样本
        pos_edge_index = data.train_pos_edge_index
        
        # 随机生成负样本
        neg_edge_index = negative_sampling(
            edge_index=data.train_pos_edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=pos_edge_index.size(1))
        
        z = model.encode(data.x, data.train_pos_edge_index)
        
        pos_out = model.decode(z, pos_edge_index)
        neg_out = model.decode(z, neg_edge_index)
        
        # 合并正负样本的预测
        pred = torch.cat([pos_out, neg_out], dim=0)
        # 构建标签：正样本为1，负样本为0
        true = torch.cat([torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))], dim=0)
        
        loss = criterion(pred, true)
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    # 评估模型
    @torch.no_grad()
    def test(pos_edge_index, neg_edge_index):
        model.eval()
        
        z = model.encode(data.x, data.train_pos_edge_index)
        
        pos_pred = model.decode(z, pos_edge_index)
        neg_pred = model.decode(z, neg_edge_index)
        
        pred = torch.cat([pos_pred, neg_pred], dim=0)
        true = torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))], dim=0)
        
        # 计算AUC和AP分数
        pred_np = pred.cpu().numpy()
        true_np = true.cpu().numpy()
        
        auc = roc_auc_score(true_np, pred_np)
        ap = average_precision_score(true_np, pred_np)
        
        # 计算准确率
        pred_labels = (pred > 0).float()
        accuracy = (pred_labels == true).sum().item() / true.size(0)
        
        return auc, ap, accuracy
    
    # 训练循环
    best_val_auc = 0
    final_test_results = None
    patience = 10
    counter = 0
    
    for epoch in range(1, 101):
        loss = train()
        
        val_auc, val_ap, val_acc = test(data.val_pos_edge_index, data.val_neg_edge_index)
        test_auc, test_ap, test_acc = test(data.test_pos_edge_index, data.test_neg_edge_index)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            final_test_results = (test_auc, test_ap, test_acc)
            counter = 0
        else:
            counter += 1
        
        if counter >= patience:
            print(f"早停: Epoch {epoch}，连续{patience}个epoch没有改进")
            break
        
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}, '
                  f'Val ACC: {val_acc:.4f}, Test AUC: {test_auc:.4f}, Test ACC: {test_acc:.4f}')
    
    # 输出最终结果
    test_auc, test_ap, test_acc = final_test_results
    print(f"最终测试结果 - AUC: {test_auc:.4f}, AP: {test_ap:.4f}, 准确率: {test_acc:.4f}")
    
    return final_test_results

# 运行训练和评估
try:
    auc, ap, accuracy = train_eval_link_prediction()
    print(f"\n链接预测任务结果:\n准确率: {accuracy:.4f}\nAUC: {auc:.4f}\nAP: {ap:.4f}")
    
    # 可视化采样的子图结构
    subgraph_df = pd.read_csv("Web/Link_create/web_chrome_sampled.csv")
    G = nx.from_pandas_edgelist(subgraph_df, 'source', 'target', create_using=nx.DiGraph())
    
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, node_size=10, node_color='blue', alpha=0.7, with_labels=False)
    plt.title(f"Web of Chrome 子图 (节点: {G.number_of_nodes()}, 边: {G.number_of_edges()})")
    plt.savefig("Web/Link_create/web_chrome_subgraph.png")
    plt.close()
    
except Exception as e:
    print(f"链接预测过程中出错: {e}")


def analyze_results(subgraph_df):
    """分析采样子图的特性和模型性能"""
    G = nx.from_pandas_edgelist(subgraph_df, 'source', 'target', create_using=nx.Graph())
    
    # 基本统计数据
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    avg_degree = 2 * num_edges / num_nodes
    
    # 计算聚类系数和连通性
    avg_clustering = nx.average_clustering(G)
    connected_components = list(nx.connected_components(G))
    largest_cc_size = len(max(connected_components, key=len))
    
    # 计算图密度
    density = nx.density(G)
    
    # 输出结果
    print("\n子图特性分析:")
    print(f"节点数: {num_nodes}")
    print(f"边数: {num_edges}")
    print(f"平均度数: {avg_degree:.2f}")
    print(f"图密度: {density:.4f}")
    print(f"平均聚类系数: {avg_clustering:.4f}")
    print(f"最大连通分量大小: {largest_cc_size} ({largest_cc_size/num_nodes*100:.1f}% 的节点)")
    print(f"连通分量数: {len(connected_components)}")
    
    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "avg_degree": avg_degree,
        "density": density,
        "avg_clustering": avg_clustering,
        "largest_cc_ratio": largest_cc_size/num_nodes
    }

# 运行分析
try:
    subgraph_df = pd.read_csv("Web/Link_create/web_chrome_sampled.csv")
    graph_stats = analyze_results(subgraph_df)
except Exception as e:
    print(f"分析结果时出错: {e}")
