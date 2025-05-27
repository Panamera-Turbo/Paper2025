# import sys
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv
# from torch_geometric.data import Data
# from sklearn.metrics import accuracy_score

# # 加载数据
# data = torch.load('Cora/cora/cora_fixed_tfidf.pt')
# # 确保数据是PyG的Data对象格式
# if not isinstance(data, Data):
#     data = Data(x=data.x, edge_index=data.edge_index, y=data.y)

# # 补充可能缺失的依赖
# try:
#     import torch
#     import torch.nn.functional as F
#     from torch_geometric.nn import GCNConv
#     from torch_geometric.data import Data
# except ImportError as e:
#     print("Missing dependencies:", e)
#     sys.exit(1)

# import torch
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv
# from torch_geometric.data import Data
# from sklearn.model_selection import train_test_split

# class GCN(torch.nn.Module):
#     def __init__(self, num_features, num_classes):
#         super(GCN, self).__init__()
#         self.conv1 = GCNConv(num_features, 16)
#         self.conv2 = GCNConv(16, num_classes)
    
#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)
#         x = self.conv2(x, edge_index)
#         return F.log_softmax(x, dim=1)

# def method(data):
#     # Set random seed for reproducibility
#     torch.manual_seed(42)
    
#     # Split data into train/val/test
#     num_nodes = data.x.size(0)
#     train_idx, test_idx = train_test_split(
#         range(num_nodes), test_size=0.3, random_state=42)
#     val_idx, test_idx = train_test_split(
#         test_idx, test_size=0.5, random_state=42)
    
#     # Create masks
#     train_mask = torch.zeros(num_nodes, dtype=torch.bool)
#     val_mask = torch.zeros(num_nodes, dtype=torch.bool)
#     test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
#     train_mask[train_idx] = True
#     val_mask[val_idx] = True
#     test_mask[test_idx] = True
    
#     data.train_mask = train_mask
#     data.val_mask = val_mask
#     data.test_mask = test_mask
    
#     # Initialize model
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = GCN(num_features=data.x.size(1), num_classes=2).to(device)
#     data = data.to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
#     # Train model
#     model.train()
#     for epoch in range(200):
#         optimizer.zero_grad()
#         out = model(data)
#         loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
#         loss.backward()
#         optimizer.step()
    
#     # Evaluate on validation set to select best model
#     model.eval()
#     with torch.no_grad():
#         logits = model(data)
#         pred = logits.argmax(dim=1)
#         acc = (pred[data.val_mask] == data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()
    
#     # Get prediction for node 0302
#     node_idx = 302  # Assuming node IDs are 0-indexed and 0302 represents index 302
#     model.eval()
#     with torch.no_grad():
#         logits = model(data)
#         pred = logits.argmax(dim=1)
#         is_high_impact = bool(pred[node_idx].item() == 1)  # Assuming class 1 represents high-impact
    
#     return is_high_impact

# # 执行段三的调用
# if __name__ == "__main__":
#     try:
#         result = method(data)
#         print("\nExecution Result:", result)
#     except Exception as e:
#         print("Execution Error:", str(e))


import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.metrics import accuracy_score

# Load data
data = torch.load('Cora/cora/cora_fixed_tfidf.pt')
# Ensure data is in PyG Data object format
if not isinstance(data, Data):
    data = Data(x=data.x, edge_index=data.edge_index, y=data.y)

import torch

def method(data):
    # 获取节点302和714的出边目标节点
    edge_index = data.edge_index
    # 找出所有以302为起点的边的终点
    node_302_out = edge_index[1, edge_index[0] == 11]
    # 找出所有以714为起点的边的终点
    node_714_out = edge_index[1, edge_index[0] == 1635]
    # 计算两个集合的交集大小
    common_out = set(node_302_out.tolist()) & set(node_714_out.tolist())
    return len(common_out)



if __name__ == "__main__":
    try:
        result = method(data)
        print("\nExecution Result:", result)
    except Exception as e:
        print("Execution Error:", str(e))