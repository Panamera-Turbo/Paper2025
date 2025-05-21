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
def read_graph_from_txt():
    file_path = "Web/Link_create/web-of-chrome-subgraph.txt" 
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

data = read_graph_from_txt()

import torch
import networkx as nx
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import from_networkx

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        
    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)
    
    def decode(self, z, edge_label_index):
        src, dst = edge_label_index
        return (z[src] * z[dst]).sum(dim=-1)

def method(data):
    # Check for GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert NetworkX graph to PyTorch Geometric format
    pyg_data = from_networkx(data)
    
    # Add random node features if none exist
    if pyg_data.x is None:
        pyg_data.x = torch.randn((data.number_of_nodes(), 16))
    
    # Create edge index for all possible edges (including non-existent ones)
    num_nodes = data.number_of_nodes()
    all_edges = torch.combinations(torch.arange(num_nodes), r=2).t()
    
    # Prepare positive and negative edges for training
    pos_edges = pyg_data.edge_index
    neg_edges = torch.randint(0, num_nodes, (2, pos_edges.size(1)), dtype=torch.long)
    
    # Combine positive and negative edges
    edge_label_index = torch.cat([pos_edges, neg_edges], dim=1)
    edge_label = torch.cat([
        torch.ones(pos_edges.size(1)),
        torch.zeros(neg_edges.size(1))
    ], dim=0)
    
    # Initialize model
    model = LinkPredictor(pyg_data.x.size(1), 64, 32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # Move data to device
    pyg_data = pyg_data.to(device)
    edge_label_index = edge_label_index.to(device)
    edge_label = edge_label.to(device)
    
    # Train the model
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        z = model.encode(pyg_data.x, pyg_data.edge_index)
        out = model.decode(z, edge_label_index).view(-1)
        loss = criterion(out, edge_label)
        loss.backward()
        optimizer.step()
    
    # Predict for specific nodes (0302 and 0714)
    node1 = 302  # assuming node IDs are integers
    node2 = 714
    if node1 >= num_nodes or node2 >= num_nodes:
        return False
    
    with torch.no_grad():
        z = model.encode(pyg_data.x, pyg_data.edge_index)
        pred = model.decode(z, torch.tensor([[node1], [node2]], device=device))
        prob = torch.sigmoid(pred).item()
    
    return prob > 0.5



print(method(data))