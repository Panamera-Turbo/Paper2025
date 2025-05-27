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
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
import networkx as nx
import numpy as np

def method(data):
    # Check for GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert NetworkX graph to PyTorch Geometric format
    edge_index = torch.tensor(list(data.edges())).t().contiguous().to(device)
    num_nodes = data.number_of_nodes()
    
    # Generate node features (using degree as simple feature)
    x = torch.tensor([data.degree(n) for n in data.nodes()], dtype=torch.float).view(-1, 1).to(device)
    
    # Create positive and negative samples for training
    pos_edges = edge_index.t().cpu().numpy()
    neg_edges = []
    
    # Generate negative samples (non-existent edges)
    adj = nx.to_numpy_array(data)
    non_edges = np.column_stack(np.where(adj == 0))
    np.random.shuffle(non_edges)
    neg_edges = non_edges[:len(pos_edges)]
    
    # Combine positive and negative samples
    edges = np.vstack([pos_edges, neg_edges])
    labels = np.hstack([np.ones(len(pos_edges)), np.zeros(len(neg_edges))])
    
    # Split into train/test
    edges_train, edges_test, y_train, y_test = train_test_split(edges, labels, test_size=0.2)
    
    # Convert to tensors
    edges_train = torch.tensor(edges_train, dtype=torch.long).to(device)
    edges_test = torch.tensor(edges_test, dtype=torch.long).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float).to(device)
    
    # Define GAT model
    class GATLinkPredictor(nn.Module):
        def __init__(self, in_features, out_features):
            super().__init__()
            self.conv1 = GATConv(in_features, 16, heads=4)
            self.conv2 = GATConv(16*4, out_features, heads=1)
            
        def forward(self, x, edge_index):
            x = F.elu(self.conv1(x, edge_index))
            x = self.conv2(x, edge_index)
            return x
        
        def predict(self, x, edge_index, edge):
            h = self.forward(x, edge_index)
            return torch.sigmoid((h[edge[0]] * h[edge[1]]).sum(dim=-1))
    
    # Initialize model
    model = GATLinkPredictor(x.size(1), 16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    
    # Training loop
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model.predict(x, edge_index, edges_train.t())
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()
    
    # Evaluate on test edge
    model.eval()
    with torch.no_grad():
        test_edge = torch.tensor([[int('0302'), int('0714')]], dtype=torch.long).t().to(device)
        prob = model.predict(x, edge_index, test_edge).item()
    
    return prob > 0.5


print(method(data))