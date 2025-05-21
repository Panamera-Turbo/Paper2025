import random

def create_directed_graph():
    # Initialize a directed graph
    G = nx.DiGraph()
    
    # Add 500 nodes
    num_nodes = 500
    G.add_nodes_from(range(num_nodes))
    
    # Add 5000 random edges
    num_edges = 5000
    edges = set()
    while len(edges) < num_edges:
        u = random.randint(0, num_nodes - 1)
        v = random.randint(0, num_nodes - 1)
        if u != v and (u, v) not in edges and not (u == 123 and v == 456) and not (u == 456 and v == 123):  # Ensure no edge between 123 and 456
            edges.add((u, v))
    
    G.add_edges_from(edges)
    return G


import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data


class GCN(torch.nn.Module):
    def __init__(self, num_features):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, 2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

def method(Graph, user_a, user_b):
    # Convert NetworkX graph to PyTorch Geometric data
    edge_index = torch.tensor(list(Graph.edges)).t().contiguous()
    num_nodes = Graph.number_of_nodes()
    
    # Create feature matrix (for simplicity, using identity matrix)
    x = torch.eye(num_nodes)
    
    # Create GCN model
    model = GCN(num_features=num_nodes)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        z = model(x, edge_index)

    # Get embeddings for the two users
    user_a_embedding = z[user_a]
    user_b_embedding = z[user_b]
    
    # Calculate similarity (cosine similarity)
    similarity = F.cosine_similarity(user_a_embedding.unsqueeze(0), user_b_embedding.unsqueeze(0))
    
    # Assuming a threshold for potential link prediction (0.5 as an example)
    threshold = 0.5
    prediction = similarity.item() > threshold
    
    return prediction

G = create_directed_graph()  # Your NetworkX graph should be defined here
print(method(G, 123, 456))
