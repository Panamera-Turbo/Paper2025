import random
import networkx as nx

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


import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx

class GCNLinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GCNLinkPredictor, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels * 2, 1)

    def forward(self, x, edge_index, node_pair):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)

        # Extract embeddings for the node pair
        node1, node2 = node_pair
        z1, z2 = x[node1], x[node2]
        z = torch.cat([z1, z2], dim=0)

        # Predict link
        return torch.sigmoid(self.fc(z))

def link_prediction(graph, follower, followed, epochs=5, hidden_channels=16, lr=0.01):
    # Convert NetworkX graph to PyTorch Geometric graph
    data = from_networkx(graph)
    num_nodes = data.num_nodes
    data.x = torch.eye(num_nodes)  # Use identity matrix as initial node features

    # Prepare the model and optimizer
    model = GCNLinkPredictor(in_channels=num_nodes, hidden_channels=hidden_channels)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # Prepare training data
    positive_edges = data.edge_index.T.tolist()
    negative_edges = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if (i, j) not in positive_edges and i != j]
    negative_edges = negative_edges[:len(positive_edges)]  # Balance positive and negative samples

    # Create labels
    edges = positive_edges + negative_edges
    labels = torch.tensor([1] * len(positive_edges) + [0] * len(negative_edges), dtype=torch.float)

    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Shuffle edges and labels
        perm = torch.randperm(len(edges))
        edges = [edges[i] for i in perm]
        labels = labels[perm]

        loss = 0
        for edge, label in zip(edges, labels):
            pred = model(data.x, data.edge_index, edge)
            loss += criterion(pred, torch.tensor([label], dtype=torch.float))

        loss.backward()
        optimizer.step()

    # Prediction
    model.eval()
    with torch.no_grad():
        prediction = model(data.x, data.edge_index, (follower, followed))
    return prediction.item()




G = create_directed_graph()
print(G.edges)
print(link_prediction(graph=G, follower=123, followed=456))