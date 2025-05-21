# import numpy as np
# import networkx as nx
# import torch
# import torch.nn.functional as F
# from torch_geometric.data import Data
# from torch_geometric.nn import GCNConv
# from sklearn.model_selection import train_test_split

# import random

# def create_directed_graph():
#     # Initialize a directed graph
#     G = nx.DiGraph()
    
#     # Add 500 nodes
#     num_nodes = 500
#     G.add_nodes_from(range(num_nodes))
    
#     # Add 5000 random edges
#     num_edges = 5000
#     edges = set()
#     while len(edges) < num_edges:
#         u = random.randint(0, num_nodes - 1)
#         v = random.randint(0, num_nodes - 1)
#         if u != v and (u, v) not in edges and not (u == 123 and v == 456) and not (u == 456 and v == 123):  # Ensure no edge between 123 and 456
#             edges.add((u, v))
    
#     G.add_edges_from(edges)
#     return G


# class GCN(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels):
#         super(GCN, self).__init__()
#         self.conv1 = GCNConv(in_channels, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, 1)

#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = self.conv2(x, edge_index)
#         return torch.sigmoid(x)

# def func(graph, follower, followed):
#     # Prepare the graph data
#     edge_index = torch.tensor(list(graph.edges)).t().contiguous()
#     num_nodes = graph.number_of_nodes()
#     x = torch.eye(num_nodes)  # One-hot encoding of nodes
    
#     # Create data object
#     data = Data(x=x, edge_index=edge_index)

#     # Prepare training and test data
#     pos_samples = [(follower, followed)]
#     neg_samples = [(follower, i) for i in range(num_nodes) if i != follower and not graph.has_edge(follower, i)]
#     samples = pos_samples + neg_samples
#     labels = torch.tensor([1] + [0] * len(neg_samples)).float()

#     # Split the data
#     train_samples, test_samples, train_labels, _ = train_test_split(samples, labels, test_size=0.2)

#     # Create edge index for training
#     train_edge_index = torch.tensor(train_samples).t().contiguous()

#     # Initialize model and optimizer
#     model = GCN(in_channels=num_nodes, hidden_channels=4)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#     # Training loop
#     model.train()
#     for epoch in range(100):
#         optimizer.zero_grad()
#         out = model(data.x, data.edge_index)
#         loss = F.binary_cross_entropy(out[train_edge_index[0]], train_labels.view(-1, 1))
#         loss.backward()
#         optimizer.step()

#     # Prediction
#     model.eval()
#     with torch.no_grad():
#         prediction = model(data.x, data.edge_index)
#         return prediction[followed].item()



# G = create_directed_graph()
# print(G.edges)
# print(func(graph=G, follower=123, followed=456))