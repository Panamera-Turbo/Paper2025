import torch
import networkx as nx
from torch_geometric.data import Data

# Load data
data = torch.load('Citation/cora/cora_fixed_tfidf.pt')

# Ensure data is in PyG Data object format
if not isinstance(data, Data):
    data = Data(x=data.x, edge_index=data.edge_index, y=data.y)

# Convert edge_index to NetworkX directed graph, ensuring `src` cites `dst` (dst < src)
def edge_index_to_networkx(edge_index):
    G = nx.DiGraph()
    edge_index = edge_index.cpu().numpy()  # Convert to numpy if it's a tensor
    edges = set()  # Use a set to avoid duplicates
    
    for src, dst in zip(edge_index[0], edge_index[1]):
        if dst < src:  # Ensure `dst` is older (smaller ID) than `src`
            edges.add((src, dst))
    
    G.add_edges_from(edges)
    return G

# Create the NetworkX graph
G = edge_index_to_networkx(data.edge_index)

