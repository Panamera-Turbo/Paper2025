import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
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


def train_transE(graph, num_epochs=1000, learning_rate=0.01, embedding_dim=50):
    nodes = list(graph.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    idx_to_node = {idx: node for idx, node in enumerate(nodes)}
    
    # Initialize embeddings
    entity_embeddings = np.random.uniform(-6 / np.sqrt(embedding_dim), 6 / np.sqrt(embedding_dim), (len(nodes), embedding_dim))
    
    for epoch in range(num_epochs):
        for u, v in graph.edges():
            u_idx = node_to_idx[u]
            v_idx = node_to_idx[v]
            negative_sample = np.random.choice(nodes)
            while negative_sample in graph.neighbors(u):
                negative_sample = np.random.choice(nodes)
            neg_idx = node_to_idx[negative_sample]
            
            pos_score = np.linalg.norm(entity_embeddings[u_idx] + entity_embeddings[v_idx])
            neg_score = np.linalg.norm(entity_embeddings[u_idx] + entity_embeddings[neg_idx])
            
            # Update embeddings
            if pos_score < neg_score:
                entity_embeddings[u_idx] += learning_rate * (entity_embeddings[v_idx] - entity_embeddings[u_idx])
                entity_embeddings[neg_idx] -= learning_rate * (entity_embeddings[neg_idx] - entity_embeddings[u_idx])
                
    return entity_embeddings, node_to_idx

def predict_following(graph, follower, followed, entity_embeddings, node_to_idx):
    follower_idx = node_to_idx[follower]
    followed_idx = node_to_idx[followed]
    
    score = np.linalg.norm(entity_embeddings[follower_idx] + entity_embeddings[followed_idx])
    return score

def func(graph, follower, followed):
    entity_embeddings, node_to_idx = train_transE(graph)
    score = predict_following(graph, follower, followed, entity_embeddings, node_to_idx)
    return score


G = create_directed_graph()
print(G.edges)

print(func(graph=G, follower=123, followed=456))
