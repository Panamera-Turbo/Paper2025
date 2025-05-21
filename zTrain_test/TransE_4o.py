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


def train_transe(graph, embedding_dim=50, epochs=100, learning_rate=0.01, margin=1.0):
    # Initialize embeddings for nodes
    nodes = list(graph.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    embeddings = np.random.uniform(-1, 1, (len(nodes), embedding_dim))
    
    # Generate positive and negative samples
    positive_edges = list(graph.edges())
    negative_edges = []
    for _ in range(len(positive_edges)):
        while True:
            u, v = np.random.choice(nodes, 2, replace=False)
            if (u, v) not in positive_edges and (v, u) not in positive_edges:
                negative_edges.append((u, v))
                break
    
    # Train TransE model
    for epoch in tqdm(range(epochs)):
        np.random.shuffle(positive_edges)
        for (u, v), (u_neg, v_neg) in zip(positive_edges, negative_edges):
            u_idx, v_idx = node_to_idx[u], node_to_idx[v]
            u_neg_idx, v_neg_idx = node_to_idx[u_neg], node_to_idx[v_neg]
            
            # Compute distances
            pos_distance = np.linalg.norm(embeddings[u_idx] + embeddings[v_idx])
            neg_distance = np.linalg.norm(embeddings[u_neg_idx] + embeddings[v_neg_idx])
            
            # Hinge loss
            loss = max(0, margin + pos_distance - neg_distance)
            
            # Gradient update
            if loss > 0:
                grad_pos = 2 * (embeddings[u_idx] + embeddings[v_idx])
                grad_neg = 2 * (embeddings[u_neg_idx] + embeddings[v_neg_idx])
                
                embeddings[u_idx] -= learning_rate * grad_pos
                embeddings[v_idx] -= learning_rate * grad_pos
                embeddings[u_neg_idx] += learning_rate * grad_neg
                embeddings[v_neg_idx] += learning_rate * grad_neg
    
    return embeddings, node_to_idx

def predict_transe(graph, embeddings, node_to_idx, follower, followed, threshold=0.5):
    if follower not in node_to_idx or followed not in node_to_idx:
        return False  # If nodes are not in the graph, return False
    
    u_idx, v_idx = node_to_idx[follower], node_to_idx[followed]
    distance = np.linalg.norm(embeddings[u_idx] + embeddings[v_idx])
    return distance < threshold

def link_prediction_task(graph, follower, followed):
    embeddings, node_to_idx = train_transe(graph)
    return predict_transe(graph, embeddings, node_to_idx, follower, followed)


G = create_directed_graph()
print(G.edges)

print(link_prediction_task(graph=G, follower=123, followed=456))
