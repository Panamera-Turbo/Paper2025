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


import numpy as np
import networkx as nx
from sklearn.preprocessing import normalize

def train_transe(graph, epochs=100, learning_rate=0.01, embedding_dim=50):
    nodes = list(graph.nodes())
    node_to_index = {node: i for i, node in enumerate(nodes)}
    num_nodes = len(nodes)
    
    # Initialize embeddings
    entity_embeddings = np.random.uniform(-6 / np.sqrt(embedding_dim), 6 / np.sqrt(embedding_dim), (num_nodes, embedding_dim))
    
    def get_edges():
        for u, v in graph.edges():
            yield (node_to_index[u], node_to_index[v])
    
    for epoch in range(epochs):
        for head, tail in get_edges():
            # Calculate the score
            score = entity_embeddings[head] - entity_embeddings[tail]
            loss = np.sum(np.maximum(0, 1 + np.linalg.norm(score)))
            if loss > 0:
                # Update embeddings
                entity_embeddings[head] += learning_rate * score
                entity_embeddings[tail] -= learning_rate * score
    
    return entity_embeddings, node_to_index

def predict_following(graph, follower, followed, epochs=100, learning_rate=0.01, embedding_dim=50):
    entity_embeddings, node_to_index = train_transe(graph, epochs, learning_rate, embedding_dim)
    
    follower_index = node_to_index[follower]
    followed_index = node_to_index[followed]
    
    score = entity_embeddings[follower_index] - entity_embeddings[followed_index]
    prediction = np.linalg.norm(score)
    
    return prediction < 1  # Threshold for prediction




G = create_directed_graph()
print(G.edges)

print(predict_following(graph=G, follower=123, followed=456))

