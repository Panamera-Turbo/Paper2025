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


import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def transE_model(graph, node1, node2):
    # Get the embeddings for the given nodes
    node1_embedding = graph.nodes()[node1]['embedding']
    node2_embedding = graph.nodes()[node2]['embedding']

    # Calculate the dot product of the two embeddings
    dot_product = np.dot(node1_embedding, node2_embedding)

    # Calculate the cosine similarity
    cosine_sim = cosine_similarity([node1_embedding], [node2_embedding])[0][0]

    # Calculate the L1 distance
    l1_distance = np.linalg.norm(node1_embedding - node2_embedding)

    # Calculate the features
    features = np.array([dot_product, cosine_sim, l1_distance]).reshape(1, -1)

    # Train a logistic regression model on the features
    model = make_pipeline(StandardScaler(), LogisticRegression())
    model.fit(features, [1])

    # Make a prediction on the test data (i.e., the same input features)
    prediction = model.predict(features)[0]

    return prediction

G = create_directed_graph()
print(G.edges)
# Invocation
print(transE_model(G, 123, 456))
    