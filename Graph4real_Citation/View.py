import torch

def read_data():
    data= torch.load('Cora/cora/cora_fixed_tfidf.pt')
    x_feature=data.x
    edge_index=data.edge_index
    true_labels=data.y
    texts=data.node_stores[0]._mapping['raw_texts']
    return 

read_data