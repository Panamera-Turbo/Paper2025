import networkx as nx
import matplotlib.pyplot as plt
import random
import json

# 读取txt文件并创建图
def read_graph_from_file(file_path):
    G = nx.Graph()
    
    with open(file_path, 'r') as file:
        for line in file:
            # 去掉行末的换行符并分割
            nodes = line.strip().split()
            if len(nodes) == 2:
                # 将节点转换为整数并添加边
                G.add_edge(int(nodes[0]), int(nodes[1]))
    
    return G

# 主程序
file_path = 'Web/web-Google_node.txt'  # 替换为你的文件路径
G = read_graph_from_file(file_path)

# 随机采样生成十个子图
num_subgraphs = 10
num_edges_to_sample = 500
edges = list(G.edges())

# 如果图中的边少于500条，调整采样数量
num_edges_to_sample = min(num_edges_to_sample, len(edges))

# 存储所有子图的信息
subgraphs_info = []

for i in range(num_subgraphs):
    # 随机选择边
    sampled_edges = random.sample(edges, num_edges_to_sample)

    # 创建子图
    H = nx.Graph()
    H.add_edges_from(sampled_edges)

    # 获取节点列表
    node_list = list(H.nodes())

    # 获取边的信息
    edge_info = [(u, v) for u, v in H.edges()]

    # 创建描述信息
    output_description = []
    for u, v in H.edges():
        output_description.append(f"Node {u} is connected to Node {v}")

    # 将子图信息存储到字典中
    subgraph_info = {
        "Node_List": node_list,
        "Edges": edge_info,
        "Description": output_description
    }

    subgraphs_info.append(subgraph_info)

# 将结果保存到json文件
output_file_path = 'Web/subgraphs_output.json'  # 替换为你想要的输出文件路径
with open(output_file_path, 'w') as output_file:
    json.dump(subgraphs_info, output_file)

print(f"Saved {num_subgraphs} subgraphs information to {output_file_path}.")
