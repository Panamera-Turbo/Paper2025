import networkx as nx
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
file_path = '/home/data2t2/wrz/visiual_ESTUnion/web/web-Google_node.txt'  # 替换为你的文件路径
G = read_graph_from_file(file_path)

# 随机采样生成十个子图
num_subgraphs = 10
num_edges_to_sample = 900
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

    # 初始化描述和边信息
    output_descriptions = []
    edge_infos = []
    current_description = []
    current_edges = []

    for u, v in H.edges():
        current_description.append(f"Node {u} is connected to Node {v}")
        current_edges.append((u, v))

        # 检查描述长度是否达到50
        if len(current_description) >= 300:
            # 将当前的描述和边信息保存到edge_infos和output_descriptions
            edge_infos.append(current_edges)
            output_descriptions.append(current_description)

            # 重置当前描述和边信息
            current_description = []
            current_edges = []

    # 如果在最后还有未保存的描述和边信息，保存它们
    if current_description:
        edge_infos.append(current_edges)
        output_descriptions.append(current_description)

    # 将子图信息存储到字典中
    subgraph_info = {
        "Node_List": node_list,
        "Edges": edge_infos,
        "Description": output_descriptions
    }
    subgraphs_info.append(subgraph_info)

# 将结果保存到json文件
output_file_path = '/home/data2t2/wrz/visiual_ESTUnion/web/subgraphs_output_edgecut_300.json'  # 替换为你想要的输出文件路径
with open(output_file_path, 'w') as output_file:
    json.dump(subgraphs_info, output_file)

print(f"Saved {len(subgraphs_info)} subgraphs information to {output_file_path}.")
