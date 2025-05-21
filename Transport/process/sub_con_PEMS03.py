# import pandas as pd
# import networkx as nx
# import matplotlib.pyplot as plt
# import random

# # 读取CSV文件
# df = pd.read_csv('./PEMS03.csv')

# # 创建一个无向图
# G = nx.Graph()

# # 添加边到图中，确保节点为整数
# for index, row in df.iterrows():
#     G.add_edge(int(row['from']), int(row['to']), weight=row['distance'])

# # 随机采样300条边
# num_edges_to_sample = 300
# edges = list(G.edges(data=True))

# # 如果图中的边少于300条，调整采样数量
# num_edges_to_sample = min(num_edges_to_sample, len(edges))

# # 随机选择边
# sampled_edges = random.sample(edges, num_edges_to_sample)

# # 创建子图
# H = nx.Graph()
# H.add_edges_from(sampled_edges)

# # 绘制子图
# pos = nx.spring_layout(H)  # 使用spring布局
# nx.draw(H, pos, with_labels=True, node_size=500, node_color='lightblue', font_size=10, font_color='black', font_weight='bold')

# # 显示图
# plt.title('Subgraph Visualization with Randomly Sampled Edges')
# plt.show()

# # 输出子图的节点列表
# node_list = list(H.nodes())
# print(f"Node list: {node_list}")

# # 输出图的描述
# output_description = []
# for u, v, data in H.edges(data=True):
#     # 格式化权值，确保没有多余的点
#     weight = f"{data['weight']:.2f}"  # 保留两位小数
#     output_description.append(f"Node {u} is connected to Node {v} with weight {weight}")

# # 以自然语言形式输出
# print("G: " + ", ".join(output_description) + ".")

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import random

# 读取CSV文件
df = pd.read_csv('Transport\PEMS03.csv')

# 创建一个无向图
G = nx.Graph()

# 添加边到图中，确保节点为整数
for index, row in df.iterrows():
    G.add_edge(int(row['from']), int(row['to']), weight=row['distance'])

# 随机采样300条边
num_edges_to_sample = 300
edges = list(G.edges(data=True))

# 如果图中的边少于300条，调整采样数量
num_edges_to_sample = min(num_edges_to_sample, len(edges))

# 随机选择边
sampled_edges = random.sample(edges, num_edges_to_sample)

# 创建子图
H = nx.Graph()
H.add_edges_from(sampled_edges)

# 绘制子图
pos = nx.spring_layout(H)  # 使用spring布局
nx.draw(H, pos, with_labels=True, node_size=500, node_color='lightblue', font_size=10, font_color='black', font_weight='bold')

# 显示图
plt.title('Subgraph Visualization with Randomly Sampled Edges')
plt.show()

# 输出子图的节点列表
node_list = list(H.nodes())
print(f"Node list: {node_list}")

# 输出图的描述
output_description = []
for u, v, data in H.edges(data=True):
    weight = f"{data['weight']}"  # 保留两位小数
    output_description.append(f"Node {u} is connected to Node {v} with weight {weight}")

# 以自然语言形式输出
print("G: " + ", ".join(output_description) + ".")

# 1. 检查图中是否有环
def has_cycle(graph):
    return not nx.is_tree(graph)

# 2. 获取图中边的个数
def number_of_edges(graph):
    return graph.number_of_edges()

# 3. 检查图中是否存在特定的边
def edge_exists(graph, edge):
    return graph.has_edge(*edge)

# 4. 获取特定节点的度
def node_degree(graph, node):
    return graph.degree(node)

# 5. 计算无向图的最大流
def maximum_flow(graph, source, target):
    # 将无向图转换为有向图
    directed_graph = nx.DiGraph()
    for u, v, data in graph.edges(data=True):
        weight = data['weight']
        directed_graph.add_edge(u, v, capacity=weight)
        directed_graph.add_edge(v, u, capacity=weight)  # 添加反向边
    
    flow_value, flow_dict = nx.maximum_flow(directed_graph, _s=source, _t=target)
    return flow_value

# 6. 计算两个节点之间的最短路径
def shortest_path(graph, source, target):
    return nx.shortest_path(graph, source=source, target=target, weight='weight')

# 7. 检查某两个节点之间是否存在可达路径
def is_reachable(graph, source, target):
    return nx.has_path(graph, source, target)

# 8. 检查某个特定节点是否存在于图中
def node_exists(graph, node):
    return node in graph.nodes()

# 9. 获取图中共有多少个节点
def number_of_nodes(graph):
    return graph.number_of_nodes()

# 10. 计算某两个节点之间存在多少条可达路径
def count_reachable_paths(graph, source, target):
    return len(list(nx.all_simple_paths(graph, source, target)))

# 找到两个之间存在路径的节点
reachable_pair_found = False
while not reachable_pair_found:
    if len(H.nodes) >= 2:  # 确保有足够的节点进行随机选择
        source_node, target_node = random.sample(list(H.nodes()), 2)
        if is_reachable(H, source_node, target_node):
            reachable_pair_found = True
            specific_node = source_node
            specific_edge = (source_node, target_node)
    else:
        raise ValueError("子图中节点数量不足，无法随机选择源节点和目标节点。")

# 调用封装的函数
print(f"图中是否有环: {'是' if has_cycle(H) else '否'}")
print(f"图中边的个数: {number_of_edges(H)}")
print(f"图中是否存在边 {specific_edge}: {'是' if edge_exists(H, specific_edge) else '否'}")
print(f"图中节点 {specific_node} 的度: {node_degree(H, specific_node)}")
print(f"图中节点 {source_node} 到节点 {target_node} 之间的最大流: {maximum_flow(H, source_node, target_node)}")
print(f"图中节点 {source_node} 到节点 {target_node} 之间的最短路径: {shortest_path(H, source_node, target_node)}")
print(f"节点 {source_node} 到节点 {target_node} 之间是否存在可达路径: {'是' if is_reachable(H, source_node, target_node) else '否'}")
print(f"节点 {specific_node} 是否存在于图中: {'是' if node_exists(H, specific_node) else '否'}")
print(f"图中共有多少个节点: {number_of_nodes(H)}")
print(f"节点 {source_node} 到节点 {target_node} 之间存在多少条可达路径: {count_reachable_paths(H, source_node, target_node)}")
