import networkx as nx
import random
import json
from tqdm import tqdm

# 读取txt文件并创建无向图
def read_graph_from_file(file_path):
    G = nx.Graph()  # 无向图
    
    with open(file_path, 'r') as file:
        for line in file:
            nodes = line.strip().split()
            if len(nodes) == 3:  # 确保每行有三个元素
                node1 = int(nodes[0])
                node2 = int(nodes[1])
                weight = random.randint(1, 50)
                G.add_edge(node1, node2, weight=weight)  # 添加带权无向边
    
    return G

# 1. 检查无向图中是否有环（使用无向图检测）
def has_cycle(graph):
    return not nx.is_forest(graph)  # 无环图是森林

# 2. 获取图中边的个数（不变）
def number_of_edges(graph):
    return graph.number_of_edges()

# 3. 检查图中是否存在特定的边（无向图不需要考虑方向）
def edge_exists(graph, node1, node2):
    return graph.has_edge(node1, node2)

# 4. 获取特定节点的度（无向图只有一种度）
def node_degree(graph, node):
    return graph.degree(node)

# 5. 计算无向图的三角形数量（直接计算）
def count_triangles(graph):
    triangle_counts = nx.triangles(graph)
    total_triangles = sum(triangle_counts.values()) // 3
    return total_triangles

# 6. 计算两个节点之间的最短路径（无向图）
def shortest_path(graph, source, target):
    try:
        # 获取最短路径的节点列表
        path = nx.shortest_path(graph, source=source, target=target)
        
        # 计算路径的权值之和
        weight_sum = sum(graph[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
        
        return weight_sum
    except nx.NetworkXNoPath:
        # 如果不存在路径，返回 0
        return 0

# 7. 检查某两个节点之间是否存在可达路径（无向图连通即可）
def is_reachable(graph, source, target):
    return nx.has_path(graph, source, target)

# 8. 检查某个特定节点是否存在于图中（不变）
def node_exists(graph, node):
    return node in graph.nodes()

# 9. 获取图中共有多少个节点（不变）
def number_of_nodes(graph):
    return graph.number_of_nodes()

# 10. 计算两个节点之间的最大流（无向图）
def max_flow(graph, source, target):
    try:
        # 使用networkx的最大流算法
        flow_value = nx.maximum_flow_value(graph, source, target, capacity='weight')
        return flow_value
    except:
        return 0  # 如果无法计算（如不连通），返回0

# 11. 检测图中是否存在欧拉回路（无向图版本）
def has_eulerian_circuit(graph):
    # 无向图的欧拉回路条件：
    # 1. 图是连通的
    # 2. 每个节点的度都是偶数
    if not nx.is_connected(graph):
        return False
    
    for node in graph.nodes():
        if graph.degree(node) % 2 != 0:
            return False
    
    return True

# 随机游走采样并返回子图的函数（基于节点数量）
def random_walk_sampling(G, num_nodes_to_sample):
    while True:  # 使用无限循环，直到成功生成子图
        sampled_nodes = set()  # 使用集合来避免重复节点
        visited_nodes = set()  # 记录已访问的节点
        start_node = random.choice(list(G.nodes()))  # 随机选择一个起始节点
        current_node = start_node
        sampled_nodes.add(current_node)  # 将起始节点加入采样集合
        visited_nodes.add(current_node)  # 将起始节点标记为已访问

        # 创建无向子图
        H = nx.Graph()

        # 初始化循环计数器
        loop_counter = 0

        while len(sampled_nodes) < num_nodes_to_sample:
            loop_counter += 1  # 增加循环计数器
            
            # 如果循环次数超过 10000，则重新开始采样
            if loop_counter > 10000:
                print("超过最大循环次数，重新开始采样...")
                break  # 退出当前循环，重新开始采样

            # 获取当前节点的邻居
            neighbors = list(G.neighbors(current_node))  # 无向图的邻居
            
            if not neighbors:  # 如果没有邻居，随机选择新的起始节点
                # 从未访问的节点中选择一个新的起始节点
                unvisited_nodes = list(set(G.nodes()) - visited_nodes)
                if not unvisited_nodes:  # 如果没有未访问的节点，退出循环
                    break
                current_node = random.choice(unvisited_nodes)
                sampled_nodes.add(current_node)  # 将新节点加入采样集合
                visited_nodes.add(current_node)  # 将新起始节点标记为已访问
                continue
            
            next_node = random.choice(neighbors)  # 随机选择一个邻居节点
            
            # 添加边到子图
            weight = G[current_node][next_node]['weight']  # 获取边的权值
            H.add_edge(current_node, next_node, weight=weight)  # 将边和权值添加到子图中
            
            # 更新节点集合
            sampled_nodes.add(next_node)  # 将新节点加入采样集合
            visited_nodes.add(next_node)  # 将新节点标记为已访问
            
            current_node = next_node  # 移动到下一个节点

        print(f"已采样节点数: {len(sampled_nodes)}")
        # 如果成功生成子图，返回结果
        if len(sampled_nodes) >= num_nodes_to_sample:
            return H  # 返回包含采样节点的子图

# 主程序
file_path = '/home/data2t1/tempuser/GTAgent/Transport/adj_matrix_node.txt'  # 替换为你的文件路径
G = read_graph_from_file(file_path)

# 随机采样生成十个子图
num_subgraphs = 1000
num_nodes_to_sample = 100  # 指定要采样的节点数量

# 存储每个函数的结果
datasets = {
    "Cycle_Detection": [],
    "Edge_Count": [],
    "Edge_Existence": [],
    "Degree_Count": [],
    "Triangle_Count": [],
    "Shortest_Path": [],
    "Path_Existence": [],
    "Node_Existence": [],
    "Node_Count": [],
    "Maxflow": [],  # 新增最大流任务
    "Euler_Path": []  # 新增欧拉回路任务
}

for i in tqdm(range(num_subgraphs), desc="Generating subgraphs"):
    # 使用随机游走采样边
    H = nx.Graph()
    H = random_walk_sampling(G, num_nodes_to_sample)

    # 获取节点列表
    node_list = list(H.nodes())

    # 初始化描述和边信息
    output_descriptions = []
    edge_infos = []
    current_description = []
    current_edges = []

    for u, v, weight in H.edges(data='weight'):
        current_description.append(f"Node {u} is connected to Node {v} with weight {weight}")
        current_edges.append((u, v, weight))

        # 检查描述长度是否达到300
        if len(current_description) >= 50:
            edge_infos.append(current_edges)
            output_descriptions.append([", ".join(current_description)])
            current_description = []
            current_edges = []

    if current_description:
        edge_infos.append(current_edges)
        output_descriptions.append([", ".join(current_description)])
    

    # 随机选择两个节点
    if len(H.nodes) >= 2:
        source_node, target_node = random.sample(list(H.nodes()), 2)
    else:
        raise ValueError("子图中节点数量不足，无法随机选择源节点和目标节点。")

    # 将子图信息存储到字典中
    subgraph_info = {
        "Node_List": node_list,
        "Edges": edge_infos,
        "Edge_Count": H.number_of_edges(),
        "Description": output_descriptions,
        "question": "",
        "answer": ""
    }

    # 生成每个数据集
    # 1. 检查图中是否有环
    subgraph_info["question"] = "Does the undirected graph have a cycle?"
    subgraph_info["answer"] = has_cycle(H)
    print(subgraph_info["question"], "->", subgraph_info["answer"])  # 打印问题和答案
    datasets["Cycle_Detection"].append(subgraph_info)

    # 2. 获取图中边的个数
    subgraph_info = subgraph_info.copy()  # 复制以避免覆盖
    subgraph_info["question"] = "What is the number of edges in the undirected graph?"
    subgraph_info["answer"] = number_of_edges(H)
    print(subgraph_info["question"], "->", subgraph_info["answer"])  # 打印问题和答案
    datasets["Edge_Count"].append(subgraph_info)

    # 3. 检查图中是否存在特定的边
    subgraph_info = subgraph_info.copy()  # 复制以避免覆盖
    subgraph_info["question"] = f"Does the edge between {source_node} and {target_node} exist?"
    subgraph_info["answer"] = edge_exists(H, source_node, target_node)
    print(subgraph_info["question"], "->", subgraph_info["answer"])  # 打印问题和答案
    datasets["Edge_Existence"].append(subgraph_info)

    # 4. 获取特定节点的度
    subgraph_info = subgraph_info.copy()  # 复制以避免覆盖
    subgraph_info["question"] = f"What is the degree of node {source_node}?"
    subgraph_info["answer"] = node_degree(H, source_node)
    print(subgraph_info["question"], "->", subgraph_info["answer"])  # 打印问题和答案
    datasets["Degree_Count"].append(subgraph_info)

    # 5. 计算每个节点的三角形数量
    subgraph_info = subgraph_info.copy()  # 复制以避免覆盖
    subgraph_info["question"] = "How many triangles are in the undirected graph?"
    subgraph_info["answer"] = count_triangles(H)
    print(subgraph_info["question"], "->", subgraph_info["answer"])  # 打印问题和答案
    datasets["Triangle_Count"].append(subgraph_info)

    # 6. 计算两个节点之间的最短路径
    subgraph_info = subgraph_info.copy()  # 复制以避免覆盖
    subgraph_info["question"] = f"What is the shortest path from {source_node} to {target_node} in the undirected graph?"
    subgraph_info["answer"] = shortest_path(H, source_node, target_node)
    print(subgraph_info["question"], "->", subgraph_info["answer"])  # 打印问题和答案
    datasets["Shortest_Path"].append(subgraph_info)

    # 7. 检查某两个节点之间是否存在可达路径
    subgraph_info = subgraph_info.copy()  # 复制以避免覆盖
    subgraph_info["question"] = f"Is there a path between {source_node} and {target_node}?"
    subgraph_info["answer"] = is_reachable(H, source_node, target_node)
    print(subgraph_info["question"], "->", subgraph_info["answer"])  # 打印问题和答案
    datasets["Path_Existence"].append(subgraph_info)

    # 8. 检查某个特定节点是否存在于图中
    subgraph_info = subgraph_info.copy()  # 复制以避免覆盖
    subgraph_info["question"] = f"Does node {source_node} exist in the undirected graph?"
    subgraph_info["answer"] = node_exists(H, source_node)
    print(subgraph_info["question"], "->", subgraph_info["answer"])  # 打印问题和答案
    datasets["Node_Existence"].append(subgraph_info)

    # 9. 获取图中共有多少个节点
    subgraph_info = subgraph_info.copy()  # 复制以避免覆盖
    subgraph_info["question"] = "What is the number of nodes in the undirected graph?"
    subgraph_info["answer"] = number_of_nodes(H)
    print(subgraph_info["question"], "->", subgraph_info["answer"])  # 打印问题和答案
    datasets["Node_Count"].append(subgraph_info)

    # 10. 计算两个节点之间的最大流（无向图）
    subgraph_info = subgraph_info.copy()  # 复制以避免覆盖
    subgraph_info["question"] = f"What is the maximum flow between {source_node} and {target_node} in the undirected graph?"
    subgraph_info["answer"] = max_flow(H, source_node, target_node)
    print(subgraph_info["question"], "->", subgraph_info["answer"])  # 打印问题和答案
    datasets["Maxflow"].append(subgraph_info)

    # 11. 检测图中是否存在欧拉回路（无向图版本）
    subgraph_info = subgraph_info.copy()  # 复制以避免覆盖
    subgraph_info["question"] = "Does the undirected graph have an Eulerian circuit?"
    subgraph_info["answer"] = has_eulerian_circuit(H)
    print(subgraph_info["question"], "->", subgraph_info["answer"])  # 打印问题和答案
    datasets["Euler_Path"].append(subgraph_info)

# 将每个数据集保存到不同的 JSON 文件中
for task_name, data in datasets.items():
    output_file_path = f'/home/data2t1/tempuser/GTAgent/Transport/gen/100_50_nodes/{task_name}.json'  # 修改文件名以区分无向图
    with open(output_file_path, 'w') as output_file:
        json.dump(data, output_file)

print("Undirected graph data sets saved to JSON files.")
