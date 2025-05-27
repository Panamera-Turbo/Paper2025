import networkx as nx
import matplotlib.pyplot as plt
import random

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
file_path = 'Web\web-Google_node.txt'  # 替换为你的文件路径
G = read_graph_from_file(file_path)

# 随机采样1000条边
num_edges_to_sample = 500
edges = list(G.edges())

# 如果图中的边少于1000条，调整采样数量
num_edges_to_sample = min(num_edges_to_sample, len(edges))

# 随机选择边
sampled_edges = random.sample(edges, num_edges_to_sample)

# 创建子图
H = nx.Graph()
H.add_edges_from(sampled_edges)

# 绘制子图
pos = nx.spring_layout(H)  # 使用spring布局
nx.draw(H, pos, with_labels=True, node_size=1000, node_color='lightblue', font_size=10, font_color='black', font_weight='bold')

# 显示图
plt.title('Subgraph Visualization with Randomly Sampled Edges')
plt.show()

# 输出子图的节点列表
node_list = list(H.nodes())
print(f"Node list: {node_list}")

# 输出图的描述
output_description = []
print(len(H))
print(len(H.edges))
for u, v in H.edges():
    output_description.append(f"Page {u} is connected to Page {v}")

# 以自然语言形式输出
print("G: " + ", ".join(output_description))

# cnt = 0
# for u, v in H.edges():
#     cnt += 1
#     print(f"{cnt} Page {u} is connected to Page {v}")


# 将结果保存到txt文件
output_file_path = 'Web/subgraph_output.txt'  # 替换为你想要的输出文件路径
with open(output_file_path, 'w') as output_file:
    # 写入节点列表
    output_file.write("Node list: " + ", ".join(map(str, node_list)) + "\n")
    
    # 写入图的描述
    output_description = []
    for u, v in H.edges():
        output_description.append(f"Node {u} is connected to Node {v}")
    
    # 写入连接关系
    output_file.write("G: " + ", ".join(output_description) + ".")