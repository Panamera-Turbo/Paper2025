import json
import networkx as nx
import matplotlib.pyplot as plt

# 读取JSON文件
def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# 绘制图形
def draw_graph(edges, descriptions):
    # 创建一个新的图
    G = nx.Graph()
    
    # 添加边
    G.add_edges_from(edges)
    
    # 绘制图形
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)  # 使用 spring 布局
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, font_color='black', edge_color='gray')
    
    output_description = []
    # 打印描述信息
    # for description in descriptions:
    #     print(description)
    #     output_description.append(description)

    print("G: " + ", ".join(descriptions) + ".")
    
    plt.title("Graph Visualization")
    plt.show()

# 主程序
output_file_path = 'web/subgraphs_output_edgecut_50.json'  # 替换为你保存的JSON文件路径
subgraphs_info = read_json_file(output_file_path)

# 循环读取每个子图并绘制
for index, subgraph_info in enumerate(subgraphs_info):
    print(f"Drawing Subgraph {index + 1}")
    
    # 针对每组边和描述进行绘制
    for edges, descriptions in zip(subgraph_info['Edges'], subgraph_info['Description']):
        draw_graph(edges, descriptions)
    
    break

