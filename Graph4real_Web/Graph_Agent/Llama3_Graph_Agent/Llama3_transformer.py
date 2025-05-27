import os
import json
import networkx as nx
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
import transformers
import torch

# 设置 CUDA 设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
model_id = "/home/data2t1/tempuser/llama3_GLandEX"

# 初始化生成文本的 pipeline
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

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
    print("G: " + ", ".join(descriptions) + ".")
    
    plt.title("Graph Visualization")
    # plt.show()
    return G

def parse_graph(input_data):
    # 使用正则表达式提取边信息
    pattern = r"\(\s*'(\d+)'\s*,\s*'(\d+)'\s*\)"
    edges = re.findall(pattern, input_data)

    # 创建一个网络图
    G = nx.Graph()

    # 添加边到图中
    G.add_edges_from([(int(node1), int(node2)) for node1, node2 in edges])

    # 计算边的总数和节点总数
    total_edges = G.number_of_edges()
    total_nodes = G.number_of_nodes()

    # 输出结果
    return G

prompt_file = '/home/data2t1/tempuser/GTAgent/Web/GPT_Graph_Agent/prompt.txt'
with open(prompt_file, 'r', encoding='utf-8') as file:
    prompt = file.read()

# 主程序
output_file_path = '/home/data2t1/tempuser/GTAgent/Web/Data_edgecut/subgraphs_output_edgecut_100.json'  # 替换为你保存的JSON文件路径
subgraphs_info = read_json_file(output_file_path)


def edge_edit_distance(G, H):
    # 获取图 G 和 H 的边
    edges_G = set(G.edges())
    edges_H = set(H.edges())

    # 计算边的添加和删除
    edges_to_add = edges_H - edges_G  # H 中的边，G 中没有的边
    edges_to_remove = edges_G - edges_H  # G 中的边，H 中没有的边

    # 计算编辑距离
    add_cost = len(edges_to_add)  # 添加的边数
    remove_cost = len(edges_to_remove)  # 删除的边数

    # 编辑距离 = 添加的边数 + 删除的边数
    edit_distance = add_cost + remove_cost

    return edit_distance


# 循环读取每个子图并绘制
for index, subgraph_info in enumerate(subgraphs_info):
    print(f"Drawing Subgraph {index + 1}")
    
    # 针对每组边和描述进行绘制
    for edges, descriptions in zip(subgraph_info['Edges'], subgraph_info['Description']):
        description  = "G: " + ", ".join(descriptions) + "."
        messages = []
        messages.append({"role": "system", "content": prompt})
        messages.append({"role": "user", "content": description})
        terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = pipeline(
            messages,
            max_new_tokens=4096,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.7,
            top_p=1,
        )
        first = outputs[0]["generated_text"][-1]['content']
    
        print(first)
        G = parse_graph(first)
        H = draw_graph(edges, descriptions)
        print(f'Graph G is as: {G.edges}')
        print(f'Graph H is as: {H.edges}')
        print(edge_edit_distance(G, H))
        break
        
    
    break