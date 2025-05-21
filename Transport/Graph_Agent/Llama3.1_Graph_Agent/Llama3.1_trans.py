import os
import json
import networkx as nx
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
from vllm import LLM
from vllm.sampling_params import SamplingParams
import transformers
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
model_id = "/home/data2t1/tempuser/Llama-3.1-8B-Instruct"

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
def draw_graph(edges, descriptions=None):
    # 创建一个新的图
    G = nx.Graph()
    
    
    # 添加边到图中，同时将权重作为属性
    for edge in edges:
        if len(edge) == 3:  # 确保是三元组 (node1, node2, weight)
            node1, node2, weight = edge
            G.add_edge(node1, node2, weight=weight)
        else:
            raise ValueError("Each edge must be a three-element tuple (node1, node2, weight)")
    
    return G

def parse_graph(input_data):
    import re
    import networkx as nx

    # 使用正则表达式提取边信息，包括起点、终点和权重
    pattern = r"\(\s*('(\d+)'|(\d+))\s*,\s*('(\d+)'|(\d+))\s*,\s*('[\d.]+'|[\d.]+)\s*\)"
    edges = re.findall(pattern, input_data)

    # 创建一个带权图
    G = nx.Graph()

    # 添加边到图中，解析出起点、终点和权重
    for edge in edges:
        # edge 是一个元组，包含多个捕获组
        node1 = edge[0]  # 第一个捕获组（可能包含引号）
        node2 = edge[3]  # 第二个捕获组（可能包含引号）
        weight = edge[6]  # 第三部分，捕获权重（可能带引号）

        # 选择非空的捕获组并转换为整数，权重转换为浮点数
        node1 = int(node1.strip("'")) if node1 else int(edge[1])
        node2 = int(node2.strip("'")) if node2 else int(edge[4])
        weight = float(weight.strip("'")) if weight.startswith("'") else float(weight)

        # 添加带权边到图中
        G.add_edge(node1, node2, weight=weight)

    # 返回解析后的图
    return G

prompt_file = '/home/data2t1/tempuser/GTAgent/Web/Graph_Agent/GPT_Graph_Agent/prompt5.txt'
with open(prompt_file, 'r', encoding='utf-8') as file:
    prompt = file.read()

# 主程序
output_file_path = '/home/data2t1/tempuser/GTAgent/.Graph4Real/Trans/Json/Small/Degree_Count.json' # 替换为你保存的JSON文件路径
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


import json
from collections import defaultdict

# Initialize full graphs and other data structures
Full_G = nx.Graph()
Full_H = nx.Graph()
ed_list = []
subgraph_results = []

for index, subgraph_info in tqdm(enumerate(subgraphs_info), total=len(subgraphs_info), desc="Processing Subgraphs"):
    print(f"Processing Subgraph {index + 1}")
    total_ed = 0
    cnt = 0
    total_retry = 0
    
    # Store individual subgraph results
    subgraph_result = {
        "index": index,
        "edges": [],
        "description": subgraph_info['Description'],
        "ed_per_subgraph": []
    }
    
    for edges, descriptions in zip(subgraph_info['Edges'], subgraph_info['Description']):
        description = "G: " + ", ".join(descriptions) + "."
        messages = []
        messages.append({"role": "system", "content": prompt})
        messages.append({"role": "user", "content": description})    
        
        # Initialize generated graph G
        G = None    
        H = draw_graph(edges, descriptions)   
        
        # Add edges to Full_H
        for u, v, data in H.edges(data=True):
            Full_H.add_edge(u, v, **data)
        
        # Use while loop to ensure generated G has correct number of edges
        while True:
            terminators = [
                pipeline.tokenizer.eos_token_id,
                pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            outputs = pipeline(
                messages,
                max_new_tokens=8192,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.7,
                top_p=1,
            )
            first = outputs[0]["generated_text"][-1]['content']
            print(first)
            G = parse_graph(first)
            
            if len(G.edges) == len(H.edges):
                cnt += 1
                break
            else:
                total_retry += 1
                print(f"Generated graph G has {len(G.edges)} edges, which is not equal to {len(H.edges)}. Regenerating...")
        
        # Add edges to Full_G
        for u, v, data in G.edges(data=True):
            Full_G.add_edge(u, v, **data)
        
        print(f'Graph G is as: {G.edges(data=True)}')
        print(f'Graph H is as: {H.edges(data=True)}')
        
        # Calculate edit distance
        ed = edge_edit_distance(G, H)
        print(ed)
        total_ed += ed
        subgraph_result["ed_per_subgraph"].append(ed)
        print(f'total cnt {cnt}, total_ed {total_ed}, total_retry {total_retry}')
    
    ed_list.append(total_ed)
    subgraph_result["total_ed"] = total_ed
    subgraph_results.append(subgraph_result)
    print(f'total_ed is {total_ed}')


# Prepare the final result dictionary
result = {
    "Full_G_edges": list(Full_G.edges(data=True)),
    "Full_H_edges": list(Full_H.edges(data=True)),
    "ed_list": ed_list,
    "total_ed": sum(ed_list),
    "subgraph_results": subgraph_results
}

# Save to JSON file
output_json_path = '/home/data2t1/tempuser/GTAgent/Transport/Graph_Agent/Llama3.1_Graph_Agent/Result/Degree_Count.json'  # Replace with your desired output path
with open(output_json_path, 'w') as f:
    json.dump(result, f, indent=4)

print("Results saved to:", output_json_path)
