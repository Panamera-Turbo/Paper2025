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

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
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
    G = nx.Graph()
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
output_file_path = '/home/data2t1/tempuser/GTAgent/.Graph4Real/Trans/Json/Middle/Trans/Shortest_Path.json' # 替换为你保存的JSON文件路径
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

# 定义保存结果的函数
def save_intermediate_results(results, file_path):
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=None)

# 存储每个子图的结果
subgraph_results = []
total_retry = 0
total_ed = 0
num = 0
output_json_path = '/home/data2t1/tempuser/GTAgent/Transport/Graph_Agent/Llama3.1_Graph_Agent/Result/test_stepbystep.json'

for index, subgraph_info in tqdm(enumerate(subgraphs_info), total=len(subgraphs_info), desc="Processing Subgraphs"):
    print(f"Processing Subgraph {index + 1}")
    cnt = 0
    retry = 0
    num += 1
    
    # 存储当前子图的结果
    current_subgraph_result = {
        "index": index,
        "description": subgraph_info['Description'],
        "pairs": []  # 存储每个 (G, H) 对的信息
    }
    
    for edges, descriptions in zip(subgraph_info['Edges'], subgraph_info['Description']):
        description = "G: " + ", ".join(descriptions) + "."
        messages = []
        messages.append({"role": "system", "content": prompt})
        messages.append({"role": "user", "content": description})    
        
        # 初始化生成图 G
        G = None    
        H = draw_graph(edges, descriptions)   
        
        # 使用 while 循环确保生成的 G 边数与 H 相同
        tolerance = 0  # 初始容忍度为0
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
            
            # 计算允许的误差范围
            allowed_min = len(H.edges) - tolerance
            allowed_max = len(H.edges) + tolerance
            
            if allowed_min <= len(G.edges) <= allowed_max:
                cnt += 1
                break
            else:
                retry += 1
                # 每5次重试增加1的容忍度
                if retry % 5 == 0:
                    tolerance += 1
                    print(f"Increasing tolerance to ±{tolerance} after {retry} retries")
                
                print(f"Generated graph G has {len(G.edges)} edges, which is not in range [{allowed_min}, {allowed_max}]. Regenerating...")
        
        print(f'Graph G is as: {G.edges(data=True)}')
        print(f'Graph H is as: {H.edges(data=True)}')
        
        # 计算编辑距离
        ed = edge_edit_distance(G, H)
        print(ed)
        total_ed += ed
        total_retry += retry
        
        # 存储当前 (G, H) 对的信息
        current_pair = {
            "G_edges": list(G.edges(data=True)),
            "H_edges": list(H.edges(data=True)),
            "edit_distance": ed,
            "retry": retry,
            "final_tolerance": tolerance  # 记录最终使用的容忍度
        }
        current_subgraph_result["pairs"].append(current_pair)
        
        # 每次添加新的 pair 后保存一次
        save_intermediate_results(subgraph_results + [current_subgraph_result], output_json_path)
        
        print(f'total cnt {cnt}, total_ed {total_ed}, total_retry {retry}')
    
    # 添加当前子图的总编辑距离
    current_subgraph_result["total_edit_distance"] = total_ed
    current_subgraph_result["total_retry_times"] = total_retry
    subgraph_results.append(current_subgraph_result)
    
    # 每次完成一个子图处理后保存一次
    save_intermediate_results(subgraph_results, output_json_path)
    
    print(f'total_ed is {total_ed}')

print("Final results saved to:", output_json_path)