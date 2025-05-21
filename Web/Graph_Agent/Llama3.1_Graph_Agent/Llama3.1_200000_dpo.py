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

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
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
def draw_graph(edges, descriptions):
    # 创建一个新的图
    G = nx.Graph()
    
    # 添加边
    G.add_edges_from(edges)
    
    # 绘制图形
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)  # 使用 spring 布局
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, font_color='black', edge_color='gray')
    # print("G: " + ", ".join(descriptions) + ".")
    
    plt.title("Graph Visualization")
    # plt.show()
    return G

def parse_graph(input_data):
    # 使用正则表达式提取边信息，支持两种形式
    pattern = r"\(\s*('(\d+)'|(\d+))\s*,\s*('(\d+)'|(\d+))\s*\)"
    edges = re.findall(pattern, input_data)

    # 创建一个网络图
    G = nx.Graph()

    # 添加边到图中，转换为整数
    for edge in edges:
        # edge 是一个元组，包含多个捕获组
        node1 = edge[0]  # 第一个捕获组
        node2 = edge[3]  # 第二个捕获组
        # 选择非空的捕获组并转换为整数
        G.add_edge(int(node1.strip("'")) if node1 else int(edge[1]), 
                   int(node2.strip("'")) if node2 else int(edge[4]))

    # 输出结果
    return G


prompt_file = '/home/data2t1/tempuser/GTAgent/Web/Graph_Agent/GPT_Graph_Agent/prompt5.txt'
with open(prompt_file, 'r', encoding='utf-8') as file:
    prompt = file.read()

# 主程序
output_file_path = '/home/data2t1/tempuser/GTAgent/Web/Dataset/10000_100/has_cycle_output.json' # 替换为你保存的JSON文件路径
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


ed_list = []
# 循环读取每个子图并绘制
for index, subgraph_info in tqdm(enumerate(subgraphs_info), total=len(subgraphs_info), desc="Drawing Subgraphs"):
    print(f"Drawing Subgraph {index + 1}")
    total_ed = 0
    cnt = 0
    total_retry = 0
    # 针对每组边和描述进行绘制
    for edges, descriptions in zip(subgraph_info['Edges'], subgraph_info['Description']):
        description = "G: " + ", ".join(descriptions) + "."
        messages = []
        messages.append({"role": "system", "content": prompt})
        messages.append({"role": "user", "content": description})    
        # 初始化生成的图 G
        G = None       
        # 使用 while 循环确保生成的 G 的边数大于等于 200
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
            
            # 检查生成的图 G 的边数是否满足条件
            if len(G.edges) == 200:
                cnt += 1
                break
            else:
                total_retry += 1
                print(f"Generated graph G has {len(G.edges)} edges, which is less than 100. Regenerating...")
        
        # 绘制 H
        H = draw_graph(edges, descriptions)
        print(f'Graph G is as: {G.edges}')
        print(f'Graph H is as: {H.edges}')
        
        # 计算编辑距离
        ed = edge_edit_distance(G, H)
        print(ed)
        total_ed += ed
        print(f'total cnt {cnt}, total_ed {total_ed}, total_retry {total_retry}')
    
    ed_list.append(total_ed)
    print(f'total_ed is {total_ed}')
    break

print(ed_list)
