import json
from openai import OpenAI
from tqdm import tqdm
import os
import json
import networkx as nx
import matplotlib.pyplot as plt
import re
# GPT_MODEL = "gpt-3.5-turbo-1106"
GPT_MODEL = 'gpt-4o-2024-08-06'
#GPT_MODEL = "gpt-3.5-turbo-0125"
# GPT_MODEL = 'gpt-4-turbo'
# client = OpenAI(
#    api_key="xxx",
#    base_url="https://ai-yyds.com/v1"
# )


client = OpenAI(
   api_key="sk-g13X2a4pa0qq7P1i48B678443b4a4983B6213eD0E54304Ab",
   base_url="https://api.holdai.top/v1"
)



def chat_completion_request(messages, model=GPT_MODEL):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=16384,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e


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
    #print("G: " + ", ".join(descriptions) + ".")
    
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

prompt_file = '/home/data2t2/wrz/visiual_ESTUnion/web/GPT_Graph_Agent/prompt.txt'
with open(prompt_file, 'r', encoding='utf-8') as file:
    prompt = file.read()

# 主程序
output_file_path = '/home/data2t2/wrz/visiual_ESTUnion/web/subgraphs_output_edgecut_1000.json'  # 替换为你保存的JSON文件路径
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
    total_ed = 0
    ed_list = []
    # 针对每组边和描述进行绘制
    for edges, descriptions in zip(subgraph_info['Edges'], subgraph_info['Description']):
        description  = "G: " + ", ".join(descriptions) + "."
        messages = []
        messages.append({"role": "system", "content": prompt})
        messages.append({"role": "user", "content": description})
        chat_response = chat_completion_request(messages)
        # print(chat_response)
    
        if isinstance(chat_response, Exception):
        # Skip the current iteration if an error occurred
            continue
    
        # Extracting the actual response content using dot notation
        first = chat_response.choices[0].message.content
        print(first)
        G = parse_graph(first)
        H = draw_graph(edges, descriptions)
        print(f'Graph G is as: {G.edges}')
        print(f'Graph H is as: {H.edges}')
        ed = edge_edit_distance(G, H)
        print(ed)
        total_ed += ed
        
    ed_list.append(total_ed)
    print(f'total_ed is {total_ed}')

print(ed_list)
#print(1500)