import os
import json
import networkx as nx
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
from vllm import LLM
from vllm.sampling_params import SamplingParams
import copy

os.environ["CUDA_VISIBLE_DEVICES"] = "12,13"
model_name = "/home/data2t2/wrz/LLaMA/llama3_GLandEX"

sampling_params = SamplingParams(max_tokens=16384, temperature=0.7, top_p=1)

llm = LLM(model=model_name, tokenizer_mode="auto", tensor_parallel_size=2, dtype="half")


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
output_file_path = '/home/data2t2/wrz/visiual_ESTUnion/web/subgraphs_output_edgecut_200.json'  # 替换为你保存的JSON文件路径
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
    cnt = 0
    G1 = ''
    G2 = ''
    questions = []
    for edges, descriptions in zip(subgraph_info['Edges'], subgraph_info['Description']):
        cnt += 1
        description  = "G: " + ", ".join(descriptions) + "." 
        if cnt == 1:
            G1 = description
            questions = [f'Is Node {e} connected to Node {v}? The ans should only be yes or no.' for e, v in edges]
        if cnt == 2:
            G2 = description
    
    print(questions)
    messages = []
    prompt = 'You shold remeber the topology information in the Graph and no need to do other work'
    messages.append({"role": "system", "content": prompt})
    messages.append({"role": "user", "content": G1})
    outputs = llm.chat(messages, sampling_params=sampling_params)
    first = outputs[0].outputs[0].text
    print(first)
    messages.append({"role": "assistant", "content": first})
    messages.append({"role": "user", "content": G2})
    outputs = llm.chat(messages, sampling_params=sampling_params)
    second = outputs[0].outputs[0].text
    print(second)
    # 定义正面和负面词列表
    positive_words = ['yes']
    negative_words = ['no', 'not']

    # 初始化计数器
    pos = 0
    neg = 0

    # 定义一个函数来检查一个词是否在文本中作为一个完整的单词出现
    def is_word_in_text(word, text):
        pattern = r'\b' + re.escape(word) + r'\b'
        return bool(re.search(pattern, text))
    orign_messages = copy.deepcopy(messages)
    for question in questions:
        messages = copy.deepcopy(orign_messages)
        messages.append({"role": "assistant", "content": second})
        messages.append({"role": "user", "content": question})
        outputs = llm.chat(messages, sampling_params=sampling_params)
        print(question)
        ans = outputs[0].outputs[0].text
        # 将字符串转为小写，以便进行不区分大小写的匹配
        ans_lower = ans.lower()
        # 检查并计数正面词
        for word in positive_words:
            if is_word_in_text(word, ans_lower):
                pos += 1
        # 检查并计数负面词
        for word in negative_words:
            if is_word_in_text(word, ans_lower):
                neg += 1
        print(f'Ans: {ans} Pos: {pos} Neg: {neg}')
        to_remove = [-2, -1]
        # 删除元素
        for index in sorted(to_remove, reverse=True):
            if index >= 0:  # 确保索引是有效的
                messages.pop(index)

    print(f'Pos: {pos}')   
    print(f'Neg: {neg}') 
    break
    
