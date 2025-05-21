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


os.environ["CUDA_VISIBLE_DEVICES"] = "5"
model_name = "/home/data2t1/wangrongzheng/LLaMA-Factory-main/models/llama3_lora_dpo1"

sampling_params = SamplingParams(max_tokens=8192, temperature=0.7, top_p=1)

llm = LLM(model=model_name, tokenizer_mode="auto", tensor_parallel_size=1, dtype="half")



# 读取JSON文件
def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


prompt_file = '/home/data2t1/wangrongzheng/GTAgent/Web/Task_Agent/Llama3_Graph_Agent/simple_toolset/prompt_origin.txt'
with open(prompt_file, 'r', encoding='utf-8') as file:
    prompt = file.read()

prompt_file = '/home/data2t1/wangrongzheng/GTAgent/Web/Task_Agent/Llama3_Graph_Agent/re_prompt5.txt'
with open(prompt_file, 'r', encoding='utf-8') as file:
    prompt2 = file.read()

# 主程序
output_file_path = '/home/data2t1/wangrongzheng/GTAgent/Web/Dataset/1000_50_5ques/node_exists_output.json' # 替换为你保存的JSON文件路径
subgraphs_info = read_json_file(output_file_path)


# 输出结果保存路径
output_json_file = '/home/data2t1/wangrongzheng/GTAgent/Web/Dataset/1000_50_5ques/output/shortest_path.json'
# 如果结果文件不存在，则创建一个空列表文件
if not os.path.exists(output_json_file):
    with open(output_json_file, 'w') as f:
        json.dump([], f)

# 初始化计数器
true_count1 = 0
false_count1 = 0
true_count2 = 0
false_count2 = 0

# 循环读取每个子图并绘制
for index, subgraph_info in tqdm(enumerate(subgraphs_info), total=len(subgraphs_info), desc="Drawing Subgraphs"):
    print(f"Drawing Subgraph {index + 1}")
    question = subgraph_info['question']
    answer = subgraph_info['answer']
    messages = []
    messages.append({"role": "system", "content": prompt})
    messages.append({"role": "user", "content": question})
    outputs = llm.chat(messages, sampling_params=sampling_params)
    first = outputs[0].outputs[0].text
    print('<-------------------------------------------------------->')
    print(question)
    print('-------------------------------')
    print(first)

    messages.append({"role": "assistant", "content": first})
    messages.append({"role": "user", "content": prompt2})

    print('<-------------------------------------------------------->')

    # 将当前结果保存为一个字典
    result = {
        "index": index + 1,
        "prompt": prompt,
        "question": question,
        "generated_response": first,
    }

    # # 将结果追加到 JSON 文件
    # with open(output_json_file, 'r+') as f:
    #     # 读取现有数据
    #     existing_data = json.load(f)
    #     # 追加新的结果
    #     existing_data.append(result)
    #     # 回到文件开头并写入更新后的数据
    #     f.seek(0)
    #     json.dump(existing_data, f, indent=4)

    # 检查 first 中是否包含 "True" 或 "False"，并更新计数器
    if "tool_name: is_node_graphexistance" in first.lower():
        true_count1 += 1
    if "False" in first:
        false_count1 += 1


    # 打印统计结果
    print(f"True count: {true_count1}")
    print(f"False count: {false_count1}")


# 打印统计结果
print(f"True count: {true_count1}")
print(f"False count: {false_count1}")

