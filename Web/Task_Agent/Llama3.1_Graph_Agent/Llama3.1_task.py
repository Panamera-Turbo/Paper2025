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


prompt_file = '/home/data2t1/tempuser/GTAgent/Web/Task_Agent/Llama3_Graph_Agent/prompt_origin.txt'
with open(prompt_file, 'r', encoding='utf-8') as file:
    prompt = file.read()

prompt_file = '/home/data2t1/tempuser/GTAgent/Web/Task_Agent/Llama3_Graph_Agent/re_prompt5.txt'
with open(prompt_file, 'r', encoding='utf-8') as file:
    prompt2 = file.read()

# 主程序
output_file_path = '/home/data2t1/tempuser/GTAgent/Web/Dataset/1000_50_5ques/node_degree_output.json' # 替换为你保存的JSON文件路径
subgraphs_info = read_json_file(output_file_path)


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
    print('<-------------------------------------------------------->')
    print(question)
    print('-------------------------------')
    print(first)
    print('-------------------------------')

    messages.append({"role": "assistant", "content": first})
    messages.append({"role": "user", "content": prompt2})

    outputs = pipeline(
        messages,
        max_new_tokens=8192,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.7,
        top_p=1,
    )

    second = outputs[0]["generated_text"][-1]['content']
    print(second)
    print('<-------------------------------------------------------->')

    # 检查 first 中是否包含 "True" 或 "False"，并更新计数器
    if "degree_graphCount" in first:
        true_count1 += 1
    if "False" in first:
        false_count1 += 1
    # 检查 first 中是否包含 "True" 或 "False"，并更新计数器
    if "True" in second:
        true_count2 += 1
    if "False" in second:
        false_count2 += 1

    # 打印统计结果
    print(f"True count: {true_count1}")
    print(f"False count: {false_count1}")
    # 打印统计结果
    print(f"True count: {true_count2}")
    print(f"False count: {false_count2}")

# 打印统计结果
print(f"True count: {true_count1}")
print(f"False count: {false_count1}")
print(f"True count: {true_count2}")
print(f"False count: {false_count2}")
