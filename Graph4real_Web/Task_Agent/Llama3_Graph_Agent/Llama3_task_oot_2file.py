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

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
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


prompt_file = '/home/data2t1/tempuser/GTAgent/Web/Task_Agent/Llama3_Graph_Agent/simple_toolset/prompt_origin.txt'
with open(prompt_file, 'r', encoding='utf-8') as file:
    prompt = file.read()

prompt_file = '/home/data2t1/tempuser/GTAgent/Web/Task_Agent/Llama3_Graph_Agent/re_prompt5.txt'
with open(prompt_file, 'r', encoding='utf-8') as file:
    prompt2 = file.read()

# 主程序
#/home/data2t1/tempuser/GTAgent/Web/Dataset/1000_50_5ques/pagerank_output.json
#/home/data2t1/tempuser/GTAgent/Web/Dataset/1000_50_5ques/common_neighbors_output.json
#/home/data2t1/tempuser/GTAgent/Web/Dataset/1000_50_5ques/jaccard_coefficient_output.json
#/home/data2t1/tempuser/GTAgent/Web/Dataset/1000_50_5ques/graph_diameter_output.json
#/home/data2t1/tempuser/GTAgent/Web/Dataset/1000_50_5ques/hits_scores_output.json
output_file_path = '/home/data2t1/tempuser/GTAgent/Web/Dataset/1000_50_5ques/pagerank_output.json' # 替换为你保存的JSON文件路径
subgraphs_info = read_json_file(output_file_path)


# 输出结果保存路径
output_json_file = '/home/data2t1/tempuser/GTAgent/Web/Dataset/1000_50_5ques/output/pagerank.json'
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

    # 将结果追加到 JSON 文件
    with open(output_json_file, 'r+') as f:
        # 读取现有数据
        existing_data = json.load(f)
        # 追加新的结果
        existing_data.append(result)
        # 回到文件开头并写入更新后的数据
        f.seek(0)
        json.dump(existing_data, f, indent=4)

    # 检查 first 中是否包含 "True" 或 "False"，并更新计数器
    if "none" in first.lower() or "null" in first.lower() or "tool_name: pagerank" in first.lower():
        true_count1 += 1
    if "False" in first:
        false_count1 += 1


    # 打印统计结果
    print(f"True count: {true_count1}")
    print(f"False count: {false_count1}")


# 打印统计结果
print(f"True count: {true_count1}")
print(f"False count: {false_count1}")

