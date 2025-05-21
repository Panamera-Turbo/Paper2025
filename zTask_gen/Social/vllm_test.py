import os
import json
import re
from tqdm import tqdm
from vllm import LLM
from vllm.sampling_params import SamplingParams
from openai import OpenAI

# GPT 模型名称
GPT_MODEL = "gpt-4o-mini-2024-07-18"

client = OpenAI(
    api_key="sk-Hv9ensWblsslUzoI13C0A69bD2094fB0B6CeDb81709e17D5",
    base_url="https://api.holdai.top/v1"
)

def chat_completion_request(messages, model=GPT_MODEL):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=4096,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e

# 配置 LLM 模型
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
model_name = "/home/data2t1/wangrongzheng/LLaMA-Factory-main/models/llama3.1_lora_dpo+grpo"
#"/home/data2t1/wangrongzheng/LLaMA-Factory-main/models/llama3.1_lora_dpo+grpo"
sampling_params = SamplingParams(max_tokens=4096, temperature=0.7, top_p=1)

#llm = LLM(model=model_name, tokenizer_mode="auto", tensor_parallel_size=1, dtype="half")
llm = LLM(
    model=model_name,
    tokenizer_mode="auto",
    tensor_parallel_size=1,
    dtype="half",
    max_model_len=4096  # 根据实际需求调整
)

# 读取 JSON 文件的函数
def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# 读取 Prompt 文件
prompt_file = 'zTask_gen/prompt_R1.txt'
with open(prompt_file, 'r', encoding='utf-8') as file:
    prompt = file.read()

# 主程序
file_path = "zTask_gen/Social/json/Triangle_Count.json"
#zTask_gen/Social/json/Cycle_Detection.json
#zTask_gen/Social/json/Degree_Count.json
#zTask_gen/Social/json/Edge_Count.json
#zTask_gen/Social/json/Edge_Existence.json
#zTask_gen/Social/json/Node_Count.json
#zTask_gen/Social/json/Node_Existence.json
#zTask_gen/Social/json/Path_Existence.json
#zTask_gen/Social/json/Shortest_Path.json
#zTask_gen/Social/json/Triangle_Count.json



# 读取 JSON 数据
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 遍历每个条目
for entry in tqdm(data, desc="Processing Entries"):
    true_count = 0
    weak_count = 0
    # 提取 answer 字段
    answer = entry.get("answer", "")
    # 删除 <think>...</think> 之间的所有内容（包括标签本身）
    clean_answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL)
    # 分割字符串，保留“问题：”之后的内容
    processed_answer = clean_answer.split("问题：")[-1].strip()
    # 打印问题和标签
    #print(processed_answer)
    label = entry['label']
    print(label)

    print("-" * 50)

    # 翻译问题
    messages_temp = [{"role": "user", "content": '翻译为英文:' + processed_answer}]
    try:
        translated_answer = chat_completion_request(messages_temp).choices[0].message.content
    except Exception as e:
        print(f"Error in chat_completion_request: {e}")
        translated_answer = "Translation failed"
    print(translated_answer)

    # 对每个问题运行 20 次循环
    for i in tqdm(range(20), desc=f"Processing Question {label}", leave=False):
        # 构造消息列表
        mess = []
        mess.append({"role": "system", "content": prompt})
        mess.append({"role": "user", "content": translated_answer})
        # 调用 LLM 进行推理
        outputs = llm.chat(mess, sampling_params=sampling_params)
        first = outputs[0].outputs[0].text
        clean_first = re.sub(r'.*</think>\n\n', '', first, flags=re.DOTALL)
        print("-" * 10)
        print('clean_first: ', clean_first)
        print("-" * 10)
        # 检查工具名称是否匹配
        if label.lower() in clean_first.lower():
            weak_count += 1
        if ("tool_name: " + label).lower() in clean_first.lower():
            true_count += 1

    print(true_count)
    # 如果工具名称匹配次数达到阈值，则将问题添加到过滤列表中
    if true_count >= 15:
        filtered_question = {
            'origin_question': processed_answer,
            'question': translated_answer,
            'label': label,
            'type': entry.get("type", ""),
            'true_count': true_count,
            'weak_count': weak_count - true_count
        }

        # 动态写入 JSON 文件
        output_file_path = os.path.join('zTask_gen/Social/Anss', f"{label}_filtered.json")
        if not os.path.exists(output_file_path):
            # 如果文件不存在，创建并写入空列表
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False, indent=4)

        # 读取现有数据，追加新数据
        with open(output_file_path, 'r+', encoding='utf-8') as f:
            existing_data = json.load(f)
            existing_data.append(filtered_question)
            f.seek(0)  # 回到文件开头
            json.dump(existing_data, f, ensure_ascii=False, indent=4)

    print("-" * 50)
