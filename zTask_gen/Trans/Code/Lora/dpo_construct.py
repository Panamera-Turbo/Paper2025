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
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
model_name = "/home/data2t1/tempuser/DeepSeek-R1-Distill-Llama-8B"
sampling_params = SamplingParams(max_tokens=4096, temperature=0.7, top_p=1)

llm = LLM(
    model=model_name,
    tokenizer_mode="auto",
    tensor_parallel_size=2,
    dtype="half",
    max_model_len=4096
)

# 读取 Prompt 文件
prompt_file = 'zTask_gen/prompt_R1_ver2.txt'
with open(prompt_file, 'r', encoding='utf-8') as file:
    prompt = file.read()

# 主程序
file_path = "zTask_gen/Trans/json/Graph_Diameter.json"

# 读取 JSON 数据
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 遍历每个条目
for entry in tqdm(data, desc="Processing Entries"):
    # 提取 answer 字段
    answer = entry.get("answer", "")
    # 删除 >Thinking...>... 之间的所有内容（包括标签本身）
    clean_answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL)
    # 分割字符串，保留"问题："之后的内容
    processed_answer = clean_answer.split("问题：")[-1].strip()
    label = entry['label']
    
    # 翻译问题
    messages_temp = [{"role": "user", "content": '翻译为英文,不要输出其他任何多余信息:' + processed_answer}]
    try:
        translated_answer = chat_completion_request(messages_temp).choices[0].message.content
    except Exception as e:
        print(f"Error in chat_completion_request: {e}")
        translated_answer = "Translation failed"
    
    # 初始化 chosen 和 rejected
    chosen = None
    rejected = None
    
    # 对每个问题运行 10 次循环
    for i in range(10):
        # 构造消息列表
        mess = []
        mess.append({"role": "system", "content": prompt})
        mess.append({"role": "user", "content": translated_answer})
        
        # 调用 LLM 进行推理
        outputs = llm.chat(mess, sampling_params=sampling_params)
        first = outputs[0].outputs[0].text
        clean_first = re.sub(r'.*\n\n', '', first, flags=re.DOTALL)
        
        # 检查是否满足 chosen 条件
        if ("tool_name: " + 'NULL').lower() in clean_first.lower() or ("tool_name: " + 'None').lower() in clean_first.lower():
            if chosen is None:
                chosen = first
        else:
            if rejected is None:
                rejected = first
        
        # 如果两者都已赋值，则跳出循环
        if chosen is not None and rejected is not None:
            break
    
    # 如果超过10次循环仍未赋值，则设为'no found'
    if chosen is None:
        chosen = 'no found'
    if rejected is None:
        rejected = 'no found'
    
    # 构建最终JSON结构
    result = {
        "conversations": [
            {
                "from": "system",
                "value": prompt
            },
            {
                "from": "human",
                "value": translated_answer
            }
        ],
        "chosen": {
            "from": "gpt",
            "value": chosen
        },
        "rejected": {
            "from": "gpt",
            "value": rejected
        }
    }
    
    # 写入JSON文件
    output_file_path = os.path.join('zTask_gen/Trans/Code/Filtered', f"{label}_filtered_v2_grpo+dpo.json")
    
    # 检查文件是否存在，不存在则创建空列表
    if not os.path.exists(output_file_path):
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=4)
    
    # 读取现有数据，追加新数据
    with open(output_file_path, 'r+', encoding='utf-8') as f:
        existing_data = json.load(f)
        existing_data.append(result)
        f.seek(0)  # 回到文件开头
        json.dump(existing_data, f, ensure_ascii=False, indent=4)
