import os
import json
import re
from tqdm import tqdm
from openai import OpenAI

# GPT 模型和 OpenAI 客户端初始化
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
            max_tokens=8192,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e

# 读取JSON文件
def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# 保存JSON文件
def save_json_file(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

# 主程序
def process_json_files(folder_path, output_folder):
    # 确保输出目录存在
    os.makedirs(output_folder, exist_ok=True)

    # 获取文件夹中所有的 JSON 文件
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

    for json_file in tqdm(json_files, desc="Processing JSON Files"):
        input_file_path = os.path.join(folder_path, json_file)
        output_file_path = os.path.join(output_folder, f"{os.path.splitext(json_file)[0]}_translated.json")

        # 读取JSON文件
        data = read_json_file(input_file_path)
        filtered_questions = []

        # 遍历每个条目
        for entry in tqdm(data, desc=f"Processing Entries in {json_file}", leave=False):
            # 提取answer字段
            answer = entry.get("answer", "")
            # 删除 <think>...</think> 之间的所有内容（包括标签本身）
            clean_answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL)
            # 分割字符串，保留“问题：”之后的内容
            processed_answer = clean_answer.split("问题：")[-1].strip()
            # 打印结果
            print(processed_answer)
            label = entry.get('label', '')
            print(label)

            # 翻译问题
            messages = [{"role": "user", "content": '翻译为英文:' + processed_answer}]
            try:
                translated_answer = chat_completion_request(messages).choices[0].message.content
            except Exception as e:
                print(f"Error in chat_completion_request: {e}")
                translated_answer = "Translation failed"
            print(translated_answer)

            # 保存过滤后的结果
            filtered_questions.append({
                'origin_question': processed_answer,
                'label': label,
                'translated_question': translated_answer
            })

        # 保存结果到新的 JSON 文件
        save_json_file(filtered_questions, output_file_path)
        print(f"Processed file saved to: {output_file_path}")

# 指定输入文件夹路径和输出文件夹路径
folder_path = "/root/wrz_temp/GTAgent_0123/zTask_Gen/Web/Json"  # 输入文件夹路径
output_folder = "/root/wrz_temp/GTAgent_0123/zTask_Gen/Web/Translated_Json"  # 输出文件夹路径

# 调用主程序
process_json_files(folder_path, output_folder)
