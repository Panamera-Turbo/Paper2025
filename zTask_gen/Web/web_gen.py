import json
from openai import OpenAI
from tqdm import tqdm
import os

GPT_MODEL = "deepseek-r1"

client = OpenAI(
    api_key="sk-8lGidjXrai4pj3TVF4Bc2aD6727b45D6B8E3C0C22b991434",
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

# 配置文件路径
input_folder = 'zTask_gen/Web/p2'  # 替换为您的txt文件夹路径
output_folder = 'zTask_gen/Web/json'  # 替换为输出JSON文件夹路径

# 如果输出文件夹不存在，则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取所有txt文件
txt_files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]

for txt_file in tqdm(txt_files, desc="Processing Files"):
    # 读取文件内容和标题
    file_path = os.path.join(input_folder, txt_file)
    with open(file_path, 'r', encoding='utf-8') as f:
        mess = f.read().strip()
    title = os.path.splitext(txt_file)[0]

    # 输出JSON文件路径（基于txt文件名）
    output_file_path = os.path.join(output_folder, f"{title}.json")

    # 初始化数据存储
    data = []
    if os.path.exists(output_file_path):
        try:
            with open(output_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            data = []

    original_string = mess
    # 总次数
    total_count = 400

    # 定义替换内容列表
    contents = [
        "网页爬虫优化",
        "搜索引擎排名优化",
        "网站结构健康诊断",
        "主题社区发现",
        "网络攻击防御"
    ]

    # 每个内容的打印次数
    count_per_content = total_count // len(contents)

    # 替换并打印整个字符串
    for content in tqdm(contents, desc=f"Calling API for {txt_file}"):
        for _ in tqdm(range(count_per_content), desc=f"content {content}"):
            # 替换字符串中的 {{}} 为当前内容
            modified_string = original_string.replace("{{}}", content)
            print(modified_string)

            messages = [{"role": "user", "content": modified_string}]
            chat_response = chat_completion_request(messages)
            
            if isinstance(chat_response, Exception):
                continue
            
            # 获取API响应内容
            answer = chat_response.choices[0].message.content
            print(answer)
            
            # 记录数据
            data.append({
                'prompt': modified_string,
                'label': title,
                'type': content,
                'answer': answer
            })
            
            # 实时保存结果
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)


print("所有JSON文件已更新。")
