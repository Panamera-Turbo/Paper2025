import re
import subprocess
import tempfile
import sys
import os

def execute_combined_code(code_part1, llm_output, conda_env_name="GLM"):
    # 代码块提取（使用正则表达式匹配三个反引号包裹的Python代码）
    code_blocks = re.findall(r'```python(.*?)```', llm_output, re.DOTALL)
    code_part2 = code_blocks[0].strip() if len(code_blocks) > 0 else ""
    code_part3 = code_blocks[1].strip() if len(code_blocks) > 1 else ""

    # 构建完整代码（添加必要依赖）
    full_code = f"""
import sys
{code_part1}

# 补充可能缺失的依赖
try:
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv
    from torch_geometric.data import Data
except ImportError as e:
    print("Missing dependencies:", e)
    sys.exit(1)

{code_part2}

# 执行段三的调用
if __name__ == "__main__":
    try:
        result = {code_part3.split('(')[0].strip()}(data)
        print("Execution Result:", result)
    except Exception as e:
        print("Execution Error:", str(e))
"""

# 创建临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(full_code)
        tmp_path = f.name

    # 构造conda执行命令（完整路径初始化）
    conda_sh_path = "/home/data2t1/wangrongzheng/miniconda3/etc/profile.d/conda.sh"
    
    if sys.platform.startswith('win'):
        python_command = fr"conda activate {conda_env_name} && python"
    else:
        # 显式加载conda初始化脚本
        python_command = (
            f"bash -c '"
            f"source {conda_sh_path} && "
            f"conda activate {conda_env_name} && "
            f"python {tmp_path}'"
        )

    # 执行脚本
    try:
        result = subprocess.run(
            python_command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        output = result.stdout
    except subprocess.CalledProcessError as e:
        output = f"ERROR: {e.stderr}"
    except Exception as e:
        output = f"UNEXPECTED ERROR: {str(e)}"

    # 清理临时文件
    os.unlink(tmp_path)
    
    return full_code, output

# 使用示例（需替换实际参数）
part1 = """import networkx as nx
import random

random.seed(42)  # 固定随机种子

def Gen():
    # 创建一个无向图（关键修改点）
    G = nx.Graph()

    # 添加节点（0到999）
    nodes = range(1000)
    G.add_nodes_from(nodes)

    # 随机添加边，边权为1-50的随机整数
    for u in nodes:
        # 随机选择一些目标节点（每个节点平均5条边）
        for v in random.sample(nodes, 5):
            if u != v:  # 避免自环
                weight = random.randint(1, 50)
                G.add_edge(u, v, capacity=weight)
    return G 

data = Gen()
    """



import os
import json
import re
from tqdm import tqdm
from vllm import LLM
from vllm.sampling_params import SamplingParams
from openai import OpenAI

# GPT 模型名称
GPT_MODEL = "deepseek-v3-0324"

client = OpenAI(
    api_key="sk-8lGidjXrai4pj3TVF4Bc2aD6727b45D6B8E3C0C22b991434",
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


# 读取 JSON 文件的函数
def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# 主程序
file_path = "zTask_gen/Trans/json/Maxflow.json"


# 读取 JSON 数据
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 遍历每个条目
for entry in tqdm(data, desc="Processing Entries"):
    # 提取 answer 字段
    answer = entry.get("answer", "")
    # 删除 <think>...</think> 之间的所有内容（包括标签本身）
    clean_answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL)
    # 分割字符串，保留“问题：”之后的内容
    processed_answer = clean_answer.split("问题：")[-1].strip()
    processed_answer = processed_answer.replace("0302", "302")
    processed_answer = processed_answer.replace("0714", "714")
    # 打印问题和标签
    #print(processed_answer)
    label = entry['label']
    print(label)
    print(processed_answer)
    messages_temp = [{"role": "user", "content": '把这个句子翻译为翻译为英文，注意仅仅输出翻译后的内容不要有其他任何信息:' + processed_answer}]
    try:
        translated_answer = chat_completion_request(messages_temp).choices[0].message.content
    except Exception as e:
        print(f"Error in chat_completion_request: {e}")
        translated_answer = "Translation failed"
    print(translated_answer)

    print("-" * 50)

    true_count  = 0
    error_count = 0
    op = []
    code_list = []
    full_code_list = []
    # 对每个问题运行 5 次循环
    for i in tqdm(range(5), desc=f"Processing Question {label}"):
        
        sys_prompt = '''
        Assumption: The undirected graph *G* is already structured in NetworkX format, represented by `data`, and the node identifiers are of integer type (`int`). Please write a Python function to solve the problem above. Your output should strictly follow the given format.  

        ```python  
        def method(data):  
            ...  
            return  
        ```  

        Additionally, provide a single line of code that calls this function. The return value must be exactly one numerical value. Only provide the function call in the specified format—I will automatically retrieve the return value:  

        ```python  
        method(data)  
        ```  

        Note: Strictly adhere to the specified format.
        '''
        mess = []
        mess.append({"role": "system", "content": translated_answer + '\n' + sys_prompt})
        mess.append({"role": "user", "content": ''})
        print(f'Ques:{translated_answer}')
        try:
            answer = chat_completion_request(mess).choices[0].message.content
        except Exception as e:
            print(f"Error in chat_completion_request: {e}")
            answer = "Fetch failed"
        llm_output = answer
        code_list.append(answer)
        try:
            full_code, output = execute_combined_code(part1, llm_output, "GLM")
            op.append(output)
            if output == 1 or output == '1':
                true_count += 1
        except Exception as e:
            print(f'Error:{e}')
            output = -1
            op.append(output)
            error_count += 1
        print(f'output: {output}')
        print("-" * 50)
        print(full_code)
        full_code_list.append(full_code)
        print("-" * 50)

    # 如果工具名称匹配次数达到阈值，则将问题添加到过滤列表中
    filtered_question = {
        'origin_question': processed_answer,
        'translated_answer': translated_answer,
        'label': label,
        'output':op,
        'code_gen': code_list,
        'full_code': full_code_list,
        'type': entry.get("type", ""),
        'true_count': true_count,
        'error_count': error_count
    }

    # 动态写入 JSON 文件
    output_file_path = os.path.join('zTask_gen/Trans/Code/Ans', f"{label}_filtered_trans.json")
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



