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
        node_min = min_vals[301, 0]
        node_max = max_vals[301, 0]
        print("Execution Result:", result * (node_max - node_min) + node_min)
    except Exception as e:
        print("Execution Error:", str(e))
"""

# 创建临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(full_code)
        tmp_path = f.name

    # 构造conda执行命令（完整路径初始化）
    conda_sh_path = "/home/data2t1/tempuser/miniconda3/etc/profile.d/conda.sh"
    
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
part1 = """import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os


# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 数据路径
npz_path = "/home/data2t1/tempuser/GTAgent/zTrain_test/transport/PEMS03.npz"
csv_path = "/home/data2t1/tempuser/GTAgent/zTrain_test/transport/PEMS03.csv"

class TrafficData:
    def __init__(self, X, y):
        self.X = X
        self.y = y

# 加载数据
def load_data():
    # 加载交通流量数据
    data = np.load(npz_path)
    traffic_data = data['data']  # 形状为 (26208, 358, 1)
    
    # 保存最后12个时刻的数据到CSV
    last_12 = traffic_data[-12:]  # 获取最后12个时刻的数据
    
    # 去掉最后12个时刻的数据
    traffic_data = traffic_data[:-12]
    
    # 加载距离数据
    distance_df = pd.read_csv(csv_path)
    
    return traffic_data, distance_df

# 数据预处理
def preprocess_data(traffic_data, seq_len=12, pred_len=1):
    # 1. 计算样本数量和数据维度
    n_samples = traffic_data.shape[0] - seq_len - pred_len + 1
    n_nodes = traffic_data.shape[1]
    n_features = traffic_data.shape[2]
    
    # 2. 初始化输入输出容器
    X = np.zeros((n_samples, seq_len, n_nodes, n_features))
    y = np.zeros((n_samples, n_nodes, n_features))
    
    # 3. 定义异常值处理函数（基于IQR四分位距）
    def remove_outliers(data):
        q1 = np.percentile(data, 25)  # 第一四分位数
        q3 = np.percentile(data, 75)  # 第三四分位数
        iqr = q3 - q1  # 四分位距
        lower_bound = q1 - 1.5 * iqr  # 下限
        upper_bound = q3 + 1.5 * iqr  # 上限
        
        # 将超出范围的值裁剪到边界
        return np.clip(data, lower_bound, upper_bound)
    
    # 4. 对每个节点的时序数据分别处理异常值
    for node in range(n_nodes):
        traffic_data[:, node, 0] = remove_outliers(traffic_data[:, node, 0])
    
    # 5. 按节点进行Min-Max归一化（保留极值用于后续反归一化）
    min_vals = np.min(traffic_data, axis=0)  # 每个节点的最小值
    max_vals = np.max(traffic_data, axis=0)  # 每个节点的最大值
    
    # 处理最大值等于最小值的情况（避免除零错误）
    max_vals[max_vals == min_vals] = 1
    traffic_data = (traffic_data - min_vals) / (max_vals - min_vals)
    
    # 6. 构建时间序列样本
    for i in range(n_samples):
        X[i] = traffic_data[i:i+seq_len]  # 输入序列
        y[i] = traffic_data[i+seq_len+pred_len-1]  # 预测目标
    
    # 7. 调整形状适应模型输入
    X = X.reshape(n_samples, seq_len, -1)  # 展平节点和特征维度
    y = y.reshape(n_samples, -1)
    
    # 8. 封装数据并保存归一化参数
    traffic_data_obj = TrafficData(X, y)
    traffic_data_obj.min_vals = min_vals  # 存储最小值（用于预测结果反归一化）
    traffic_data_obj.max_vals = max_vals  # 存储最大值
    
    return traffic_data_obj, min_vals, max_vals

traffic_data, distance_df = load_data()
data,min_vals,max_vals = preprocess_data(traffic_data, seq_len=12, pred_len=1)
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
file_path = "zTask_gen/Code/trans_predict/json/trans_predict.json"


# 读取 JSON 数据
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 遍历每个条目
for entry in tqdm(data[100:199], desc="Processing Entries"):
    # 提取 answer 字段
    answer = entry.get("answer", "")
    # 删除 <think>...</think> 之间的所有内容（包括标签本身）
    clean_answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL)
    # 分割字符串，保留“问题：”之后的内容
    processed_answer = clean_answer.split("问题：")[-1].strip()
    processed_answer = processed_answer.replace("0302", "302")
    # 打印问题和标签
    #print(processed_answer)
    label = entry['label']
    print(label)
    print(processed_answer)

    print("-" * 50)

    true_count  = 0
    error_count = 0
    op = []
    code_list = []
    full_code_list = []
    # 对每个问题运行 10 次循环
    for i in tqdm(range(10), desc=f"Processing Question {label}"):
        
        sys_prompt = '''
        假设：其中data已经。，data.X为维度为(样本数, 时间步长, 节点总数)，data.y的维度为(样本数, 节点总数)，时间步长为12，节点总数为358。
        要求：不要使用keras，一共给出我两段代码，第一段是完整的解决代码，第二段仅有一行，读入我给定的数据，调用第一段代码并计算得到结果。下一时刻指的是在已知时间之外的下一个步长的数据。代码优先选择使用GPU进行训练，如果没有可用GPU则使用CPU。
        ```python
        def method(data)
        ....
        return
        ```
        再给出我一行代码，是针对这个问题的解决函数的调用语句,返回值有且仅有一个数值，注意仅按照我的格式给出调用函数即可，我会自动获取相关的返回参数：

        ```python
        method(data)
        ```

        注意严格遵守我要求的格式
        '''
        mess = []
        mess.append({"role": "system", "content": processed_answer + '\n' + sys_prompt})
        mess.append({"role": "user", "content": ''})
        print(f'Ques:{processed_answer}')
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
        'answer': answer,
        'label': label,
        'output':op,
        'code_gen': code_list,
        'full_code': full_code_list,
        'type': entry.get("type", ""),
        'true_count': true_count,
        'error_count': error_count
    }

    # 动态写入 JSON 文件
    output_file_path = os.path.join('zTask_gen/Code/trans_predict/Ans', f"{label}_filtered_2.json")
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



