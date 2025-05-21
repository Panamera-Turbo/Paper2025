import json
import re
import numpy as np

# 标准值
standard_value = 248

# 存储提取的预测值
predictions = []

# 读取JSON文件
json_file_path = 'zTask_gen/Code/trans_predict/Count_result_oot/Ans/trans_predict_filtered_Llama_hyper_processed.json'  # 替换为你的JSON文件路径
with open(json_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)  # 假设文件是JSON数组
    # 如果文件是每行一个JSON对象，可以用以下方式：
    # for line in f:
    #     data = json.loads(line)
    #     处理逻辑...

# 遍历每个对象
for item in data:
    try:
        output = item.get('output', '')
        # 使用正则表达式提取数值
        match = re.search(r'Execution Result:\s*([\d.]+)', output)
        if match:
            value = float(match.group(1))
            predictions.append(value)
        else:
            # 无法解析时赋值为0
            predictions.append(0)
            print(f"无法解析output，赋值为0: {output}")
    except Exception as e:
        # 其他异常情况也赋值为0
        predictions.append(0)
        print(f"处理时发生异常，赋值为0: {e}")

# 转换为numpy数组便于计算
predictions = np.array(predictions)

# 计算误差
if len(predictions) > 0:
    # 平均绝对误差(MAE)
    mae = np.mean(np.abs(predictions - standard_value))
    
    # 均方误差(MSE)
    mse = np.mean((predictions - standard_value) ** 2)
    
    print(f"处理的预测值数量: {len(predictions)}")
    print(f"MAE (相对于248): {mae:.4f}")
    print(f"MSE (相对于248): {mse:.4f}")
else:
    print("没有提取到任何预测值")



import json
import re
import numpy as np

# 标准值
standard_value = 248

# 存储提取的预测值
predictions = []

# 读取JSON文件
json_file_path = 'zTask_gen/Code/trans_predict/Count_result_oot/Ans/trans_predict_filtered_Llama_hyper_processed.json'  # 替换为你的JSON文件路径
with open(json_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)  # 假设文件是JSON数组
    # 如果文件是每行一个JSON对象，可以用以下方式：
    # for line in f:
    #     data = json.loads(line)
    #     处理逻辑...

# 遍历每个对象
for item in data:
    try:
        output = item.get('output', '')
        # 使用正则表达式提取数值
        match = re.search(r'Execution Result:\s*([\d.]+)', output)
        if match:
            value = float(match.group(1))
            predictions.append(value)
    except (AttributeError, ValueError) as e:
        print(f"跳过无法解析的项目: {e}")
        continue

# 转换为numpy数组便于计算
predictions = np.array(predictions)

# 计算误差
if len(predictions) > 0:
    # 平均绝对误差(MAE)
    mae = np.mean(np.abs(predictions - standard_value))
    
    # 均方误差(MSE)
    mse = np.mean((predictions - standard_value) ** 2)
    
    print(f"提取的预测值数量: {len(predictions)}")
    print(f"MAE (相对于248): {mae:.4f}")
    print(f"MSE (相对于248): {mse:.4f}")
else:
    print("没有提取到有效的预测值")



# import json
# import re
# import numpy as np

# # 标准值
# standard_value = 248

# # 存储提取的预测值
# predictions = []

# # 读取JSON文件
# json_file_path = 'zTask_gen/Code/trans_predict/Count_result_oot/Ans/trans_predict_filtered_Llama_hyper.json'  # 替换为你的JSON文件路径
# with open(json_file_path, 'r', encoding='utf-8') as f:
#     data = json.load(f)  # 假设文件是JSON数组
#     # 如果文件是每行一个JSON对象，可以用以下方式：
#     # for line in f:
#     #     data = json.loads(line)
#     #     处理逻辑...

# # 遍历每个对象
# for item in data:
#     try:
#         output = item.get('output', '')
#         # 使用正则表达式提取数值
#         match = re.search(r'Execution Result:\s*([\d.]+)', output)
#         if match:
#             value = float(match.group(1))
#             # 添加条件：仅当数值 >= 200时才保留
#             if value >= 200:
#                 predictions.append(value)
#             else:
#                 print(f"跳过数值小于200的结果: {value}")
#     except (AttributeError, ValueError) as e:
#         print(f"跳过无法解析的项目: {e}")
#         continue

# # 转换为numpy数组便于计算
# predictions = np.array(predictions)

# # 计算误差
# if len(predictions) > 0:
#     # 平均绝对误差(MAE)
#     mae = np.mean(np.abs(predictions - standard_value))
    
#     # 均方误差(MSE)
#     mse = np.mean((predictions - standard_value) ** 2)
    
#     print(f"提取的预测值数量: {len(predictions)}")
#     print(f"过滤后预测值示例: {predictions[:5]}...")  # 打印前5个值作为示例
#     print(f"MAE (相对于248): {mae:.4f}")
#     print(f"MSE (相对于248): {mse:.4f}")
# else:
#     print("没有提取到有效的预测值（或所有值都小于200）")

