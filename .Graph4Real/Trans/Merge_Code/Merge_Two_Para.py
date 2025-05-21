# import json
# import re

# def format_edges(edges):
#     edge_strs = []
#     for edge in edges:
#         edge_strs.append(f"{edge}")
#     return ", ".join(edge_strs)

# def extract_numbers_from_question(question):
#     """从问题中提取两个数字"""
#     numbers = re.findall(r'\d+', question)
#     if len(numbers) >= 2:
#         return numbers[0], numbers[1]
#     return None, None

# def replace_specific_patterns(question, pattern_replacements):
#     """替换问题中的特定模式"""
#     for pattern, replacement in pattern_replacements.items():
#         question = question.replace(pattern, replacement)
#     return question

# def merge_jsons(json1_path, json2_path, output_path):
#     # 加载两个JSON文件
#     with open(json1_path, 'r', encoding='utf-8') as f1:
#         data1 = json.load(f1)
    
#     with open(json2_path, 'r', encoding='utf-8') as f2:
#         data2 = json.load(f2)
    
#     # 确保data1比data2长
#     if len(data1) < len(data2):
#         raise ValueError("json1的长度必须大于或等于json2的长度")
    
#     merged_data = []
    
#     # 遍历json2中的每个对象
#     for i, item2 in enumerate(data2):
#         # 获取对应的json1中的对象
#         item1 = data1[i]
        
#         # 1. 从json1的question中提取两个数字
#         first_num, second_num = extract_numbers_from_question(item1.get("question", ""))
        
#         # 2. 处理json2的question，替换特定模式
#         original_question = item2.get("question", "")
#         if first_num and second_num:
#             modified_question = replace_specific_patterns(
#                 original_question,
#                 {
#                     "0302": first_num,
#                     "0714": second_num
#                 }
#             )
#         else:
#             modified_question = original_question
        
#         # 3. 处理 Edges，拼接成字符串
#         edges_str = format_edges(item1.get("Edges", []))
#         full_edges = f"The edges are: {edges_str}."
        
#         # 4. 拼接最终的question = 修改后的question + edges信息
#         full_ques = f"{modified_question} {full_edges}".strip()
        
#         # 创建新的合并后的对象
#         merged_item = {
#             "origin_question": item2.get("origin_question", ""),
#             "translated_question": item2.get("question", ""),
#             "question": full_ques,
#             "type": item2.get("label", ""),
#             "background_type": item2.get("type", ""),
#             "Node_List": item1.get("Node_List", []),
#             "Edges": item1.get("Edges", []),
#             "Edge_Count": item1.get("Edge_Count", 0),
#             "Description": item1.get("Description", ""),
#             "answer": item1.get("answer", ""),
#         }
        
#         merged_data.append(merged_item)
    
#     # 将合并后的数据写入新文件
#     with open(output_path, 'w', encoding='utf-8') as f:
#         json.dump(merged_data, f, ensure_ascii=False, indent=4)
    
#     print(f"合并完成，结果已保存到 {output_path}")


# # 使用示例
# merge_jsons('/home/data2t1/tempuser/GTAgent/Transport/gen/1000_50_nodes/is_reachable_output.json', 
#             '/home/data2t1/tempuser/GTAgent/zTask_gen/Trans/Ans/Path_Existence_filtered.json', 
#             '/home/data2t1/tempuser/GTAgent/.Graph4Real/Trans/Path_Existence.json')

# # '/home/data2t1/tempuser/GTAgent/.Graph4Real/Trans/Cycle_Detection.json'


import json
import re
import os
from pathlib import Path

def format_edges(edges):
    edge_strs = []
    for edge in edges:
        edge_strs.append(f"{edge}")
    return ", ".join(edge_strs)

def extract_numbers_from_question(question):
    """从问题中提取两个数字"""
    numbers = re.findall(r'\d+', question)
    if len(numbers) >= 2:
        return numbers[0], numbers[1]
    return None, None

def replace_specific_patterns(question, pattern_replacements):
    """替换问题中的特定模式"""
    for pattern, replacement in pattern_replacements.items():
        question = question.replace(pattern, replacement)
    return question

def process_single_pair(json1_path, json2_path, output_path):
    """处理单个JSON文件对"""
    # 加载两个JSON文件
    with open(json1_path, 'r', encoding='utf-8') as f1:
        data1 = json.load(f1)
    
    with open(json2_path, 'r', encoding='utf-8') as f2:
        data2 = json.load(f2)
    
    # 确保data1比data2长
    if len(data1) < len(data2):
        raise ValueError(f"{json1_path}的长度必须大于或等于{json2_path}的长度")
    
    merged_data = []
    
    # 遍历json2中的每个对象
    for i, item2 in enumerate(data2):
        # 获取对应的json1中的对象
        item1 = data1[i]
        
        # 1. 从json1的question中提取两个数字
        first_num, second_num = extract_numbers_from_question(item1.get("question", ""))
        
        # 2. 处理json2的question，替换特定模式
        original_question = item2.get("question", "")
        if first_num and second_num:
            modified_question = replace_specific_patterns(
                original_question,
                {
                    "0302": first_num,
                    "0714": second_num
                }
            )
        else:
            modified_question = original_question
        
        # 3. 处理 Edges，拼接成字符串
        edges_str = format_edges(item1.get("Edges", []))
        full_edges = f"The edges are: {edges_str}."
        
        # 4. 拼接最终的question = 修改后的question + edges信息
        full_ques = f"{modified_question} {full_edges}".strip()
        
        # 创建新的合并后的对象
        merged_item = {
            "origin_question": item2.get("origin_question", ""),
            "translated_question": modified_question,
            "question": full_ques,
            "type": item2.get("label", ""),
            "background_type": item2.get("type", ""),
            "Node_List": item1.get("Node_List", []),
            "Edges": item1.get("Edges", []),
            "Edge_Count": item1.get("Edge_Count", 0),
            "Description": item1.get("Description", ""),
            "answer": item1.get("answer", ""),
        }
        
        merged_data.append(merged_item)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 将合并后的数据写入新文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)
    
    print(f"合并完成，结果已保存到 {output_path}")

def merge_jsons_from_folders(folder1_path, folder2_path, output_folder_path):
    """处理两个文件夹中的所有匹配的JSON文件"""
    # 获取第一个文件夹中的所有JSON文件
    folder1_files = [f for f in os.listdir(folder1_path) if f.endswith('.json')]
    
    # 遍历第一个文件夹中的每个JSON文件
    for filename in folder1_files:
        json1_path = os.path.join(folder1_path, filename)
        json2_path = os.path.join(folder2_path, filename)
        output_path = os.path.join(output_folder_path, filename)
        
        # 检查第二个文件夹中是否存在同名文件
        if os.path.exists(json2_path):
            try:
                process_single_pair(json1_path, json2_path, output_path)
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")
        else:
            print(f"警告: {json2_path} 不存在，跳过处理")

# 使用示例
merge_jsons_from_folders(
    '/home/data2t1/tempuser/GTAgent/Transport/gen/100_50_nodes/Two_para', 
    '/home/data2t1/tempuser/GTAgent/zzzQuesGen/Trans/translated_json', 
    '/home/data2t1/tempuser/GTAgent/zzzQuesGen/Trans/Ques_to_evalute'
)


# 使用示例
# merge_jsons_from_folders(
#     'Transport/gen/40_50_nodes/Two_para', 
#     'zTask_gen/Trans/Ans', 
#     '.Graph4Real/Trans/Json/Small/Trans'
# )