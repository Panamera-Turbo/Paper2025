import json
import re
import os
from pathlib import Path

def format_edges(edges):
    edge_strs = []
    for edge in edges:
        edge_strs.append(f"{edge}")
    return ", ".join(edge_strs)

def extract_number_from_question(question):
    """从问题中提取数字"""
    match = re.search(r'\d+', question)
    return match.group(0) if match else None

def replace_pattern_in_question(question, pattern, replacement):
    """替换问题中的所有模式匹配项"""
    return re.sub(pattern, replacement, question)

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
        
        # 1. 从json1的question中提取数字
        num_from_json1 = extract_number_from_question(item1.get("question", ""))
        
        # 2. 处理json2的question，替换所有"0302"为提取的数字
        original_question = item2.get("question", "")
        if num_from_json1:
            # 替换所有4位数字字符串（如0302）
            modified_question = replace_pattern_in_question(
                original_question, 
                r'0302',  # 匹配4位数字
                num_from_json1
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

def merge_json_folders(folder1_path, folder2_path, output_folder_path):
    """处理两个文件夹中的所有JSON文件（保持与merge0.py相同的接口）"""
    # 获取第一个目录中的所有JSON文件
    json1_files = [f for f in os.listdir(folder1_path) if f.endswith('.json')]
    
    # 遍历第一个目录中的每个JSON文件
    for json1_file in json1_files:
        json1_path = os.path.join(folder1_path, json1_file)
        
        # 检查第二个目录中是否存在同名文件
        json2_path = os.path.join(folder2_path, json1_file)
        if not os.path.exists(json2_path):
            print(f"警告: {json2_path} 不存在，跳过处理")
            continue
        
        # 设置输出路径
        output_path = os.path.join(output_folder_path, json1_file)
        
        # 处理这对文件
        try:
            process_single_pair(json1_path, json2_path, output_path)
        except Exception as e:
            print(f"处理文件 {json1_file} 时出错: {str(e)}")
            continue

# 以下部分仅为测试用（实际调用由merge.py完成）
if __name__ == "__main__":
    # 测试调用（与merge.py的调用方式一致）
    merge_json_folders(
        'Transport/gen/40_50_nodes/One_para',
        'zTask_gen/Trans/Ans',
        '.Graph4Real/Trans/Json/Small/Trans'
    )
