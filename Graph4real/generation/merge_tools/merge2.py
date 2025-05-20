import json
import re
import os
from pathlib import Path

def format_edges(edges):
    """格式化边列表为字符串"""
    edge_strs = [f"{edge}" for edge in edges]
    return ", ".join(edge_strs)

def extract_numbers_from_question(question):
    """从问题中提取两个数字（保持与merge1.py相似的函数名但不同实现）"""
    numbers = re.findall(r'\d+', question)
    return numbers[:2] if len(numbers) >= 2 else (None, None)

def replace_patterns(question, replacements):
    """替换问题中的多个模式（通用化实现）"""
    for old, new in replacements.items():
        question = question.replace(old, new)
    return question

def process_single_pair(json1_path, json2_path, output_path):
    """处理单个JSON文件对（保持相同函数签名）"""
    with open(json1_path, 'r', encoding='utf-8') as f1:
        data1 = json.load(f1)
    with open(json2_path, 'r', encoding='utf-8') as f2:
        data2 = json.load(f2)

    if len(data1) < len(data2):
        raise ValueError(f"{json1_path} 必须比 {json2_path} 长")

    merged_data = []
    for i, item2 in enumerate(data2):
        item1 = data1[i]
        
        # 提取两个参数（区别于merge1.py的单参数提取）
        param1, param2 = extract_numbers_from_question(item1.get("question", ""))
        
        # 双参数替换逻辑
        original_question = item2.get("question", "")
        if param1 and param2:
            modified_question = replace_patterns(
                original_question,
                {"0302": param1, "0714": param2}  # 同时替换两个占位符
            )
        else:
            modified_question = original_question

        # 统一拼接edges信息
        edges_str = format_edges(item1.get("Edges", []))
        full_edges = f"The edges are: {edges_str}."
        full_ques = f"{modified_question} {full_edges}".strip()

        # 保持完全相同的输出结构
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

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)
    print(f"Saved to {output_path}")

def merge_json_folders(folder1_path, folder2_path, output_folder_path):
    """统一入口函数（与merge0/1.py完全一致的签名）"""
    os.makedirs(output_folder_path, exist_ok=True)
    for filename in os.listdir(folder1_path):
        if filename.endswith('.json'):
            json1_path = os.path.join(folder1_path, filename)
            json2_path = os.path.join(folder2_path, filename)
            output_path = os.path.join(output_folder_path, filename)
            
            if os.path.exists(json2_path):
                try:
                    process_single_pair(json1_path, json2_path, output_path)
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
            else:
                print(f"Skipping {filename} (not found in {folder2_path})")

# 测试代码（实际使用时由merge.py调用）
if __name__ == "__main__":
    merge_json_folders(
        "/path/to/input1",
        "/path/to/input2",
        "/path/to/output"
    )
