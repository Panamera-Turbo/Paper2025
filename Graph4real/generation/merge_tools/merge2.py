import json
import re
import os
from pathlib import Path

def format_edges(edges):
    edge_strs = [f"{edge}" for edge in edges]
    return ", ".join(edge_strs)

def extract_numbers_from_question(question):
    numbers = re.findall(r'\d+', question)
    return numbers[:2] if len(numbers) >= 2 else (None, None)

def replace_patterns(question, replacements):
    for old, new in replacements.items():
        question = question.replace(old, new)
    return question

def process_single_pair(json1_path, json2_path, output_path):
    with open(json1_path, 'r', encoding='utf-8') as f1:
        data1 = json.load(f1)
    with open(json2_path, 'r', encoding='utf-8') as f2:
        data2 = json.load(f2)

    if len(data1) < len(data2):
        raise ValueError(f"{json1_path} must be longer than {json2_path}")

    merged_data = []
    for i, item2 in enumerate(data2):
        item1 = data1[i]
        
        param1, param2 = extract_numbers_from_question(item1.get("question", ""))
        
        original_question = item2.get("question", "")
        if param1 and param2:
            modified_question = replace_patterns(
                original_question,
                {"0302": param1, "0714": param2}
            )
        else:
            modified_question = original_question

        edges_str = format_edges(item1.get("Edges", []))
        full_edges = f"The edges are: {edges_str}."
        full_ques = f"{modified_question} {full_edges}".strip()

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

if __name__ == "__main__":
    merge_json_folders(
        "/path/to/input1",
        "/path/to/input2",
        "/path/to/output"
    )
