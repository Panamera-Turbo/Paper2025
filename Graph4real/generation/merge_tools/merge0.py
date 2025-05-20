from .utils import format_edges, load_json_file, save_json_file
import os

def merge_json_folders(folder1_path, folder2_path, output_folder_path):
    """0 参数任务的合并逻辑"""
    os.makedirs(output_folder_path, exist_ok=True)
    
    for filename in os.listdir(folder1_path):
        if filename.endswith('.json'):
            json1_path = os.path.join(folder1_path, filename)
            json2_path = os.path.join(folder2_path, filename)
            output_path = os.path.join(output_folder_path, filename)
            
            if os.path.exists(json2_path):
                data1 = load_json_file(json1_path)
                data2 = load_json_file(json2_path)
                
                if len(data1) < len(data2):
                    raise ValueError(f"{json1_path} 的长度必须 ≥ {json2_path}")
                
                merged_data = []
                for i, item2 in enumerate(data2):
                    item1 = data1[i]
                    edges_str = format_edges(item1.get("Edges", []))
                    full_edges = f"The edges are: {edges_str}."
                    question = item2.get("question", "")
                    full_ques = f"{question} {full_edges}".strip()
                    
                    merged_item = {
                        "origin_question": item2.get("origin_question", ""),
                        "translated_question": item2.get("question", ""),
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
                
                save_json_file(merged_data, output_path)
                print(f"合并完成: {output_path}")
            else:
                print(f"跳过: {filename} 在 {folder2_path} 中不存在")
