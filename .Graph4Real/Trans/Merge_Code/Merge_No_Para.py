# import json

# def format_edges(edges):
#     edge_strs = []
#     for edge in edges:
#         edge_strs.append(f"{edge}")
#     return ", ".join(edge_strs)

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
        
#         # 处理 Edges，拼接成字符串
#         edges_str = format_edges(item1.get("Edges", []))
#         full_edges = f"The edges are: {edges_str}."
        
#         # 拼接 full_ques = "The edges are: ..." + question
#         question = item2.get("question", "")
#         full_ques = f"{question} {full_edges}".strip()
        
#         # 创建新的合并后的对象
#         merged_item = {
#             "origin_question": item2.get("origin_question", ""),
#             "translated_question": item2.get("question", ""),
#             "question": full_ques,  # 新增字段
#             "type": item2.get("label", ""),
#             "background_type": item2.get("type", ""),
#             "Node_List": item1.get("Node_List", []),
#             "Edges": item1.get("Edges", []),  # 仍然保留原始 Edges
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
# merge_jsons('/home/data2t1/wangrongzheng/GTAgent/Transport/gen/1000_50_nodes/number_of_nodes_output.json', 
#             '/home/data2t1/wangrongzheng/GTAgent/zTask_gen/Trans/Ans/Node_Count_filtered.json', 
#             '/home/data2t1/wangrongzheng/GTAgent/.Graph4Real/Trans/Node_Count.json')

# # '/home/data2t1/wangrongzheng/GTAgent/.Graph4Real/Trans/Cycle_Detection.json'


import json
import os

def format_edges(edges):
    edge_strs = []
    for edge in edges:
        edge_strs.append(f"{edge}")
    return ", ".join(edge_strs)

def merge_json_pair(json1_path, json2_path, output_path):
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
        
        # 处理 Edges，拼接成字符串
        edges_str = format_edges(item1.get("Edges", []))
        full_edges = f"The edges are: {edges_str}."
        
        # 拼接 full_ques = "The edges are: ..." + question
        question = item2.get("question", "")
        full_ques = f"{question} {full_edges}".strip()
        
        # 创建新的合并后的对象
        merged_item = {
            "origin_question": item2.get("origin_question", ""),
            "translated_question": item2.get("question", ""),
            "question": full_ques,  # 新增字段
            "type": item2.get("label", ""),
            "background_type": item2.get("type", ""),
            "Node_List": item1.get("Node_List", []),
            "Edges": item1.get("Edges", []),  # 仍然保留原始 Edges
            "Edge_Count": item1.get("Edge_Count", 0),
            "Description": item1.get("Description", ""),
            "answer": item1.get("answer", ""),
        }
        
        merged_data.append(merged_item)
    
    # 将合并后的数据写入新文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)
    
    print(f"合并完成，结果已保存到 {output_path}")

def merge_json_folders(folder1_path, folder2_path, output_folder_path):
    # 确保输出文件夹存在
    os.makedirs(output_folder_path, exist_ok=True)
    
    # 遍历第一个文件夹中的所有json文件
    for filename in os.listdir(folder1_path):
        if filename.endswith('.json'):
            json1_path = os.path.join(folder1_path, filename)
            json2_path = os.path.join(folder2_path, filename)
            output_path = os.path.join(output_folder_path, filename)
            
            # 检查第二个文件夹中是否有同名文件
            if os.path.exists(json2_path):
                try:
                    merge_json_pair(json1_path, json2_path, output_path)
                except Exception as e:
                    print(f"处理文件 {filename} 时出错: {str(e)}")
            else:
                print(f"警告: {folder2_path} 中没有找到 {filename}，跳过处理")

# 使用示例
merge_json_folders(
    'Web/Dataset/40_50_nodes/No_para',
    'zTask_gen/Web/Ans',
    '.Graph4Real/Trans/Json/Small/Web'
)



# merge_json_folders(
#     'Transport/gen/40_50_nodes/No_para',
#     'zTask_gen/Trans/Ans',
#     '.Graph4Real/Trans/Json/Small/Trans'
# )