import json

# 输入文件路径
input_file = "zTask_gen/Code/trans_predict/Count_result_oot/Ans/trans_predict_filtered_Llama_hyper.json"
# 输出文件路径
output_file = "zTask_gen/Code/trans_predict/Count_result_oot/Ans/trans_predict_filtered_Llama_hyper_processed.json"

def process_json(input_path, output_path):
    # 读取原始JSON文件
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 处理每个JSON对象
    processed_data = []
    for item in data:
        new_item = {
            "origin_question": item["origin_question"],
            "output": item["output"][0] if item["output"] else None  # 取output的第一个元素
        }
        processed_data.append(new_item)
    
    # 写入新的JSON文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    print(f"处理完成，结果已保存到 {output_path}")

# 执行处理
process_json(input_file, output_file)
