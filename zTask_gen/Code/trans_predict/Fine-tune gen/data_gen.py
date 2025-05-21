import json
import os

sys_prompt = '''
        Assumption: The undirected graph *G* is already structured in NetworkX format, represented by `data`. Please write a Python function to solve the problem above. Your output should strictly follow the given format:  

        ```python  
        def method(data):  
            ...  
            return  
        ```  

        Additionally, provide a single line of code that calls this function. The return value must be exactly True or False. Only provide the function call in the specified format—I will automatically retrieve the return value:  

        ```python  
        method(data)  
        ```  

        Note: Strictly adhere to the specified format.
        '''

def process_json_file(file_path):
    """处理单个JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    new_data = []
    
    # 确保data是列表形式，如果不是则转换为列表
    if not isinstance(data, list):
        data = [data]
    
    for item in data:
        output_list = item.get("output", [])
        code_gen_list = item.get("code_gen", [])
        origin_question = item.get("translated_answer", "")
        
        for i, output in enumerate(output_list):
            if output.startswith("Execution Result:"):
                try:
                    value_str = output.split("Execution Result:")[1].strip()
                    value = float(value_str)
                    
                    if 280 <= value <= 290 and i < len(code_gen_list):
                        new_item = {
                            "instruction": origin_question + '\n' + sys_prompt,
                            "input": "",
                            "output": code_gen_list[i]
                        }
                        new_data.append(new_item)
                except (ValueError, IndexError):
                    continue
    return new_data

def process_json_folder(input_folder, output_file):
    """处理文件夹中的所有JSON文件"""
    all_new_data = []
    
    # 遍历文件夹中的所有JSON文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            file_path = os.path.join(input_folder, filename)
            try:
                new_data = process_json_file(file_path)
                all_new_data.extend(new_data)
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
                continue
    
    # 写入新的JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_new_data, f, ensure_ascii=False, indent=2)

# 使用示例
input_folder = 'zTask_gen/Trans/Code/Ans'  # 包含多个JSON文件的文件夹路径
output_file = 'zTask_gen/Code/trans_predict/Fine-tune gen/output_Flow.json'  # 输出文件路径
process_json_folder(input_folder, output_file)
