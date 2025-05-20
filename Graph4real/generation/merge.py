import importlib
import json
import os

# 定义任务名称与参数数量的映射
TASK_PARAM_MAP = {
    # 0 参数任务
    "Edge Count": 0,
    "Node Count": 0,
    "Cycle Detection": 0,
    "Triangle Count": 0,
    
    # 1 参数任务
    "Degree Count": 1,
    "Node Existence": 1,
    
    # 2 参数任务
    "Edge Existence": 2,
    "Path Existence": 2,
    "Shortest Path": 2,
}

def get_param_count(task_name):
    """根据任务名称获取参数数量"""
    return TASK_PARAM_MAP.get(task_name, -1)  # 默认返回 -1 表示未知任务

def merge_json_folders(folder1_path, folder2_path, output_folder_path, task_name):
    """根据任务名称选择对应的合并脚本"""
    param_count = get_param_count(task_name)
    if param_count == -1:
        raise ValueError(f"未知任务名称: {task_name}")
    
    # 动态导入对应的合并模块
    merge_module = importlib.import_module(f"merge{param_count}")
    
    # 调用对应的合并函数
    merge_module.merge_json_folders(folder1_path, folder2_path, output_folder_path)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="合并 JSON 文件夹，根据任务名称选择参数数量")
    parser.add_argument("folder1", help="第一个 JSON 文件夹路径")
    parser.add_argument("folder2", help="第二个 JSON 文件夹路径")
    parser.add_argument("output_folder", help="输出文件夹路径")
    parser.add_argument("--task", required=True, help="任务名称（如 'Edge Count', 'Path Existence' 等）")
    
    args = parser.parse_args()
    
    try:
        merge_json_folders(args.folder1, args.folder2, args.output_folder, args.task)
        print("合并完成！")
    except Exception as e:
        print(f"合并失败: {str(e)}")
