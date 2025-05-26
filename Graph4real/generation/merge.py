import importlib
import json
import os

TASK_PARAM_MAP = {
    # 0-parameter tasks
    "Edge Count": 0,
    "Node Count": 0,
    "Cycle Detection": 0,
    "Triangle Count": 0,
    
    # 1-parameter tasks
    "Degree Count": 1,
    "Node Existence": 1,
    
    # 2-parameter tasks
    "Edge Existence": 2,
    "Path Existence": 2,
    "Shortest Path": 2,
}

def get_param_count(task_name):
    return TASK_PARAM_MAP.get(task_name, -1)  # Default -1 for unknown tasks

def merge_json_folders(folder1_path, folder2_path, output_folder_path, task_name):
    param_count = get_param_count(task_name)
    if param_count == -1:
        raise ValueError(f"Unknown task name: {task_name}")
    merge_module = importlib.import_module(f"merge{param_count}")
    
    merge_module.merge_json_folders(folder1_path, folder2_path, output_folder_path)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Merge JSON folders based on task name parameter count")
    parser.add_argument("folder1", help="Path to first JSON folder")
    parser.add_argument("folder2", help="Path to second JSON folder")
    parser.add_argument("output_folder", help="Path to output folder")
    parser.add_argument("--task", required=True, help="Task name (e.g., 'Edge Count', 'Path Existence')")
    
    args = parser.parse_args()
    
    try:
        merge_json_folders(args.folder1, args.folder2, args.output_folder, args.task)
        print("Merge completed!")
    except Exception as e:
        print(f"Merge failed: {str(e)}")
