import json

# 任务参数映射（唯一定义处）
TASK_PARAM_MAP = {
    # 0 参数任务
    "Edge Count": 0,
    "Node Count": 0,
    "Cycle Detection": 0,
    # 1 参数任务
    "Degree Count": 1,
    "Node Existence": 1,
    "Triangle Count": 1,
    # 2 参数任务
    "Edge Existence": 2,
    "Path Existence": 2,
    "Shortest Path": 2,
}

def get_param_count(task_name):
    """根据任务名称获取参数数量（唯一实现）"""
    return TASK_PARAM_MAP.get(task_name, -1)

def format_edges(edges):
    """格式化边列表为字符串"""
    return ", ".join([f"{edge}" for edge in edges])

def load_json_file(path):
    """加载 JSON 文件"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json_file(data, path):
    """保存 JSON 文件"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
