import re
import subprocess
import tempfile
import sys
import os

def execute_combined_code(code_part1, llm_output, conda_env_name="GLM"):
    # 代码块提取（使用正则表达式匹配三个反引号包裹的Python代码）
    code_blocks = re.findall(r'```python(.*?)```', llm_output, re.DOTALL)
    code_part2 = code_blocks[0].strip() if len(code_blocks) > 0 else ""
    code_part3 = code_blocks[1].strip() if len(code_blocks) > 1 else ""

    # 构建完整代码（添加必要依赖）
    full_code = f"""
import sys
{code_part1}

# 补充可能缺失的依赖
try:
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv
    from torch_geometric.data import Data
except ImportError as e:
    print("Missing dependencies:", e)
    sys.exit(1)

{code_part2}

# 执行段三的调用
if __name__ == "__main__":
    try:
        result = {code_part3.split('(')[0].strip()}(data)
        print("\\nExecution Result:", result)
    except Exception as e:
        print("Execution Error:", str(e))
"""

# 创建临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(full_code)
        tmp_path = f.name

    # 构造conda执行命令（完整路径初始化）
    conda_sh_path = "/home/data2t1/wangrongzheng/miniconda3/etc/profile.d/conda.sh"
    
    if sys.platform.startswith('win'):
        python_command = fr"conda activate {conda_env_name} && python"
    else:
        # 显式加载conda初始化脚本
        python_command = (
            f"bash -c '"
            f"source {conda_sh_path} && "
            f"conda activate {conda_env_name} && "
            f"python {tmp_path}'"
        )

    # 执行脚本
    try:
        result = subprocess.run(
            python_command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        output = result.stdout
    except subprocess.CalledProcessError as e:
        output = f"ERROR: {e.stderr}"
    except Exception as e:
        output = f"UNEXPECTED ERROR: {str(e)}"

    # 清理临时文件
    os.unlink(tmp_path)
    
    return output

# 使用示例（需替换实际参数）
part1 = """import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.metrics import accuracy_score

# 加载数据
data = torch.load('Cora/cora/cora_fixed_tfidf.pt')
# 确保数据是PyG的Data对象格式
if not isinstance(data, Data):
    data = Data(x=data.x, edge_index=data.edge_index, y=data.y)"""

llm_output = """以下是一个基于PyTorch Geometric (PyG) 实现的GCN模型函数，用于预测节点0302的研究领域类别编号：

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

def method(data):
    # 划分训练集和测试集
    train_idx, test_idx = train_test_split(
        torch.arange(data.num_nodes), test_size=0.2, random_state=42)
    
    # 初始化模型
    model = GCN(num_features=data.x.size(1), num_classes=data.y.max().item()+1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    # 训练模型
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[train_idx], data.y[train_idx])
        loss.backward()
        optimizer.step()
    
    # 预测节点0302的类别
    model.eval()
    with torch.no_grad():
        pred = model(data)[302].argmax().item()  # 假设节点0302的索引就是302
    
    return pred
```

调用函数的代码：

```python
method(data)
```

注意：
1. 这个实现假设节点0302的索引就是302，如果不是，请调整索引值
2. 模型使用了简单的两层GCN结构，可以根据需要调整网络深度和隐藏层维度
3. 训练过程中使用了20%的数据作为测试集，其余作为训练集
4. 返回的是节点0302的预测类别编号
"""

print(execute_combined_code(part1, llm_output, "GLM"))