import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.metrics import accuracy_score

# 加载数据
data = torch.load('Cora/cora/cora_fixed_tfidf.pt')
# 确保数据是PyG的Data对象格式
if not isinstance(data, Data):
    data = Data(x=data.x, edge_index=data.edge_index, y=data.y)

# 检查是否有GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

# 划分训练集、验证集和测试集 (这里简单按比例划分，实际应该使用固定划分)
num_nodes = data.x.size(0)
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
val_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)

# 随机划分：60%训练，20%验证，20%测试
perm = torch.randperm(num_nodes)
train_mask[perm[:int(0.6*num_nodes)]] = True
val_mask[perm[int(0.6*num_nodes):int(0.8*num_nodes)]] = True
test_mask[perm[int(0.8*num_nodes):]] = True

# GCN模型定义
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# 模型参数
input_dim = data.x.size(1)  # 特征维度
hidden_dim = 64
output_dim = data.y.max().item() + 1  # 类别数

model = GCN(input_dim, hidden_dim, output_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = nn.NLLLoss()

# 训练函数
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# 测试函数
def test(mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        acc = accuracy_score(data.y[mask].cpu(), pred[mask].cpu())
    return acc

# 训练循环
best_val_acc = 0
for epoch in range(200):
    loss = train()
    train_acc = test(train_mask)
    val_acc = test(val_mask)
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pt')
    
    if epoch % 10 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
              f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

# 最终测试
model.load_state_dict(torch.load('best_model.pt'))
test_acc = test(test_mask)
print(f'Final Test Accuracy: {test_acc:.4f}')
