import torch
import torch.nn.functional as F
from torch.nn import Linear

# 加载本地 Cora 数据集
data_path = '/home/data2t1/tempuser/GTAgent/zTrain_test/cora/cora_fixed_tfidf.pt'
data = torch.load(data_path)

# 检查数据是否包含 train_mask，如果没有则手动创建
if not hasattr(data, 'train_mask'):
    print("train_mask not found, creating masks...")
    num_nodes = data.num_nodes
    train_ratio, val_ratio, test_ratio = 0.6, 0.2, 0.2
    indices = torch.randperm(num_nodes)
    train_size = int(train_ratio * num_nodes)
    val_size = int(val_ratio * num_nodes)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    data.train_mask[train_indices] = True
    data.val_mask[val_indices] = True
    data.test_mask[test_indices] = True

print(f'Training nodes: {data.train_mask.sum()}')
print(f'Validation nodes: {data.val_mask.sum()}')
print(f'Test nodes: {data.test_mask.sum()}')

# 定义 MLP 模型
class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(MLP, self).__init__()
        # 第一层线性变换
        self.lin1 = Linear(in_channels, hidden_channels)
        # 第二层线性变换
        self.lin2 = Linear(hidden_channels, out_channels)

    def forward(self, x):
        # 第一层 + ReLU 激活
        x = self.lin1(x)
        x = F.relu(x)
        # 第二层 + Log Softmax 输出
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)

# 模型实例化
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLP(data.num_features, 64, data.y.max().item() + 1).to(device)  # 隐藏层维度设置为 64
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# 训练模型
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x)  # MLP 不需要 edge_index
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# 测试模型
def test():
    model.eval()
    out = model(data.x)
    pred = out.argmax(dim=1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = pred[mask].eq(data.y[mask]).sum().item()
        accs.append(correct / mask.sum().item())
    return accs

# 训练和测试
for epoch in range(200):
    loss = train()
    train_acc, val_acc, test_acc = test()
    if epoch % 10 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')

# 最终测试结果
train_acc, val_acc, test_acc = test()
print(f'Final Test Accuracy: {test_acc:.4f}')
