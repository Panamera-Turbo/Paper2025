import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

# 加载本地 Cora 数据集
data_path = '/home/data2t1/wangrongzheng/GTAgent/zTrain_test/cora/cora_fixed_tfidf.pt'
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

# 定义 GAT 模型
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8):
        super(GAT, self).__init__()
        # 第一层 GAT
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.6)
        # 第二层 GAT
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        # 第一层 GAT + ReLU 激活
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        # 第二层 GAT + Log Softmax 输出
        x = self.gat2(x, edge_index)
        return F.log_softmax(x, dim=1)

# 模型实例化
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GAT(data.num_features, 8, data.y.max().item() + 1, heads=8).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

# 训练模型
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# 测试模型
def test():
    model.eval()
    out = model(data.x, data.edge_index)
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
