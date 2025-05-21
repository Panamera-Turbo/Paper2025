import pandas as pd

# 读取CSV文件
file_path = "/home/data2t1/tempuser/GTAgent/zTrain_test/transport/PEMS03.csv"
df = pd.read_csv(file_path)

# 获取所有节点
nodes = set(df['from']).union(set(df['to']))

# 输出节点数量
print("图中节点数量为：", len(nodes))