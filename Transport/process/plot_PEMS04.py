import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# 读取CSV文件
df = pd.read_csv('./PEMS04.csv')

# 创建一个无向图
G = nx.Graph()

# 添加边到图中
for index, row in df.iterrows():
    G.add_edge(row['from'], row['to'], weight=row['cost'])

# 绘制图
pos = nx.spring_layout(G)  # 使用spring布局
nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', font_size=10, font_color='black', font_weight='bold')
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

# 显示图
plt.title('Graph Visualization')
plt.show()
