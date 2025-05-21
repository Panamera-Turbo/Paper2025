csv文件对应于对应数据集的节点连接情况
如果要运行可视化程序，请解压本项目，然后在vscode中打开本项目，运行main.py文件，按照要求输入英文输入法下的数字和字符即可

###  PEMS0X数据集介绍

**来源**：[Learning dynamics and heterogeneity of spatial-temporal graph data for traffic forecasting, TKDE'21](https://ieeexplore.ieee.org/abstract/document/9346058)。[数据链接](https://github.com/guoshnBJTU/ASTGNN/tree/main/data)。

**描述**：PEMS0X是一系列交通流量数据集，包括PEMS03、PEMS04、PEMS07和PEMS08。X代表数据收集所在地区的代码。交通信息每5分钟记录一次。与METR-LA和PEMS-BAY类似，PEMS0X也包括一个传感器图，表示传感器之间的依赖关系。邻接矩阵的计算细节可以在[ASTGCN](https://ojs.aaai.org/index.php/AAAI/article/view/3881/3759)中找到。

**时间段**：

- PEMS03: 2018/09/01 -> 2018/11/30
- PEMS04: 2018/01/01 -> 2018/2/28
- PEMS07: 2017/05/01 -> 2017/08/31
- PEMS08: 2016/07/01 -> 2016/08/31

**时间步数**：

- PEMS03: 26208
- PEMS04: 16992
- PEMS07: 28224
- PEMS08: 17856

**数据集分割**：6:2:2。

**变量**：每个变量代表一个传感器的交通流量。

**变量数目**：

- PEMS03: 358
- PEMS04: 307
- PEMS07: 883
- PEMS08: 170

**典型设置**：

- 多变量时间序列预测。输入所有特征，预测所有特征。