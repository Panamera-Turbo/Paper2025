import sys
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QComboBox, QTextEdit
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# 定义主窗口类
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # 设置窗口标题和初始几何大小
        self.setWindowTitle("PEMS Data Visualization")
        self.setGeometry(100, 100, 800, 600)
        # 初始化用户界面
        self.initUI()

    # 初始化用户界面的函数
    def initUI(self):
        # 创建垂直布局
        layout = QVBoxLayout()

        # 创建下拉选择框，用于选择数据集
        self.dataset_combobox = QComboBox()
        self.dataset_combobox.addItems(["PEMS03", "PEMS04", "PEMS07", "PEMS08"])
        self.dataset_combobox.currentIndexChanged.connect(self.load_data)
        layout.addWidget(self.dataset_combobox)

        # 创建一个只读文本编辑框，用于显示摘要信息
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        layout.addWidget(self.summary_text)

        # 创建一个只读文本编辑框，用于显示整体结果
        self.overall_results_text = QTextEdit()
        self.overall_results_text.setReadOnly(True)
        layout.addWidget(self.overall_results_text)

        # 创建画布并设置绘图区
        self.canvas = FigureCanvas(Figure())
        self.ax = self.canvas.figure.subplots()
        layout.addWidget(self.canvas)

        # 设置窗口的中央组件，并应用之前定义的布局
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        # 首次加载数据
        self.load_data()

    # 加载数据并更新UI的函数
    def load_data(self):
        # 获取当前选择的数据集
        dataset = self.dataset_combobox.currentText()
        filepath = f"{dataset}.txt"

        # 打开文件并读取内容
        with open(filepath, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        # 将文件头部信息设置到摘要信息框中
        summary_info = "".join(lines[:5])
        self.summary_text.setText(summary_info)

        # 处理步数数据，每行都应有四个数据点
        data = {"Step": [], "MAE": [], "RMSE": [], "MAPE": []}
        for line in lines[10:22]:
            parts = line.strip().split()
            if len(parts) == 4:
                data["Step"].append(int(parts[0]))
                data["MAE"].append(float(parts[1]))
                data["RMSE"].append(float(parts[2]))
                data["MAPE"].append(float(parts[3]))

        # 确保数据不为空
        if data["Step"]:
            # 转换为DataFrame并绘制图形
            steps_df = pd.DataFrame(data)
            overall_results = "".join(lines[24:27])
            self.overall_results_text.setText(overall_results)

            # 清除图形，为每个指标绘制线，并添加数据点注释
            self.ax.clear()
            for metric in ["MAE", "RMSE", "MAPE"]:
                self.ax.plot(steps_df["Step"], steps_df[metric], label=metric, marker='o')
                for x, y in zip(steps_df["Step"], steps_df[metric]):
                    self.ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0,10), ha='center')

            # 设置横坐标标签
            self.ax.set_xticks(steps_df["Step"])
            self.ax.set_title("Multi-Step Prediction Results")
            self.ax.set_xlabel("Step")
            self.ax.set_ylabel("Value")
            self.ax.legend()
            self.canvas.draw()

# 主程序入口
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWindow()
    ex.show()
    sys.exit(app.exec_())
