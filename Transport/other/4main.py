import sys
import subprocess
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIcon, QPixmap

class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PEMS Data Visualization Main Menu")
        self.setGeometry(100, 100, 300, 200)
        self.setWindowIcon(QIcon('your-icon.png'))  # 设置窗口图标，替换为你的图标路径
        self.initUI()

    def initUI(self):
        self.setStyleSheet("QPushButton { font-size: 18px; }")  # 设置全局样式

        layout = QVBoxLayout()

        # 添加一个标题标签
        label_title = QLabel("PEMS Data Visualization")
        label_title.setAlignment(Qt.AlignCenter)
        label_title.setFont(QFont('Arial', 22))
        layout.addWidget(label_title)

        # 创建并添加按钮，附加样式
        btn_npzauto = QPushButton("Traffic Flow")
        btn_npzauto.setIcon(QIcon('traffic-icon.png'))  # 设置按钮图标，替换为你的图标路径
        btn_npzauto.clicked.connect(lambda: self.run_script('1npzauto.py'))
        layout.addWidget(btn_npzauto)

        btn_csvauto = QPushButton("Node Graph")
        btn_csvauto.setIcon(QIcon('node-icon.png'))  # 设置按钮图标，替换为你的图标路径
        btn_csvauto.clicked.connect(lambda: self.run_script('2nodeauto.py'))
        layout.addWidget(btn_csvauto)

        btn_txtauto = QPushButton("Training Results")
        btn_txtauto.setIcon(QIcon('training-icon.png'))  # 设置按钮图标，替换为你的图标路径
        btn_txtauto.clicked.connect(lambda: self.run_script('3resultauto.py'))
        layout.addWidget(btn_txtauto)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def run_script(self, script_name):
        subprocess.Popen(['python', script_name], creationflags=subprocess.CREATE_NEW_CONSOLE)

# 主程序入口
if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainApp()
    main_window.show()
    sys.exit(app.exec_())
