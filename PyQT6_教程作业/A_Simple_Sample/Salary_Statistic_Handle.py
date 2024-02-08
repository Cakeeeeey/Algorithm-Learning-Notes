from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6 import uic

class Stats:

    def __init__(self):
        # 从文件中加载UI定义
        self.ui = uic.loadUi("UI/Salary_Statistics.ui")
        self.ui.Button.clicked.connect(self.handleCalc)

    def handleCalc(self):
        info = self.ui.TextEdit.toPlainText()

        salary_above_20k = ''
        salary_below_20k = ''
        for line in info.splitlines():
            if not line.strip():
                continue
            parts = line.split(' ')

            parts = [p for p in parts if p]
            name,salary,age = parts
            if int(salary) >= 20000:
                salary_above_20k += name + '\n'
            else:
                salary_below_20k += name + '\n'

        QMessageBox.about(self.ui,
                    '统计结果',
                    f'''薪资20000 以上的有：\n{salary_above_20k}
                    \n薪资20000 以下的有：\n{salary_below_20k}'''
                    )

app = QApplication([])
stats = Stats()
stats.ui.show()
app.exec()