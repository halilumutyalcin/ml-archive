from PyQt5 import uic

with open("chatbot.py","w",encoding="utf-8") as fout:
    uic.compileUi("form.ui",fout)

# self.listWidget.currentItem().setBackground(QColor("#000000"))
