# -*- coding: utf-8 -*-
# @Author: Brandon Han
# @Date:   2019-09-06 16:48:57
# @Last Modified by:   Brandon Han
# @Last Modified time: 2019-09-06 17:16:03

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import *
import matplotlib.pyplot as plt
import sys
import numpy as np


def gauss3(x, a1, a2, a3, m1, m2, m3, s1, s2, s3):

    return a1 * np.exp(-((x - m1) / s1)**2) + a2 * np.exp(-((x - m2) / s2)**2) + \
        a3 * np.exp(-((x - m3) / s3)**2)


class App(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(App, self).__init__(parent)

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Spectrum DIY')
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.button_plot = QtWidgets.QPushButton("Plot")
        self.a1 = QLineEdit()
        self.a1.setPlaceholderText('a1')
        self.a2 = QLineEdit()
        self.a2.setPlaceholderText('a2')
        self.a3 = QLineEdit()
        self.a3.setPlaceholderText('a3')
        self.m1 = QLineEdit()
        self.m1.setPlaceholderText('m1')
        self.m2 = QLineEdit()
        self.m2.setPlaceholderText('m2')
        self.m3 = QLineEdit()
        self.m3.setPlaceholderText('m3')
        self.s1 = QLineEdit()
        self.s1.setPlaceholderText('s1')
        self.s2 = QLineEdit()
        self.s2.setPlaceholderText('s2')
        self.s3 = QLineEdit()
        self.s3.setPlaceholderText('s3')
        self.button_plot.clicked.connect(self.plot_)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.a1)
        layout.addWidget(self.a2)
        layout.addWidget(self.a3)
        layout.addWidget(self.m1)
        layout.addWidget(self.m2)
        layout.addWidget(self.m3)
        layout.addWidget(self.s1)
        layout.addWidget(self.s2)
        layout.addWidget(self.s3)
        layout.addWidget(self.button_plot)
        self.setLayout(layout)

    def plot_(self):
        ax = self.figure.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.clear()
        x = np.linspace(400, 680, 1000)
        a1 = eval(self.a1.text())
        a2 = eval(self.a2.text())
        a3 = eval(self.a3.text())
        m1 = eval(self.m1.text())
        m2 = eval(self.m2.text())
        m3 = eval(self.m3.text())
        s1 = eval(self.s1.text())
        s2 = eval(self.s2.text())
        s3 = eval(self.s3.text())
        ax.plot(x, gauss3(x, a1, a2, a3, m1, m2, m3, s1, s2, s3))
        self.canvas.draw()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main_window = App()
    main_window.show()
    app.exec()
