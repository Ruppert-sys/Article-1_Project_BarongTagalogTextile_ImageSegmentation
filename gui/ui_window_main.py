# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui/ui_window_main.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(480, 320)
        MainWindow.setMinimumSize(QtCore.QSize(480, 320))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.stackedWidget = QtWidgets.QStackedWidget(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.stackedWidget.setFont(font)
        self.stackedWidget.setObjectName("stackedWidget")
        self.page_1 = QtWidgets.QWidget()
        self.page_1.setStyleSheet("#page_1 {\n"
"    border-image: url(:/bg/gui/assets/barong_bg2.png);\n"
"}")
        self.page_1.setObjectName("page_1")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.page_1)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.widget = QtWidgets.QWidget(self.page_1)
        self.widget.setObjectName("widget")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setSpacing(0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.widget_3 = QtWidgets.QWidget(self.widget)
        self.widget_3.setObjectName("widget_3")
        self.verticalLayout_3.addWidget(self.widget_3)
        self.widget_2 = QtWidgets.QWidget(self.widget)
        self.widget_2.setStyleSheet("background-color: rgba(0, 0, 0, 100);")
        self.widget_2.setObjectName("widget_2")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.widget_2)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_4.setSpacing(0)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.lbl_begin = QtWidgets.QLabel(self.widget_2)
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.lbl_begin.setFont(font)
        self.lbl_begin.setStyleSheet("color: rgb(255, 255, 255);\n"
"background-color: transparent;")
        self.lbl_begin.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_begin.setObjectName("lbl_begin")
        self.verticalLayout_4.addWidget(self.lbl_begin)
        self.verticalLayout_3.addWidget(self.widget_2)
        self.verticalLayout_3.setStretch(0, 80)
        self.verticalLayout_3.setStretch(1, 20)
        self.verticalLayout_2.addWidget(self.widget)
        self.stackedWidget.addWidget(self.page_1)
        self.page_2 = QtWidgets.QWidget()
        self.page_2.setStyleSheet("background-color: rgb(0, 37, 68);")
        self.page_2.setObjectName("page_2")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.page_2)
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_5.setSpacing(0)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.widget_4 = QtWidgets.QWidget(self.page_2)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.widget_4.setFont(font)
        self.widget_4.setObjectName("widget_4")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.widget_4)
        self.verticalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_6.setSpacing(0)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.widget_6 = QtWidgets.QWidget(self.widget_4)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.widget_6.setFont(font)
        self.widget_6.setObjectName("widget_6")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.widget_6)
        self.horizontalLayout_4.setContentsMargins(5, 5, 5, 5)
        self.horizontalLayout_4.setSpacing(5)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.verticalLayout_11 = QtWidgets.QVBoxLayout()
        self.verticalLayout_11.setSpacing(5)
        self.verticalLayout_11.setObjectName("verticalLayout_11")
        self.label = QtWidgets.QLabel(self.widget_6)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setStyleSheet("color: rgb(255, 255, 255);\n"
"background-color: transparent;")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.verticalLayout_11.addWidget(self.label)
        self.label_2 = QtWidgets.QLabel(self.widget_6)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setStyleSheet("color: rgb(255, 255, 255);\n"
"background-color: transparent;")
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_11.addWidget(self.label_2)
        self.verticalLayout_11.setStretch(0, 50)
        self.verticalLayout_11.setStretch(1, 50)
        self.horizontalLayout_4.addLayout(self.verticalLayout_11)
        self.verticalLayout_10 = QtWidgets.QVBoxLayout()
        self.verticalLayout_10.setSpacing(5)
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.lbl_filename = QtWidgets.QLabel(self.widget_6)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lbl_filename.sizePolicy().hasHeightForWidth())
        self.lbl_filename.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.lbl_filename.setFont(font)
        self.lbl_filename.setStyleSheet("color: rgb(255, 255, 255);\n"
"background-color: transparent;\n"
"padding: 5px;\n"
"border-color: rgb(255, 255, 255);\n"
"border-style: outset;\n"
"border-width: 2px;")
        self.lbl_filename.setText("")
        self.lbl_filename.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_filename.setObjectName("lbl_filename")
        self.verticalLayout_10.addWidget(self.lbl_filename)
        self.lbl_prediction = QtWidgets.QLabel(self.widget_6)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lbl_prediction.sizePolicy().hasHeightForWidth())
        self.lbl_prediction.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.lbl_prediction.setFont(font)
        self.lbl_prediction.setStyleSheet("color: rgb(255, 255, 255);\n"
"background-color: transparent;\n"
"padding: 5px;\n"
"border-color: rgb(255, 255, 255);\n"
"border-style: outset;\n"
"border-width: 2px;")
        self.lbl_prediction.setText("")
        self.lbl_prediction.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_prediction.setObjectName("lbl_prediction")
        self.verticalLayout_10.addWidget(self.lbl_prediction)
        self.verticalLayout_10.setStretch(0, 50)
        self.verticalLayout_10.setStretch(1, 50)
        self.horizontalLayout_4.addLayout(self.verticalLayout_10)
        self.verticalLayout_12 = QtWidgets.QVBoxLayout()
        self.verticalLayout_12.setSpacing(5)
        self.verticalLayout_12.setObjectName("verticalLayout_12")
        self.label_7 = QtWidgets.QLabel(self.widget_6)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_7.sizePolicy().hasHeightForWidth())
        self.label_7.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.label_7.setFont(font)
        self.label_7.setStyleSheet("color: rgb(255, 255, 255);\n"
"padding: 5px;\n"
"border-color: rgb(0, 37, 68);\n"
"border-width: 2px;")
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.verticalLayout_12.addWidget(self.label_7)
        self.lbl_accuracy = QtWidgets.QLabel(self.widget_6)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lbl_accuracy.sizePolicy().hasHeightForWidth())
        self.lbl_accuracy.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.lbl_accuracy.setFont(font)
        self.lbl_accuracy.setStyleSheet("color: rgb(255, 255, 255);\n"
"background-color: transparent;\n"
"padding: 5px;\n"
"border-color: rgb(255, 255, 255);\n"
"border-style: outset;\n"
"border-width: 2px;")
        self.lbl_accuracy.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_accuracy.setObjectName("lbl_accuracy")
        self.verticalLayout_12.addWidget(self.lbl_accuracy)
        self.verticalLayout_12.setStretch(0, 50)
        self.verticalLayout_12.setStretch(1, 50)
        self.horizontalLayout_4.addLayout(self.verticalLayout_12)
        self.verticalLayout_8 = QtWidgets.QVBoxLayout()
        self.verticalLayout_8.setSpacing(5)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setSpacing(5)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.btn_capture = QtWidgets.QPushButton(self.widget_6)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btn_capture.sizePolicy().hasHeightForWidth())
        self.btn_capture.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.btn_capture.setFont(font)
        self.btn_capture.setStyleSheet("QPushButton {\n"
"    padding: 5px;\n"
"    border-color: black;\n"
"    border-style: outset;\n"
"    border-width: 0px;\n"
"    background-color: rgb(115, 115, 115);\n"
"    color: rgb(255, 255, 255);\n"
"}\n"
"\n"
"QPushButton:hover {    \n"
"    background-color: rgb(211, 215, 207);\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"    background-color: blue;\n"
"}")
        self.btn_capture.setObjectName("btn_capture")
        self.horizontalLayout_2.addWidget(self.btn_capture)
        self.btn_load = QtWidgets.QPushButton(self.widget_6)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btn_load.sizePolicy().hasHeightForWidth())
        self.btn_load.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.btn_load.setFont(font)
        self.btn_load.setStyleSheet("QPushButton {\n"
"    padding: 5px;\n"
"    border-color: black;\n"
"    border-style: outset;\n"
"    border-width: 0px;\n"
"    background-color: rgb(115, 115, 115);\n"
"    color: rgb(255, 255, 255);\n"
"}\n"
"\n"
"QPushButton:hover {    \n"
"    background-color: rgb(211, 215, 207);\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"    background-color: blue;\n"
"}")
        self.btn_load.setObjectName("btn_load")
        self.horizontalLayout_2.addWidget(self.btn_load)
        self.horizontalLayout_2.setStretch(0, 50)
        self.horizontalLayout_2.setStretch(1, 50)
        self.verticalLayout_8.addLayout(self.horizontalLayout_2)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSpacing(5)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.btn_predict = QtWidgets.QPushButton(self.widget_6)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btn_predict.sizePolicy().hasHeightForWidth())
        self.btn_predict.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.btn_predict.setFont(font)
        self.btn_predict.setStyleSheet("QPushButton {\n"
"    padding: 5px;\n"
"    border-color: black;\n"
"    border-style: outset;\n"
"    border-width: 0px;\n"
"    background-color: rgb(115, 115, 115);\n"
"    color: rgb(255, 255, 255);\n"
"}\n"
"\n"
"QPushButton:hover {    \n"
"    background-color: rgb(211, 215, 207);\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"    background-color: blue;\n"
"}")
        self.btn_predict.setObjectName("btn_predict")
        self.horizontalLayout.addWidget(self.btn_predict)
        self.btn_save = QtWidgets.QPushButton(self.widget_6)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btn_save.sizePolicy().hasHeightForWidth())
        self.btn_save.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.btn_save.setFont(font)
        self.btn_save.setStyleSheet("QPushButton {\n"
"    padding: 5px;\n"
"    border-color: black;\n"
"    border-style: outset;\n"
"    border-width: 0px;\n"
"    background-color: rgb(115, 115, 115);\n"
"    color: rgb(255, 255, 255);\n"
"}\n"
"\n"
"QPushButton:hover {    \n"
"    background-color: rgb(211, 215, 207);\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"    background-color: blue;\n"
"}")
        self.btn_save.setObjectName("btn_save")
        self.horizontalLayout.addWidget(self.btn_save)
        self.horizontalLayout.setStretch(0, 50)
        self.horizontalLayout.setStretch(1, 50)
        self.verticalLayout_8.addLayout(self.horizontalLayout)
        self.btn_exit = QtWidgets.QPushButton(self.widget_6)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btn_exit.sizePolicy().hasHeightForWidth())
        self.btn_exit.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.btn_exit.setFont(font)
        self.btn_exit.setStyleSheet("QPushButton {\n"
"    padding: 5px;\n"
"    border-color: black;\n"
"    border-style: outset;\n"
"    border-width: 0px;\n"
"    background-color: rgb(115, 115, 115);\n"
"    color: rgb(255, 255, 255);\n"
"}\n"
"\n"
"QPushButton:hover {    \n"
"    background-color: rgb(211, 215, 207);\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"    background-color: blue;\n"
"}")
        self.btn_exit.setObjectName("btn_exit")
        self.verticalLayout_8.addWidget(self.btn_exit)
        self.verticalLayout_8.setStretch(0, 33)
        self.verticalLayout_8.setStretch(1, 33)
        self.verticalLayout_8.setStretch(2, 33)
        self.horizontalLayout_4.addLayout(self.verticalLayout_8)
        self.horizontalLayout_4.setStretch(0, 10)
        self.horizontalLayout_4.setStretch(1, 50)
        self.horizontalLayout_4.setStretch(2, 10)
        self.horizontalLayout_4.setStretch(3, 30)
        self.verticalLayout_6.addWidget(self.widget_6)
        self.widget_5 = QtWidgets.QWidget(self.widget_4)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.widget_5.setFont(font)
        self.widget_5.setObjectName("widget_5")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.widget_5)
        self.verticalLayout_7.setContentsMargins(5, 5, 5, 5)
        self.verticalLayout_7.setSpacing(5)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.widget_7 = QtWidgets.QWidget(self.widget_5)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.widget_7.setFont(font)
        self.widget_7.setStyleSheet("background-color: rgb(115, 115, 115);")
        self.widget_7.setObjectName("widget_7")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout(self.widget_7)
        self.verticalLayout_9.setContentsMargins(5, 5, 5, 5)
        self.verticalLayout_9.setSpacing(5)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.lbl_captured = QtWidgets.QLabel(self.widget_7)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.lbl_captured.setFont(font)
        self.lbl_captured.setText("")
        self.lbl_captured.setScaledContents(True)
        self.lbl_captured.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_captured.setObjectName("lbl_captured")
        self.verticalLayout_9.addWidget(self.lbl_captured)
        self.verticalLayout_7.addWidget(self.widget_7)
        self.verticalLayout_6.addWidget(self.widget_5)
        self.verticalLayout_6.setStretch(0, 20)
        self.verticalLayout_6.setStretch(1, 80)
        self.verticalLayout_5.addWidget(self.widget_4)
        self.stackedWidget.addWidget(self.page_2)
        self.page_3 = QtWidgets.QWidget()
        self.page_3.setStyleSheet("background-color: rgb(0, 37, 68);")
        self.page_3.setObjectName("page_3")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.page_3)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setSpacing(0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.widget_11 = QtWidgets.QWidget(self.page_3)
        self.widget_11.setObjectName("widget_11")
        self.horizontalLayout_3.addWidget(self.widget_11)
        self.verticalLayout_13 = QtWidgets.QVBoxLayout()
        self.verticalLayout_13.setObjectName("verticalLayout_13")
        self.widget_10 = QtWidgets.QWidget(self.page_3)
        self.widget_10.setObjectName("widget_10")
        self.verticalLayout_13.addWidget(self.widget_10)
        self.widget_8 = QtWidgets.QWidget(self.page_3)
        self.widget_8.setStyleSheet("#widget_8 {\n"
"    background-color: rgba(0, 0, 0, 100);\n"
"    border :5px;\n"
"    border-radius: 10px;\n"
"}")
        self.widget_8.setObjectName("widget_8")
        self.verticalLayout_14 = QtWidgets.QVBoxLayout(self.widget_8)
        self.verticalLayout_14.setContentsMargins(10, 10, 10, 10)
        self.verticalLayout_14.setSpacing(10)
        self.verticalLayout_14.setObjectName("verticalLayout_14")
        self.lbl_begin_2 = QtWidgets.QLabel(self.widget_8)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.lbl_begin_2.setFont(font)
        self.lbl_begin_2.setStyleSheet("color: rgb(255, 255, 255);\n"
"background-color: transparent;")
        self.lbl_begin_2.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_begin_2.setObjectName("lbl_begin_2")
        self.verticalLayout_14.addWidget(self.lbl_begin_2)
        self.pbar_analyze = QtWidgets.QProgressBar(self.widget_8)
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.pbar_analyze.setFont(font)
        self.pbar_analyze.setStyleSheet("QProgressBar{\n"
"\n"
"border :5px;\n"
"border-radius: 10px;\n"
"}\n"
"\n"
"QProgressBar::chunk{\n"
"    background-color: rgb(138, 226, 52);\n"
"\n"
"border :5px;\n"
"border-radius: 10px;\n"
"}\n"
"")
        self.pbar_analyze.setProperty("value", 24)
        self.pbar_analyze.setAlignment(QtCore.Qt.AlignCenter)
        self.pbar_analyze.setTextVisible(False)
        self.pbar_analyze.setOrientation(QtCore.Qt.Horizontal)
        self.pbar_analyze.setInvertedAppearance(False)
        self.pbar_analyze.setObjectName("pbar_analyze")
        self.verticalLayout_14.addWidget(self.pbar_analyze)
        self.verticalLayout_14.setStretch(0, 50)
        self.verticalLayout_14.setStretch(1, 50)
        self.verticalLayout_13.addWidget(self.widget_8)
        self.widget_9 = QtWidgets.QWidget(self.page_3)
        self.widget_9.setObjectName("widget_9")
        self.verticalLayout_13.addWidget(self.widget_9)
        self.verticalLayout_13.setStretch(0, 40)
        self.verticalLayout_13.setStretch(1, 20)
        self.verticalLayout_13.setStretch(2, 40)
        self.horizontalLayout_3.addLayout(self.verticalLayout_13)
        self.widget_12 = QtWidgets.QWidget(self.page_3)
        self.widget_12.setObjectName("widget_12")
        self.horizontalLayout_3.addWidget(self.widget_12)
        self.horizontalLayout_3.setStretch(0, 30)
        self.horizontalLayout_3.setStretch(1, 40)
        self.horizontalLayout_3.setStretch(2, 30)
        self.stackedWidget.addWidget(self.page_3)
        self.verticalLayout.addWidget(self.stackedWidget)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.stackedWidget.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.lbl_begin.setText(_translate("MainWindow", "Tap here to classify"))
        self.label.setText(_translate("MainWindow", "FILE NAME : "))
        self.label_2.setText(_translate("MainWindow", "PREDICTION : "))
        self.label_7.setText(_translate("MainWindow", "ACCURACY : "))
        self.lbl_accuracy.setText(_translate("MainWindow", "%"))
        self.btn_capture.setText(_translate("MainWindow", "CAPTURE"))
        self.btn_load.setText(_translate("MainWindow", "LOAD"))
        self.btn_predict.setText(_translate("MainWindow", "PREDICT"))
        self.btn_save.setText(_translate("MainWindow", "SAVE"))
        self.btn_exit.setText(_translate("MainWindow", "EXIT"))
        self.lbl_begin_2.setText(_translate("MainWindow", "Classifying Image . . ."))
import barong_rc
