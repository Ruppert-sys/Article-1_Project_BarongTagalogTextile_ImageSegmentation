
import os
import sys
import cv2
import ntpath

from datetime import datetime

from PIL import Image

from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox, QFileDialog, QHeaderView, QTableWidgetItem, QGraphicsDropShadowEffect

from gui.ui_window_main import Ui_MainWindow

from src.constants import *


class MainWindow(QMainWindow, Ui_MainWindow):
	
	def __init__(self, callback, parent=None):
		super(MainWindow, self).__init__(parent)
		QMainWindow.__init__(self, parent)
		self.setupUi(self)
		
		self.callback = callback
		
		self.alert = QMessageBox()
		self.options = QFileDialog.Options()
		
		# page 1 welcome screen
		self.lbl_begin.mousePressEvent = self.lbl_begin_clicked
		
		# page 2 predict screen
		self.btn_capture.clicked.connect(self.btn_capture_clicked)
		self.btn_load.clicked.connect(self.btn_load_clicked)
		self.btn_save.clicked.connect(self.btn_save_clicked)
		self.btn_predict.clicked.connect(self.btn_predict_clicked)
		self.btn_exit.clicked.connect(self.btn_exit_clicked)
		
		# page 3 progress screen
		self.q_timer = QTimer(self)
		self.q_timer.timeout.connect(self.update_progress)
		self.progress_value = 0
		
		self.set_screen(SCREEN_WELCOME)
		
		self.selected_image = None
		self.predicted_image = False
	
	# page 2 welcome screen
	def lbl_begin_clicked(self, event):
		self.set_screen(SCREEN_PREDICT)
	
	def btn_capture_clicked(self):
		self.callback.on_capture()
		self.selected_image = "CAMERA_CAPTURE"
	
	def btn_load_clicked(self):
		self.open_filename_dialog()
	
	def btn_save_clicked(self):
		if self.predicted_image:
			file_name = "{}.jpg".format(datetime.now().strftime(DATETIME_FORMAT))
			img = cv2.imread(TEMP_IMG)
			
			h, w, _ = img.shape
			
			prediction = "{} : {}".format(self.lbl_prediction.text(), self.lbl_accuracy.text())
			cv2.rectangle(img, (10, 10), (w - 10, h - 10), (0, 0, 255), 2)
			cv2.putText(img, prediction, (15, 25), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
			
			cv2.imwrite(os.path.join(SAVED_PREDICTIONS, file_name), img)
			
			self.show_notifications("SUCCESS", "Prediction image has been saved!")
			
		else:
			self.show_notifications("ERROR","Predict and image first!")
	
	def btn_predict_clicked(self):
		if self.selected_image is not None:
			self.set_screen(SCREEN_PROGRESS)
			
			# set the progress bar progress to 0
			self.progress_value = 0
			self.pbar_analyze.setValue(0)
			self.q_timer.start(MILLIS_INCREMENT)  # milliseconds
			
			self.callback.on_classify()
		else:
			self.show_notifications("ERROR", "Please capture or load an image!")
	
	def update_progress(self):
		self.progress_value += 1
		if self.progress_value > 100:
			self.progress_value = 0
			self.q_timer.stop()
			
			self.predicted_image = True
			
			# self.show_notifications("SUCCESS", "Done analyzing!")
			self.set_screen(SCREEN_PREDICT)
			return
		
		self.pbar_analyze.setValue(self.progress_value)
	
	def btn_exit_clicked(self):
		self.set_screen(SCREEN_WELCOME)
		
		self.selected_image = None
		self.predicted_image = False
		
		self.lbl_filename.setText("")
		self.lbl_prediction.setText("")
		self.lbl_accuracy.setText("%")
		self.lbl_captured.setPixmap(QPixmap(""))
	
	def open_filename_dialog(self):
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		filename, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "JPG (*.jpg);;JPEG (*.jpeg);;PNG (*.png);;All Files (*)", options=options)
		
		if filename:
			# print(filename)
			if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
				self.selected_image = filename
				self.lbl_filename.setText(filename.split("/")[-1])
				
				self.lbl_prediction.setText("")
				self.lbl_accuracy.setText("%")
				
				# save the image in the temporary location
				img = cv2.imread(self.selected_image)
				cv2.imwrite(TEMP_IMG, img)
				
				self.lbl_captured.setPixmap(QPixmap(TEMP_IMG))
				
			else:
				self.show_notifications("ERROR", "Invalid Image format!")
				self.selected_image = None
	def show_notifications(self, msg_title, msg):
		self.alert.setWindowTitle(msg_title)
		self.alert.setText(msg)
		self.alert.show()
	
	def on_confirm(self, title, msg):
		response = self.alert.question(self, title, msg, self.alert.Yes | self.alert.No)
		if response == self.alert.Yes:
			return True
		else:
			return False
	
	def set_screen(self, screen):
		self.stackedWidget.setCurrentIndex(screen)
	
	def closeEvent(self, event):
		if self.on_confirm('Exit?', 'You sure?'):
			if self.callback is not None:
				self.callback.on_exit()
			event.accept()
		else:
			event.ignore()
		self.alert.close()
	
	def keyPressEvent(self, event):
		val = event.key()
		
		if val == Qt.Key_Escape:
			pass


def run():
	app = QApplication(sys.argv)
	m_window = MainWindow(None, None)
	m_window.showMaximized()
	
	try:
		while True:
			app.processEvents()
	except KeyboardInterrupt:
		pass
	m_window.close()
	app.quit()


if __name__ == '__main__':
	run()
