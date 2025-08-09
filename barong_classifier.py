
import sys

from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication

from src.constants import *

from gui.window_main import MainWindow

from src.service_frame_grabber import ServiceGrabber


class BarongClassifier:
	def __init__(self):
		
		self.app = QApplication(sys.argv)
		self.main_window = MainWindow(callback=self)
		
		if not RPI_PLATFORM:
			self.main_window.show()
		else:
 			self.main_window.showFullScreen()
		#self.main_window.show()
		# self.main_window.showFullScreen()
		
		self.camera_process = ServiceGrabber()
		
		self.running = True
	
	def on_capture(self):
		
		self.camera_process.tx.put((CAPTURE_IMG,))
		
		self.main_window.lbl_filename.setText("CAMERA CAPTURE")
		self.main_window.lbl_prediction.setText("")
		self.main_window.lbl_accuracy.setText("%")
		
	def on_classify(self):
		self.camera_process.tx.put((CLASSIFY_IMG,))
			
	def update_screen(self):
		
		command, data = self.camera_process.get_data()
		
		if command is not None:
			if command == CLASSIFICATION_DONE:
				# set text result for classification
				class_name = data.get("class_name")
				confidence_score = data.get("confidence_score")
				
				self.main_window.lbl_prediction.setText(class_name)
				self.main_window.lbl_accuracy.setText("{}%".format(confidence_score))
				
			elif command == CAPTURE_IMG_DONE:
				self.main_window.lbl_captured.setPixmap(QPixmap(TEMP_IMG))
	
	def run(self):
		try:
			while self.running:
				self.update_screen()
				
				self.app.processEvents()
		except KeyboardInterrupt:
			pass
		self.quit()
	
	def quit(self):
		self.app.exit()
	
	def on_exit(self):
		self.camera_process.quit()
		self.running = False


if __name__ == '__main__':
	BarongClassifier().run()