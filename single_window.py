from PyQt5.QtCore import pyqtSignal, QThread, Qt, QSize
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QPushButton, QLabel, QVBoxLayout, QWidget, QSizePolicy
from ultralytics import YOLO
import cv2
import math
import sys

class Detection(QWidget):
    def __init__(self):
        super(Detection, self).__init__()

        # Initialize the layout
        self.VBL = QVBoxLayout()

        # Initialize the label to display the video feed
        self.FeedLabel = QLabel()
        self.FeedLabel.setAlignment(Qt.AlignCenter)  # Center align the feed label
        self.FeedLabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Set size policy
        self.VBL.addWidget(self.FeedLabel)

        # Initialize the cancel button
        self.CancelBTN = QPushButton("Cancel")
        self.CancelBTN.clicked.connect(self.CancelFeed)
        self.VBL.addWidget(self.CancelBTN)

        # Initialize the worker thread for processing video feed
        self.Worker1 = Worker1()
        self.Worker1.start()
        self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)

        # Set the layout
        self.setLayout(self.VBL)
        # Set the window title
        self.setWindowTitle("Sleepiness Detection")  

    # Slot to update the image label with the latest frame
    def ImageUpdateSlot(self, Image):
        pixmap = QPixmap.fromImage(Image)
        scaled_pixmap = pixmap.scaled(self.FeedLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)  # Resize pixmap
        self.FeedLabel.setPixmap(scaled_pixmap)

    # Function to stop the video feed processing
    def CancelFeed(self):
        self.Worker1.stop()

class Worker1(QThread):
    ImageUpdate = pyqtSignal(QImage)

    def run(self):
        self.ThreadActive = True

        # Load class names for object detection
        with open('obj.names', 'r') as f:
            classes = [line.strip() for line in f.readlines()] 

        # Initialize YOLO model
        model = YOLO('best.pt')
        # Initialize video capture
        cap = cv2.VideoCapture(1)

        while self.ThreadActive:
            ret, frame = cap.read()

            if ret:
                # Convert frame to RGB
                Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Flip frame horizontally
                FlippedImage = cv2.flip(Image, 1)
                # Convert to QImage format
                ConvertToQtFormat = QImage(FlippedImage.data, FlippedImage.shape[1], FlippedImage.shape[0], QImage.Format_RGB888)
                
                # Convert to QImage with detected boxes drawn
                image_with_boxes = self.draw_boxes(FlippedImage, model, classes)
                # Emit the updated image
                self.ImageUpdate.emit(image_with_boxes)

    # Function to draw bounding boxes around detected objects
    def draw_boxes(self, image, model, classes):
        result = model(image, stream=True)

        for info in result:
            for box in info.boxes:
                confidence = box.conf[0]
                confidence = math.ceil(confidence * 100)
                Class = int(box.cls[0])

                if confidence > 0:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    # Draw rectangle around the object
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
                    # Put text label with object class and confidence level
                    cv2.putText(image, f'{classes[Class]} {confidence}%', (x1 + 8, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 3)

        # Convert image to QImage format
        image_with_boxes = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
        return image_with_boxes

    # Function to stop the worker thread
    def stop(self):
        self.ThreadActive = False
        self.quit()

if __name__== "__main__":
    App = QApplication(sys.argv)
    Root = Detection()
    Root.showMaximized()  # Show the window maximized
    sys.exit(App.exec())
