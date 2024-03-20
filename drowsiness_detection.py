import cv2
from ultralytics import YOLO
import math

class Detection:
    def __init__(self):
        # Initialize the worker thread for processing video feed
        self.Worker1 = Worker1()

    def run(self):
        # Start the video feed processing
        self.Worker1.run()

class Worker1:
    def __init__(self):
        self.ThreadActive = True

    def run(self):
        # Load class names for object detection
        with open('obj.names', 'r') as f:
            classes = [line.strip() for line in f.readlines()] 

        # Initialize YOLO model
        model = YOLO('best.pt')
        # Initialize video capture
        cap = cv2.VideoCapture(0)  # Change the argument to use a different camera if needed

        while self.ThreadActive:
            ret, frame = cap.read()

            if ret:
                # Flip frame horizontally
                FlippedImage = cv2.flip(frame, 1)                
                # Convert to image with detected boxes drawn
                image_with_boxes = self.draw_boxes(FlippedImage, model, classes)
                # Add text indicating to press 'q' to exit
                cv2.putText(image_with_boxes, "Press 'q' to exit", (10, image_with_boxes.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                # Show the image with boxes
                cv2.imshow('Sleepiness Detection', image_with_boxes)

                # Break the loop if 'q' key is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # Release the video capture
        cap.release()
        cv2.destroyAllWindows()

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

        return image

    # Function to stop the worker thread
    def stop(self):
        self.ThreadActive = False

if __name__== "__main__":
    Root = Detection()
    Root.run()
