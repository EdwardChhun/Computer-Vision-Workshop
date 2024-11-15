import cv2
from ultralytics import YOLO
from tts import speak
import threading

# Load the model
yolo = YOLO(r'rps_prediction\weights\best.pt')

# Load the video capture
videoCap = cv2.VideoCapture(0)

# Custom Function to get class colors
def getColours(cls_num):
    if cls_num == 0:
        return (255, 0, 0)  # Red for Paper
    elif cls_num == 1:
        return (0, 255, 0)  # Green for Rock
    elif cls_num == 2:
        return (0, 0, 255)  # Blue for Scissors
    else:
        return (255, 255, 255)  # White for any other class

# Function to run speak in a thread
def runSpeak(cls):
    if cls == 0: 
        speak("Paper")
    elif cls == 1: 
        speak("Rock")
    elif cls == 2: 
        speak("Scissors")

# Wrapper for threading
# https://www.geeksforgeeks.org/multithreading-python-set-1/
def classCondition(cls):
    # args has to be a tuple
    thread = threading.Thread(target=runSpeak, args=(cls,))
    thread.start()
    
# Example with multiple arguents
# def example_function(arg1, arg2, arg3):
#     print(arg1, arg2, arg3)

# thread = threading.Thread(target=example_function, args=(10, 20, 30))
# thread.start()

while True:
    ret, frame = videoCap.read()
    if not ret:
        continue
    
    # Perform prediction
    # https://docs.ultralytics.com/modes/predict/#inference-arguments
    results = yolo.predict(frame, iou=0.2)

    for result in results:
        # Get the classes names
        classes_names = result.names

        # Iterate over each box
        for box in result.boxes:
            # Check if confidence is greater than 35 percent
            if box.conf[0] > 0.35:
                # Get coordinates
                [x1, y1, x2, y2] = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Get the class
                cls = int(box.cls[0])

                # Function call when found a class (non-blocking)
                classCondition(cls)

                # Get the respective colour
                colour = getColours(cls)

                # Draw the rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

                # Put the class name and confidence on the image
                cv2.putText(frame, f'{classes_names[cls]} {box.conf[0]:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)
                
    # Show the image
    cv2.imshow('frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and destroy all windows
videoCap.release()#
cv2.destroyAllWindows()
