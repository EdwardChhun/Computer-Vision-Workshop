# https://www.geeksforgeeks.org/object-detection-with-yolo-and-opencv/
import cv2
from ultralytics import YOLO

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

def ifRock():
    pass

def ifPaper():
    pass

def ifScissors():
    pass

while True:
    ret, frame = videoCap.read()
    if not ret:
        continue
    results = yolo.track(frame, stream=True)


    for result in results:
        # get the classes names
        classes_names = result.names

        # iterate over each box
        # https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Boxes
        for box in result.boxes:
            # check if confidence is greater than 20 percent
            if box.conf[0] > 0.2:
                # get coordinates
                [x1, y1, x2, y2] = box.xyxy[0]
                # convert to int
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # get the class
                cls = int(box.cls[0])

                # get the class name
                class_name = classes_names[cls]

                # get the respective colour
                colour = getColours(cls)

                # draw the rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

                # put the class name and confidence on the image
                cv2.putText(frame, f'{classes_names[int(box.cls[0])]} {box.conf[0]:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)
                
    # show the image
    cv2.imshow('frame', frame)

    # break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the video capture and destroy all windows
videoCap.release()
cv2.destroyAllWindows()
