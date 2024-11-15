from ultralytics import YOLO


model = YOLO('yolo11n.pt')


# Training.
# https://docs.ultralytics.com/modes/train/#train-settings
results = model.train(
   data=r'D:\Developer\Computer-Vision-Workshop\rock-paper-scissors.v14i.yolov11\data.yaml', # Path to your data.yaml file
   imgsz=640,
   epochs=10,
   batch=16,
   save=True,
   device = "cpu", # Change to CUDA if you have CUDA Toolkit
   pretrained = True,
   project = r'D:\Developer\Computer-Vision-Workshop', # Path to your working directory
   name='rps_prediction')