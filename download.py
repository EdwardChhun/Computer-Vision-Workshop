from roboflow import Roboflow

rf = Roboflow(api_key="r7qSIBlL0ryPm5qonSPH")
project = rf.workspace("roboflow-58fyf").project("rock-paper-scissors-sxsw")
version = project.version(14)
dataset = version.download("yolov11")
                