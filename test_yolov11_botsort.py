import ypyb.methods.yolov11_botsort.ops as o
import ypyb.methods.yolov11_botsort.classes as c 
import cv2
import os

conf  = 0.5
image = cv2.imread(f'{os.getcwd()}/gallery/cat.jpeg')
model = c.Yolo11('yolov11n.pt', device='cuda')

detection = o.detect_on_image(image, conf, model)

cv2.imshow('Image', detection)

while True:
    key = cv2.waitKey(0)
    if key == ord('q'):
        cv2.destroyAllWindows()

print("Hello world")