import ypyb.methods.yolov8_botsort.ops as ops
import ypyb.methods.yolov8_botsort.classes as c
import cv2
import os

image = cv2.imread(f'{os.getcwd()}/gallery/cat.jpeg')
conf  = 0.5
model = c.Yolo8('yolov8n.pt', device='cuda')


detection = ops.detect_on_image(image, conf, model)

cv2.imshow('Image', detection)

while True:
    key = cv2.waitKey(0)
    if key == ord('q'):
        cv2.destroyAllWindows()
