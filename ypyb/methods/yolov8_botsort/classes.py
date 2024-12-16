from ultralytics import YOLO
import cv2
import torch

class Yolo8:
    def __init__(self, weights, device):
        self.weights = weights
        self.device = device
        self.model = YOLO(self.weights)

    def detect(self, img, conf_thres=0.5):
        results = self.model(img, conf=conf_thres)
        return results 

    def track(self, img, conf_thres=0.5):
        results = self.model.track(img, conf=conf_thres, persist=True)
        return results

    def plot_prediction(self, img, results):
        num_bboxes = results[0].boxes.data.shape[0]
        if num_bboxes < 1: return img

        for i in range(results[0].boxes.xywh[:].shape[0]):
            x = results[0].boxes.xywh[i][0].item()
            y = results[0].boxes.xywh[i][1].item()
            w = results[0].boxes.xywh[i][2].item()
            h = results[0].boxes.xywh[i][3].item()

            cv2.circle(img, (int(x),int(y)), int(h/2), (0,0,255), 2)

            if results[0].boxes.id is not None:

                text = str(int(results[0].boxes.id[i].item()))

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_thickness = 2
                (text_width, text_height), baseline = cv2.getTextSize(text, 
                                                                      font, 
                                                                      font_scale, 
                                                                      font_thickness)
                box_center = (int(x),int(y))
                text_x = box_center[0] - text_width // 2
                text_y = box_center[1] + text_height // 2

                cv2.putText(img, text, (text_x, text_y), font, font_scale, 
                            (255, 255, 255), font_thickness)

        return img


