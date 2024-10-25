from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

class Yolo8:
    def __init__(self, weights, device):
        self.weights = weights
        self.device = device
        self.model = YOLO(self.weights)
        self.letterbox = LetterBox()

    def predict(self, img, conf_thres=0.5):
        results = self.model(img, conf=conf_thres)
        return results
    

    def plot_prediction(self, img, results):

        num_bboxes = results[0].boxes.data.shape[0]

        if num_bboxes < 1: return img
        
        img = results[0].plot(conf=False, masks=False, labels=True, font_size = 0.1)

        return img
