from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
import cv2
import torch

class Yolo9:
    def __init__(self, weights, device):
        self.weights = weights
        self.device = device
        self.model = YOLO(self.weights, verbose=False)
        self.letterbox = LetterBox()

    def detect(self, img, conf_thres=0.5):
        '''

        Args
            img        : is input where we perform object detection
            conf_thres : float between 0 and 1 
        '''

        results = None
        return results 


    def plot_prediction(self, img, results):
        '''

        Args
            img     : is input where we perform object detection
            results : are the output of the model (what returns detect method) 
        '''
        img = None
        return img
