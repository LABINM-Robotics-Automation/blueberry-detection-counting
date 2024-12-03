from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
import cv2
import torch

class Yolo8:
    def __init__(self, weights, device):
        self.weights = weights
        self.device = device
        self.model = YOLO(self.weights, verbose=False)
        self.letterbox = LetterBox()


    def predict(self, img, conf_thres=0.5, enable_tracking = False):
        if enable_tracking != False:
            results = self.model.track(img, conf=conf_thres, persist=True)
            return results

        results = self.model(img, conf=conf_thres)
        return results    


    def plot_prediction(self, img, results):

        num_bboxes = results[0].boxes.data.shape[0]

        if num_bboxes < 1: return img

        for i in range(results[0].boxes.xywh[:].shape[0]):
            x = results[0].boxes.xywh[i][0].item()
            y = results[0].boxes.xywh[i][1].item()
            w = results[0].boxes.xywh[i][2].item()
            h = results[0].boxes.xywh[i][3].item()
            #id = int(results[0].boxes.id[i].item())

            cv2.rectangle(img, (int(x - w/2),int(y - h/2)), (int(x + w/2),int(y + h/2)), 
                          (0,0,255), 2) 

        return img


class counter:
    def __init__(self, count_mode, threshold_track, direction):
        self.LIST_0 = []
        self.LIST_1 = []
        self.count_mode = count_mode
        self.threshold_track = threshold_track
        self.direction = direction

        if self.direction == 'top2down':
            self.count_condition = lambda y: y > self.threshold_track
        elif self.direction == 'down2top':
            self.count_condition = lambda y: y < self.threshold_track 
        elif self.direction == 'left2right':
            self.count_condition = lambda x: x > self.threshold_track
        elif self.direction == 'right2left':
            self.count_condition = lambda x: x < self.threshold_track
        

    def update_count(self, prediction=None):
        
        if prediction[0] is not None and prediction is not None and prediction[0].boxes.shape[0] > 2:
            print('prediction')
            boxes = prediction[0].boxes.xywh.cpu()
            centers = boxes[:,:2]
            
            if prediction[0].boxes.id is not None:
                print('counting')
                track_ids = prediction[0].boxes.id.int().cpu()
                track_ids = track_ids.reshape(track_ids.shape[0], 1)

                to_count = torch.cat((track_ids, centers),1)

                set_0 = set(self.LIST_0)
                set_1 = set(self.LIST_1)
                
                for (id, x, y) in to_count:
                    id = id.item()

                    if self.count_mode == 'horizontal':
                        if self.count_condition(x.item()):
                            set_0.add(id)       # Adds the id if not already present                        
                            set_1.discard(id)   # Removes the id if present
                        elif id in set_0:
                            set_1.add(id)

                    if self.count_mode == 'vertical':
                        if self.count_condition(y.item()):    
                            set_0.add(id)       # Adds the id if not already present                        
                            set_1.discard(id)   # Removes the id if present
                        elif id in set_0:
                            set_1.add(id)

                self.LIST_0 = list(set_0)
                self.LIST_1 = list(set_1)
    
        return

    def get_number_counted(self):
        return {
            'counted': len(self.LIST_1)
        }
    
    def plot_line_threshold(self, img_pred):
        if self.count_mode == 'vertical':
            cv2.line(img_pred, (0,self.threshold_track), (self.threshold_track,self.threshold_track), (0,255,0), 2)
        if self.count_mode == 'horizontal':
            cv2.line(img_pred, (self.threshold_track,0), (self.threshold_track,self.threshold_track), (0,255,0), 2)
        return img_pred


