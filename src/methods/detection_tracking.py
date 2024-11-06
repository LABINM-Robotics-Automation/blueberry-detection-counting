from blueberry_detection_counting.detection.yolov8 import Yolo8
import cv2
import torch
import time

class Counter:
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
                            set_0.add(id) 
                            set_1.discard(id)
                        elif id in set_0:
                            set_1.add(id)

                    if self.count_mode == 'vertical':
                        if self.count_condition(y.item()):    
                            set_0.add(id) 
                            set_1.discard(id)
                        elif id in set_0:
                            set_1.add(id)

                self.LIST_0 = list(set_0)
                self.LIST_1 = list(set_1)    
        return

    def get_number_counted(self):
        return { 'counted': len(self.LIST_1) }
    
    def plot_line_threshold(self, img_pred):
        if self.count_mode == 'vertical':
            cv2.line(img_pred, 
                     (0,self.threshold_track), 
                     (self.threshold_track,self.threshold_track), 
                     (0,255,0), 2)

        if self.count_mode == 'horizontal':
            cv2.line(img_pred, 
                     (self.threshold_track,0), 
                     (self.threshold_track,self.threshold_track), 
                     (0,255,0), 2)

        return img_pred


class Method:
    def __init__(self, weights_path):

        self.counter = Counter(count_mode='vertical', 
                               threshold_track=300, 
                               direction='down2top')
         
        self.detector = Yolo8(weights = weights_path,
                              device = 'cuda')
          
        return

    def process_image(self, image):
        prediction = self.detector.predict(image, conf_thres = 0.5)
        number_blueberries = prediction[0].boxes.shape[0] 
        return number_blueberries


    def process_video(self, video_path):

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print('Error')
            exit()

        blueberry_counter =  self.counter

        number_blueberries = 0
          
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print('Ret is not')
                break

            start_time = time.time()
            prediction = self.detector.predict(frame, 
                                               conf_thres = 0.5,
                                               enable_tracking=True)
            end_time = time.time()

            blueberry_counter.update_count(prediction)
            frame = blueberry_counter.plot_line_threshold(frame)
            number_blueberries = blueberry_counter.get_number_counted()['counted']
            frame = self.detector.plot_prediction(frame, prediction)

            print(f'inference-time: {(end_time - start_time)*1000:.2f} [mS]')
            print(f'number-blueber: {number_blueberries}')

            if 0xFF == ord('q'):
                break

        return number_blueberries  

