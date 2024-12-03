from classes import *  
import cv2
import time
import torch

def crop_image(image):
    # Get dimensions
    height, width = image.shape[:2]

    # Determine the size of the square crop
    square_size = min(width, height)

    # Calculate coordinates for the center crop
    x_center = width // 2
    y_center = height // 2

    x_start = x_center - square_size // 2
    y_start = y_center - square_size // 2
    x_end = x_start + square_size
    y_end = y_start + square_size

    # Crop the image
    cropped_image = image[y_start:y_end, x_start:x_end]

    return cropped_image

def count_on_video(
        video_path, 
        weights_path,
        conf_threshold=0.5,
        show_video=False
    ):

    number_blueberries = 0
    yolov8 = Yolo8(
        weights = weights_path,
        device = 'cuda'
    )

    print(video_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print('Error')
        exit()

    blueberry_counter = counter(count_mode='vertical', 
                                threshold_track=300, 
                                direction='down2top')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print('Ret is not')
            break

        start_time = time.time()

        frame = crop_image(frame)

        prediction = yolov8.predict(frame, 
                                    conf_thres = conf_threshold,
                                    enable_tracking=True)
        end_time = time.time()

        blueberry_counter.update_count(prediction)
        frame = blueberry_counter.plot_line_threshold(frame)
        number_blueberries = blueberry_counter.get_number_counted()['counted']
        frame = yolov8.plot_prediction(frame, prediction)

        print(f'inference-time: {(end_time - start_time)*1000:.2f} [mS]')
        print(f'number-blueber: {number_blueberries}')

        if show_video:
            cv2.imshow('Video', frame)
            cv2.waitKey(1)
            continue

        if 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return blueberry_counter.get_number_counted()['counted']


