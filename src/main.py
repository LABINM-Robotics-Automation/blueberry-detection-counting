import pandas as pd
import gdown
# from methods import method1
import utils
import os
import shutil
from methods import detection_tracking as dt

def download_videos():
    df = pd.read_csv('videos.csv')
    file_id = df.iloc[0]['fileId']
    url = f'https://drive.google.com/uc?id={file_id}'
    file_path = gdown.download(url, quiet=False)
    destination = './videos'
    os.makedirs(destination, exist_ok=True)
    shutil.move(file_path, destination)
    return file_path

def download_weights():
    df = pd.read_csv('weights.csv')
    file_id = df.iloc[0]['fileId']
    url = f'https://drive.google.com/uc?id={file_id}'
    file_path = gdown.download(url, quiet=False)
    destination = './weights'
    os.makedirs(destination, exist_ok=True)
    shutil.move(file_path, destination)
    return file_path


def test_counting_method(method):
    
    number_blueberries = method.process_video(video_path = './videos/20230929_2.mp4')
    print('number_blueberries:', number_blueberries)

    # for video in database.T_Videos:
    #     if not does_exist_video(video): utils.download_video(video)
    #     number_blueberries = method.process_video(video)
    #     metrics = utils.compute_metrics(number_blueberries, video.number_blueberries)
    #     utils.log(video, method, metrics)
    #

if __name__ == '__main__':
    # test_counting_method(method1)
    # download_videos()
    # download_weights()
    method = dt.Method(weights_path = './weights/yolov8m_best.pt')
    test_counting_method(method)

