import zipfile
import os
import gdown
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
from google.cloud import storage
from count import *
import numpy as np


def get_run_parameters(trainer):
    return {
        'epochs': trainer.epochs,
        'batch' : trainer.batch,
        'imgsz' : trainer.imgsz,
        'lr0' : trainer.lr0,
        'lrf' : trainer.lrf,
        'pretrained' : trainer.pretrained,
        'momentum' : trainer.momentum,
        'max_det' : trainer.max_det,
        'iou'   : trainer.iou,
        'decay' : trainer.weight_decay,
        'warmup_epochs' : trainer.warmup_epochs,
        'warmup_momentum' : trainer.warmup_momentum,
        'warmup_bias_lr' : trainer.warmup_bias_lr,
        'patience' : trainer.patience,
        'optimizer' : trainer.optimizer,
        'seed' : trainer.seed,
        'deterministic' : trainer.deterministic,
        'single_cls' : trainer.single_cls,
        'rect' : trainer.rect,
        'cos_lr' : trainer.cos_lr,
        'close_mosaic' : trainer.close_mosaic,
        'amp' : trainer.amp,
        'fraction' : trainer.fraction,
        'profile' : trainer.profile,
        'freeze' : trainer.freeze,
        'multi_scale' : trainer.multi_scale,
        'overlap_mask' : trainer.overlap_mask,
        'mask_ratio' : trainer.mask_ratio,
        'dropout' : trainer.dropout,
        'val' : trainer.val,
        'split' : trainer.split,
        'conf' : trainer.conf,
        'half' : trainer.half,
        'dnn' : trainer.dnn,
        'source' : trainer.source,
        'vid_stride' : trainer.vid_stride,
        'augment' : trainer.augment,
        'agnostic_nms' : trainer.agnostic_nms,
        'classes' : trainer.classes,
        'retina_masks' : trainer.retina_masks,
        'embed' : trainer.embed,
        'optimize' : trainer.optimize,
        'int8' : trainer.int8,
        'dynamic' : trainer.dynamic,
        'simplify' : trainer.simplify,
        'opset' : trainer.opset,
        'workspace' : trainer.workspace,
        'nms' : trainer.nms,
        'box' : trainer.box,
        'cls' : trainer.cls,
        'dfl' : trainer.dfl,
        'pose' : trainer.pose,
        'kobj' : trainer.kobj,
        'label_smoothing' : trainer.label_smoothing,
        'nbs' : trainer.nbs,
        'hsv_h' : trainer.hsv_h,
        'hsv_s' : trainer.hsv_s,
        'hsv_v' : trainer.hsv_v,
        'degrees' : trainer.degrees,
        'translate' : trainer.translate,
        'scale' : trainer.scale,
        'shear' : trainer.shear,
        'perspective' : trainer.perspective,
        'flipud' : trainer.flipud,
        'fliplr' : trainer.fliplr,
        'bgr' : trainer.bgr,
        'mosaic' : trainer.mosaic,
        'mixup' : trainer.mixup,
        'copy_paste' : trainer.copy_paste,
        'copy_paste_mode' : trainer.copy_paste_mode,
        'auto_augment' : trainer.auto_augment,
        'erasing' : trainer.erasing,
        'crop_fraction' : trainer.crop_fraction
    }


def get_methadata(
        model,
        model_type,
        experiment_id
    ):

    run_params = get_run_parameters(model.trainer.args)

    local_best_path = os.path.join(os.getcwd(), model.trainer.args.save_dir,'weights/best.pt')
    storage_best_path = None
    if os.path.exists(local_best_path):
        storage_best_path = f'yolov8/{experiment_id}/best.pt'


    local_last_path = os.path.join(os.getcwd(), model.trainer.args.save_dir, 'weights/last.pt')
    storage_last_path = None
    if os.path.exists(local_last_path):
        storage_last_path = f'yolov8/{experiment_id}/last.pt'

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'experiment_id' : experiment_id,
        'run_params' : run_params,
        'date' : datetime.now().isoformat(),
        'total_params' : total_params,
        'trainable_params' : trainable_params,
        'model_type' : model_type,
        'storage_best_url' : storage_best_path,
        'storage_last_url' : storage_last_path,
        'local_best_path' : local_best_path,
        'local_last_path' : local_last_path
    }


def get_detect_metrics(model):

    validation = model.trainer.validator.metrics  

    return {
        'map_50'    : validation.box.map50,
        'map_50:95' : validation.box.map,
        'p_50'      : validation.box.p[0],
        'r_50'      : validation.box.r[0],
        'f1_50'     : validation.box.f1[0]
    }


def get_error(estimated, correct):
    estimated = float(estimated)
    correct = float(correct)
    return np.abs(estimated-correct)/correct


def get_number_blueberries(
    video_path,
    weights_path,
    conf_threshold
   
):
    number_blueberries = count_on_video(video_path, 
                                        weights_path,
                                        conf_threshold,
                                        show_video=True)
    return number_blueberries


def compute_mean_error(
    weights_path,
    video_data,
    conf_threshold
    ):

    error_list = []

    for video in video_data:

        print(f"------------start processing video -----------------")

        video_path = video.get('video_path')

        estim_num_blueb = count_on_video(video_path, weights_path, conf_threshold, show_video=True)

        corr_num_blueb = video.get('number')

        error = get_error(estim_num_blueb, corr_num_blueb)

        error_list.append(error)

    return np.mean(error_list)


def get_videos_data():
    return [
        {
            'video_path' :  os.path.join(os.getcwd(), 'videos', '20230929_2.mp4'),
            'number' : 155 
        },
        {
            'video_path' :  os.path.join(os.getcwd(), 'videos', '20230929_5.mp4'),
            'number' : 162
        }
    ]


def get_count_metrics(
    model
):

    video_data = get_videos_data() 

    weights_path = os.path.join(os.getcwd(), model.trainer.args.save_dir,'weights/best.pt')

    merror_50 = compute_mean_error(
        weights_path,
        video_data,
        conf_threshold=0.5
    )

    merror_30 = compute_mean_error(
        weights_path,
        video_data,
        conf_threshold=0.3
    )

    return {
        'merror_50' : str(merror_50),
        'merror_30' : str(merror_30)
    }


def download_dataset(dir_path):

    if os.path.exists(dir_path) : return

    file_id = '1WrfMHleFm7PKcFrdV8YMj7JfOip8UBGD'
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, quiet=False)

    zip_file_path = './blueberryOD_640x640_669imgs.zip'
    os.makedirs(dir_path, exist_ok=True)

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(dir_path)
        print(f"Extracted all files to: {dir_path}")

    return


def download_videos(dir_path):

    if os.path.exists(dir_path) : return

    output = dir_path
    os.makedirs(output, exist_ok=True)

    file_id = '1ldVz-gyiRlGQtAquvRfxE6s_3oT632gH' 
    url = f'https://drive.google.com/uc?id={file_id}'
    output = os.path.join(dir_path, '20230929_5.mp4')
    gdown.download(url, output=output, quiet=False)

    file_id = '1OjWGfilEQGFfq8snqTwchCJ9AS0s-ARr' 
    url = f'https://drive.google.com/uc?id={file_id}'
    output = os.path.join(dir_path, '20230929_2.mp4')
    gdown.download(url, output=output, quiet=False)

    return


def save_metrics(
    service_account_json_path : str,
    collection_name : str,
    detect_metrics,
    count_metrics,
    methadata
):
    cred = credentials.Certificate(service_account_json_path) 
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred) 
    db = firestore.client()

    storage_client = storage.Client.from_service_account_json(service_account_json_path)
    bucket = storage_client.bucket('blueberry-detection-benchmark.firebasestorage.app')

    try:
        db.collection(collection_name).document(methadata['experiment_id']).set(
            {
                'methadata' : methadata,
                'detect_metrics' : detect_metrics,
                'count_metrics' : count_metrics
            }
        )
          
        if os.path.exists(methadata['local_best_path']):
            blob_best = bucket.blob(methadata['storage_best_url'])
            blob_best.upload_from_filename(methadata['local_best_path'])

        if os.path.exists(methadata['local_last_path']):
            blob_last = bucket.blob(methadata['storage_last_url'])
            blob_last.upload_from_filename(methadata['local_last_path'])
        
        print(f"Metrics saved for experiment_id: {methadata['experiment_id']}")
    except Exception as e:
        print(f"Failed to save metrics: {e}")
    return


def add_metrics(
    firebase_credentials_json_path : str,
    collection_name : str,
    model_type : str,
    folder_path : str,
    imgsz : int,
    epochs : int
):
    try:

        download_dataset(os.path.join(os.getcwd(),'data'))
        download_videos(os.path.join(os.getcwd(),'videos'))

        experiment_id = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')  

        model = YOLO(model_type)
        model.train(
            data=os.path.join(folder_path, "data.yaml"),
            epochs=epochs,
            imgsz=imgsz,
            project="blueberry-detection",
            name="run"
        )
        model.val()

        methadata = get_methadata(model, model_type, experiment_id)

        detect_metrics = get_detect_metrics(model)

        count_metrics = get_count_metrics(model)

        save_metrics(
            service_account_json_path=firebase_credentials_json_path,
            collection_name=collection_name,
            detect_metrics = detect_metrics,
            count_metrics = count_metrics,
            methadata = methadata
        )

    except Exception as e:
        print(f"An error occurred: {e}")
        raise

#
# if __name__ == '__main__':
#
#     epochs = 1
#     imgsz = 640
#     model_type = 'yolov8n.pt'
#     firebase_credentials_json_path = './blueberry-detection-benchmark-firebase-adminsdk-khx5o-41dd2be9b2.json'
#     collection_name = 'yolov8_metrics'
#     
#     download_dataset(os.path.join(os.getcwd(),'data'))
#     download_videos(os.path.join(os.getcwd(),'videos'))
#     
#
#     experiment_id = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')  
#      
#
#     model = YOLO(model_type)
#     model.train(
#         data=os.path.join(os.getcwd(),'data','blueberryOD_640x640_669imgs', "data.yaml"),
#         epochs=epochs,
#         imgsz=imgsz,
#         project="blueberry-detection",
#         name="run"
#     )
#     model.val()
#    
#     methadata = get_methadata(model, model_type, experiment_id)
#     print(methadata)
#    
#     detect_metrics = get_detect_metrics(model)
#     print(detect_metrics)
#    
#     count_metrics = get_count_metrics(model)
#     print(count_metrics)
#     
#    
#     save_metrics(
#         service_account_json_path=firebase_credentials_json_path,
#         collection_name=collection_name,
#         detect_metrics = detect_metrics,
#         count_metrics = count_metrics,
#         methadata = methadata
#     )
#     

#
#     firebase_credentials_json_path = './blueberry-detection-benchmark-firebase-adminsdk-khx5o-41dd2be9b2.json'
#
#     # this must be added using 
#     folder_path = os.path.join(os.getcwd(), 'blueberryOD_640x640_669imgs')
#     collection_name = 'yolov8_metrics'
#     model_type = 'yolov8n.pt'
#     epochs = 1
#     imgsz = 640
#
#     main(
#         firebase_credentials_json_path,
#         collection_name,
#         model_type,
#         folder_path,
#         imgsz,
#         epochs
#     )
#
