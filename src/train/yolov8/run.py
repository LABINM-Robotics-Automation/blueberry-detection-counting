import zipfile
import os
import gdown
from ultralytics import YOLO
# import pprint
import firebase_admin
from firebase_admin import credentials, firestore
# import uuid
from datetime import datetime
from google.cloud import storage



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


def get_metrics(validation):
    return {
        'map_50'    : validation.box.map50,
        'map_50:95' : validation.box.map,
        'p_50'      : validation.box.p[0],
        'r_50'      : validation.box.r[0],
        'f1_50'     : validation.box.f1[0]
    }


def download_file():

    file_id = '1WrfMHleFm7PKcFrdV8YMj7JfOip8UBGD'
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, quiet=False)
    return


def unzip_file():
    zip_file_path = './blueberryOD_640x640_669imgs.zip'
    extract_to_path = './'
    os.makedirs(extract_to_path, exist_ok=True)

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_path)
        print(f"Extracted all files to: {extract_to_path}")


def save_metrics(
    service_account_json_path : str,
    collection_name : str,
    run_params : dict, 
    metrics : dict,
    model,
    model_type
):

    experiment_id = datetime.now().strftime('%Y_%m_%d__%H_%M_%S') 


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


    metrics['experiment_id'] = experiment_id 
    metrics['methadata'] = {
        'run_params' : run_params,
        'date' : datetime.now().isoformat(),
        'total_params' : total_params,
        'trainable_params' : trainable_params,
        'model_type' : model_type,
        'best_url' : storage_best_path,
        'last_url' : storage_last_path 
    } 


    cred = credentials.Certificate(service_account_json_path) 
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred) 
    db = firestore.client()


    storage_client = storage.Client.from_service_account_json(service_account_json_path)
    bucket = storage_client.bucket('blueberry-detection-benchmark.firebasestorage.app')


    try:
        db.collection(collection_name).document(experiment_id).set(metrics)

        if os.path.exists(local_best_path):
            print(f'last pt found:{local_best_path}')
            blob_best = bucket.blob(storage_best_path) 
            blob_best.upload_from_filename(local_best_path)

        if os.path.exists(local_last_path):
            print(f'best pt found: {local_last_path}')
            blob_last = bucket.blob(storage_last_path)
            blob_last.upload_from_filename(local_last_path)

        print(f"Metrics saved for experiment_id: {metrics['experiment_id']}")
    except Exception as e:
        print(f"Failed to save metrics: {e}")
    return


def main(
    firebase_credentials_json_path : str,
    collection_name : str,
    model_type : str,
    folder_path : str,
    imgsz : int,
    epochs : int
):
    try:

        if not os.path.exists(folder_path):
            download_file()
            unzip_file()
        else:
            print(f"Folder '{folder_path}' already exists. No need to download.")

        model = YOLO(model_type)
        model.train(
            data=os.path.join(folder_path, "data.yaml"),
            epochs=epochs,
            imgsz=imgsz,
            project="blueberry-detection",
            name="run"
        )
        model.val()

        run_params = get_run_parameters(model.trainer.args)
        metrics = get_metrics(model.trainer.validator.metrics)

        save_metrics(
            service_account_json_path=firebase_credentials_json_path,
            collection_name=collection_name,
            run_params=run_params,
            metrics=metrics,
            model=model,
            model_type=model_type
        )

    except Exception as e:
        # Log and stop the code if any error occurs
        print(f"An error occurred: {e}")
        raise  # Re-raise the exception if you want it to terminate the script


if __name__ == '__main__':

    firebase_credentials_json_path = './blueberry-detection-benchmark-firebase-adminsdk-khx5o-41dd2be9b2.json'

    # this must be added using 
    folder_path = os.path.join(os.getcwd(), 'blueberryOD_640x640_669imgs')
    collection_name = 'yolov8_metrics'
    model_type = 'yolov8n.pt'
    epochs = 1
    imgsz = 640

    main(
        firebase_credentials_json_path,
        collection_name,
        model_type,
        folder_path,
        imgsz,
        epochs
    )

