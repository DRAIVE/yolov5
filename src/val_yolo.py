from yolov5 import val
from PIL import Image
import mlflow
import yaml
import os

def main():
    mlflow.set_tracking_uri("http://jensen.etit.tu-chemnitz.de:9040")
    mlflow.set_experiment("License plate detection using YOLOv5")
    
    #load params.yaml file
    params_val = yaml.safe_load(open("params.yaml"))["validate"]
    params_train = yaml.safe_load(open("params.yaml"))["train"]
    params_dl = yaml.safe_load(open("params.yaml"))["download"]
    
  
    #start mlfow
    mlflow.start_run(
        description=f'Dataset in LakeFS:{params_dl["url"]}/repositories/{params_dl["bucket"]}/objects?path={params_dl["path"]}&ref={params_dl["commit"]}'
    )

    #do validation
    #metrics = (mp, mr, map50, map, speed)
    metrics, maps, t = val.run(weights='models\data_lp.yaml',
                  project='runs/val',
                  name='run',
                  exist_ok=True,
                  save_json=False,
                  save_txt=False,
                  data='models/data_lp.yaml',
                  conf_thres=params_val["conf_thres"],
                  iou_thres=params_val["iou_thres"])
    
    #log parameters
    mlflow.log_param("Training epochs", params_train["epochs"])
    mlflow.log_param("Training Batch size", params_train["batch_size"])
    mlflow.log_param("Confidence threshold", params_val["conf_thres"])
    mlflow.log_param("IoU threshold", params_val["iou_thres"])
    
    #validation results
    val_results = {
        "mAP_IoU_.5-.95": metrics[3],
        "mAP_IoU_0.50": metrics[2],
        "Precision": metrics[0],
        "Recall": metrics[1]
    }
    #log metrics
    mlflow.log_metrics(val_results)
    
    #log plots 
    if(os.path.exists("runs/train/run/results.png")):
        train_results = Image.open("runs/train/run/results.png")
        train_results = train_results.resize((1280, 720))
        mlflow.log_image(train_results,"models/train_results.png")
    if(os.path.exists("runs/val/run/F1_curve.png")):
        F1 = Image.open("runs/val/run/F1_curve.png")
        F1 = F1.resize((740, 590))
        mlflow.log_image(F1, "models/F1.png")
    if(os.path.exists("runs/val/run/PR_curve.png")):
        PR = Image.open("runs/val/run/PR_curve.png")
        PR = PR.resize((740, 590))
        mlflow.log_image(PR, "models/PR.png")
        
    #stop mlfow
    mlflow.end_run()

if __name__ == '__main__':
    main()