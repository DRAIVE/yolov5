from yolov5 import val
from PIL import Image
import mlflow
import yaml
import json
import os

def main():
    #start mlfow
    mlflow.start_run()
    
    #load params.yaml file
    params_val = yaml.safe_load(open("params.yaml"))["validate"]
    params_train = yaml.safe_load(open("params.yaml"))["train"]

    #do validation
    #metrics = (mp, mr, map50, map, speed)
    metrics, maps, t = val.run(weights='runs/train/run4/weights/best.pt',
                  project='runs/val',
                  name='run',
                  exist_ok=True,
                  save_json=False,
                  save_txt=False,
                  data='models/data_lp.yaml',
                  conf_thres=params_val["conf_thres"],
                  iou_thres=params_val["iou_thres"])
    
    #log parameters
    mlflow.log_param("Dataset", f"{os.path.abspath(os.curdir)}/data/ai-dataset-dvc-poc/data/OpenALPR/")
    mlflow.log_param("Epochs", params_train["epochs"])
    mlflow.log_param("Batch size", params_train["batch_size"])
    mlflow.log_param("Confidence threshold", params_val["conf_thres"])
    mlflow.log_param("IoU threshold", params_val["iou_thres"])
    
    #save the validation results to json file
    val_results = {
        "mAP_IoU_.5-.95": metrics[3],
        "mAP_IoU_0.50": metrics[2],
        "Precision": metrics[0],
        "Recall": metrics[1]
    }
    #log metrics
    mlflow.log_metrics(val_results)
    
    #log plots as images
    if(os.path.exists("runs/val/run/F1_curve.png")):
        F1 = Image.open("runs/val/run/F1_curve.png")
        F1 = F1.resize((640, 430))
        mlflow.log_image(F1, "models/F1_curve_val.png")
    if(os.path.exists("runs/val/run/PR_curve.png")):
        PR = Image.open("runs/val/run/PR_curve.png")
        PR = PR.resize((640, 430))
        mlflow.log_image(PR, "models/PR_curve_val.png")
        
    #stop mlfow
    mlflow.end_run()

if __name__ == '__main__':
    main()