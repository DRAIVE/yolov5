from yolov5 import val
import yaml
import json
import os

def main():
    #load params.yaml file
    params = yaml.safe_load(open("params.yaml"))["validate"]

    #do validation
    #metrics = (mp, mr, map50, map, speed)
    metrics, maps, t = val.run(weights='runs/train/run/weights/best.pt',
                  project='runs/val',
                  name='run',
                  exist_ok=False,
                  save_json=False,
                  save_txt=False,
                  data='models/data_lp.yaml',
                  conf_thres=params["conf_thres"],
                  iou_thres=params["iou_thres"])
    
    #save the validation results to json file
    val_results = {
        "mAP @ IoU [.5,.95]": metrics[3],
        "mAP @ IoU 0.50": metrics[2],
        "Precision": metrics[0],
        "Recall": metrics[1]
    }
    if(not os.path.exists('dvc_metrics')):
        os.mkdir('dvc_metrics')
    with open("dvc_metrics/val_results.json", "w") as f:
        json.dump(val_results,f)
    
    #The output plots are not created if predictions are not there
    #temporary work around
    if(not os.path.exists("runs/val/run/best_predictions.json")):
        open("runs/val/run/best_predictions.json", "x")
    if(not os.path.exists("runs/val/run/F1_curve.png")):
        open("runs/val/run/F1_curve.png", "x")
    if(not os.path.exists("runs/val/run/PR_curve.png")):
        open("runs/val/run/PR_curve.png", "x")

if __name__ == '__main__':
    main()