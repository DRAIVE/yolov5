from yolov5 import train
import torch
import yaml
import os

def main():
    #load params.yaml file
    params = yaml.safe_load(open("params.yaml"))["train"]
    
    config = (
        f"path: {os.path.abspath(os.curdir)}/data/ai-dataset-dvc-poc/data/OpenALPR/{chr(10)}"
        f"train: train/images{chr(10)}"
        f"val: valid/images{chr(10)}{chr(10)}"
        f"nc: 1{chr(10)}"
        f"names: ['plates']{chr(10)}")
    #train: path_to_train_images
    #val: path_to_val_images
    #nc: number of classes
    #classes: names of classes in the format - ['name']
    
    #Create model directory
    if(not os.path.exists('models')):
        os.mkdir('models')
    with open("models\data_lp.yaml", "w") as f:
        f.write(config)

    #start training
    opt = train.run(batch_size=params["batch_size"],
                    epochs=params["epochs"],
                    weights='yolov5s.pt',
                    project='runs/train',
                    name='run',
                    exist_ok=True,
                    data='models\data_lp.yaml')
    
    #load and save the trained model
    t = torch.load(f'{opt.save_dir}/weights/best.pt')
    torch.save(t, 'models/best_latest.pt')
    
if __name__ == '__main__':
    main()