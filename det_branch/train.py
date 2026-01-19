import argparse
from ultralytics import YOLO
import random
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--single-cls', action="store_true", default=False, help='One class only')
    parser.add_argument('--data', type=str, default='ultralytics/cfg/datasets/polyp_2.yaml', help='data path')
    parser.add_argument('--model', type=str, default='n', help='n, s, l, x')
    parser.add_argument('--name', type=str, default='test', help='Experiment Name')
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=300)
    
    arg = parser.parse_args()
    
    random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # model = YOLO(f'/nfs/P111yhchen/code/detection/det_branch/pretrained/{arg.model}.pt')
    model = YOLO(f'{arg.model}.yaml')
    if arg.model[-1]=='n':
        scale = 0.5
        mixup = 0.0
        copy_paste=0.1
    elif arg.model[-1]=='s':
        scale = 0.9
        mixup = 0.05
        copy_paste=0.15
    elif arg.model[-1]=='l':
        scale = 0.9
        mixup = 0.15
        copy_paste=0.5
    else:
        scale = 0.9
        mixup = 0.2
        copy_paste=0.6
        
    # Train the model
    results = model.train(
      data=arg.data,
      name=arg.name,
      single_cls=arg.single_cls,
      epochs=arg.epochs, 
      batch=arg.batch, 
      imgsz=640,
      scale=scale,  # S:0.9; L:0.9; X:0.9
      mosaic=1.0,
      mixup=mixup,  # S:0.05; L:0.15; X:0.2
      copy_paste=copy_paste,  # S:0.15; L:0.5; X:0.6
      device="0",
    )
    
    # Evaluate model performance on the validation set
    metrics = model.val(
        data=arg.data,
        single_cls=arg.single_cls,
    )
