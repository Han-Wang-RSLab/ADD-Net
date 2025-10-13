import warnings
warnings.filterwarnings('ignore')
import argparse, yaml, copy
from ultralytics.models.rtdetr.distill import RTDETRDistiller

if __name__ == '__main__':
    param_dict = {
        # origin
        'model': 'ultralytics\cfg\models\rt-detr\ADD-Net.yaml',
        'data':'/root/code/dataset/Visdrone/data.yaml',
        'imgsz': 640,
        'epochs': 300,
        'batch': 8,
        'workers': 4,
        'cache': False,
        # 'device': '0,1', 
        'project':'runs/distill',
        'name':'rtdetr-logloss',
        
        # distill
        'prune_model': False,
        'teacher_weights': 'runs/train/yolov8n-detr/weights/best.pt',
        'teacher_cfg': 'ultralytics/cfg/models/yolo-detr/yolov8-detr.yaml',
        'kd_loss_type': 'logical',
        'kd_loss_decay': 'constant',
        'kd_loss_epoch': 1.0,
        
        'logical_loss_type': 'logical', 
        'logical_loss_ratio': 0.6,
        
        'teacher_kd_layers': '5,10,12,19,22,25',
        'student_kd_layers': '5,10,12,19,22,25',
        'feature_loss_type': 'cwd', 
        'feature_loss_ratio': 0.6
    }
    
    model = RTDETRDistiller(overrides=param_dict)
    model.distill()