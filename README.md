# ADD-Net: Adaptive Dynamic Architecture with Knowledge Distillation for UAV Small Object Detection

By Han Wang, Yiqing Li, and Wen Zhou, Hao Zhang

This repository contains the implementation accompanying our paper ADD-Net: Adaptive Dynamic Architecture with Knowledge Distillation for UAV Small Object Detection.

If you find this project helpful, please consider giving it a star ⭐


 We leave our system information for reference.

    python: 3.8.16
    torch: 1.13.1+cu117
    torchvision: 0.14.1+cu117
    timm: 0.9.8
    mmcv: 2.1.0
    mmengine: 0.9.0

Other operating environments    

pip install timm==0.9.8 thop efficientnet_pytorch==0.7.1 einops grad-cam==1.4.8 dill==0.3.6 albumentations==1.3.1 pytorch_wavelets==1.3.0 tidecv PyWavelets -i https://pypi.tuna.tsinghua.edu.cn/simple

## Dataset Preparation
Please construct the datasets following these steps:

- Download the datasets from their sources. 
You can download the processed xView and VisDrone-Datasets and HIT-UAV-Datasets and DroneVehicle from this Web [link](https://github.com/VisDrone/VisDrone-Dataset) and  [link](https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset). and [link](https://downloads.greyc.fr/vedai/).

- Convert the annotation files into TXT-format annotations.

- Modify the dataset path setting within the script.

```
'dateset's name': {
    'train_img'  : '',  #train image dir
    'train_Label' : '',  #train txt format label file
    'val_img'    : '',  #val image dir
    'val_label'   : '',  #val txt format label file
},
```
- Add domain adaptation direction within the script(./datasets/). During training, the domain adaptation direction will be automatically parsed and corresponding data will be loaded. In our paper, we provide four adaptation directions for remote sensing scenarios.
```

```

## Training / Evaluation / Inference
We provide training script on single node as follows.
- Training with single GPU
```
python train.py
```
- Valing with dataset
```
python val.py
```
- Hybrid Knowledge Distillation of ADD-Net
```
python distill.py
```
- get_COCO_metrice method
```
valCOCO.py
```


## Result Visualization 

![](https://github.com/Han-Wang-RSLab/ADD-Net/blob/main/ADD-Net/figs/1.png)


## Generalization Experimental

![](https://github.com/Han-Wang-RSLab/ADD-Net/blob/main/ADD-Net/figs/2.png)

