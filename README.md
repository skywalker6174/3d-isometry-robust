# [CVPR2020] On Isometry Robustness of Deep 3D Point Cloud Models under Adversarial Attacks
## Environment
Ubuntu 16.04.5 LTS  
GPU RTX2080ti  
Python 3.7  
Install the python dependencies with  
```bash
pip install -r requirements.txt
```

## Data
**ModelNet40** automatically downloaded  
**ShapeNet** [/fxia22/pointnet.pytorch](https://github.com/fxia22/pointnet.pytorch) (follow the guidence for downloading)  
The default path of data is '/data'.  

## Usage Sample
**Train model** With default parameters setting, run 
```bash
python train.py --data modelnet40 --model pointnet
```
**Launch attack** If you don't want to retrain the model, download a trained model [here](https://drive.google.com/file/d/1bQSIyTjVl4DAdMGQtLbySdfG8TCeMLpu/view?usp=sharing) (with ModelNet40 data, PointNet model), move it to '/checkpoints', then run
```bash
python ctri_attack.py --data modelnet40 --model pointnet --model_path 'example'
```

## References
PointNet  [/charlesq34/pointnet](https://github.com/charlesq34/pointnet), [/fxia22/pointnet.pytorch](https://github.com/fxia22/pointnet.pytorch)    
PointNet++  [/charlesq34/pointnet2](https://github.com/charlesq34/pointnet2), [/yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)  
DG-CNN  [/WangYueFt/dgcnn](https://github.com/WangYueFt/dgcnn)  
RS-CNN  [/Yochengliu/Relation-Shape-CNN](https://github.com/Yochengliu/Relation-Shape-CNN)  
Thompson Sampling  [/andrecianflone/thompson](https://github.com/andrecianflone/thompson)  
Adversarial Attacks [/MadryLab](https://github.com/MadryLab), [/YyzHarry/ME-Net](https://github.com/YyzHarry/ME-Net)   
