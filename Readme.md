# Model Training Logging

## 1. Model Training & Logging
* Trained with regnetz_c16 and logs it
* logs f1 score, precision, recall, confusion-matrix

**Tensorboard dev**
```
$ tensorboard dev upload --logdir logs \
    --name "My latest experiment - Model Training" \
    --description " Logging of different metrics, training on regnetz_c16"
```

**Tensorboard dev: https://tensorboard.dev/experiment/HMtbAgckTjmz1EWi3mJelQ/**

**Hyperparameters in `hparams.yaml`**
```
learning_rate: 1.2e-05
lr: 1.2e-05
model_name: regnetz_c16
num_classes: 6
optimizer_name: ADAM
```

For augmentation of Images I use random transformation like degree rotation, contrast increase and so on. It is detailed on following code:
```python
import torchvision.transforms as T

transforms1 = T.RandomApply(
            [
                T.RandomRotation(degrees=(0, 70)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=(0.1, 0.6), contrast=1, saturation=0, hue=0.3),
                T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                T.RandomHorizontalFlip(p=0.3),
            ], 
            p=0.3
        )
transforms = T.Compose([
                transforms1,
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
```

For confusion matrix in tensorboard logs [**Image not uploaded to tensorboard dev experiment**]:
![](files/confusion_matrix_1.gif)

### 2 - Optuna Hparam Search
**Tensorboard dev:   https://tensorboard.dev/experiment/IE1yultOSAKjX0XPJUVwlQ/**

```python
Trail with : 

lr_rate:0.00011006008295331135 model name: resnet18 optimizer name: RMS
=========================================
Trail with : 

lr_rate:0.0003880858334105841 model name: resnet18 optimizer name: ADAM
=========================================
Study statistics: 
  Number of finished trials:  2
  Number of pruned trials:  0
  Number of complete trials:  2
Number of finished trials: 2
Best trial:
  Value: 0.921875
Trail with : 
  Params: 
    lr: 0.0003880858334105841
    model_name: resnet18
    optimizer: ADAM


Trail with : 

lr_rate:1.547691630918372e-05 model name: efficientnet_b0 optimizer name: SGD
=========================================
Trail with : 

lr_rate:0.00027665284312550367 model name: efficientnet_b0 optimizer name: RMS
=========================================

Study statistics: 
  Number of finished trials:  2
  Number of pruned trials:  0
  Number of complete trials:  2
Number of finished trials: 2
Best trial:
  Value: 0.734375
Trail with : 
  Params: 
    lr: 0.00027665284312550367
    model_name: efficientnet_b0
    optimizer: RMS
    
    
Trail with : 

lr_rate:2.1172144867677624e-05 model name: regnetz_c16 optimizer name: SGD
=========================================
Trail with : 

lr_rate:0.00012310173574776534 model name: regnetz_c16 optimizer name: ADAM
=========================================

Study statistics: 
  Number of finished trials:  2
  Number of pruned trials:  0
  Number of complete trials:  2
Number of finished trials: 2
Best trial:
  Value: 0.83203125
  
Params: 
    lr: 0.00012310173574776534
    model_name: regnetz_c16
    optimizer: ADAM
```

