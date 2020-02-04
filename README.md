# EBS-Rasnet50
Epoxy Best Setup image classification python3 application with Rasnet50 on tensorflow keras

## Proceduce

1. Unzip dataset for folder of EBS images both training and testing

2. python training.py ## To start training model

3. python validate.py ## To validate single image

## Experiments of training with result of image P2-002.jpg 

### 1. bs = 16, lr = 2.5e-4, epoch = 100, pre-train None, P2-002 = 99%   ok

### 2. bs = 16, lr = 2.5e-4, epoch = 100, pre-train 'imagenet', P2-002 = 99%   ok

### 3. bs = 8, lr = 2.5e-4, epoch = 100, pre-train 'imagenet', P2-002 = 99%   ok

4. bs = 20, lr = 2.5e-4, epoch = 100, pre-train 'imagenet', P2-002 = 86%   fair

5. bs = 32, lr = 2.5e-4, epoch = 100, pre-train 'imagenet', P2-002 = 24%   failed

6. bs = 16, lr = 2.5e-3, epoch = 100, pre-train 'imagenet', P2-002 = 2%   failed

7. bs = 16, lr = 2.5e-3, epoch = 150, pre-train 'imagenet', P2-002 = 2%  failed

8. bs = 16, lr = 2.5e-3, epoch = 200, pre-train 'imagenet', P2-002 = 1%  failed

## Cheer
