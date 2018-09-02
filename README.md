# About
This this the code repository for my final project of MSc Data Science and Machine Learning degree at UCL. It's better to check out this repo with the thesis.

# Introduction
In this project, we explore training strategies for deep architectures of Computer Vision. Especially, we focus on "ImageNet pre-training", and "multi-task learning" and "regularization" are also applied.

# Requirements
* Python 3
* PyTorch >= 0.4.0
* Numpy 

# Data
In this project, we use several datasets for semantic segmentation and human part segmentation. See and modify their paths in config.json for your convenience.

* [Pascal Voc 2012](http://host.robots.ox.ac.uk/pascal/VOC/)
* [Semantic Boundary Dataset](http://home.bharathh.info/pubs/codes/SBD/download.html)
* [Look Into Person Dataset](http://sysu-hcp.net/lip/)

# Usage
```bash
python train_imagenet.py [--arch ARCH] [--epochs N]
                         [--dataset D] [--data_portion DP]
                         [--batch_size N] [--lr LR] [--momentum M]
                         [--momentum_bn M_BN] [--weight-decay W]
                         [--pretrained] [--dprob DPB]
                         [--n_classes C] [--optim O]
                         [--manual_seed MANUALSEED]

--arch, -a            model architecture: sunet64 | fcn32resnet18 |
                      fcn32resnet 34 | fcn32resnet50 | fcn32resnet101 |   
                      fcn32resnet152 | fcn32vgg
--dataset             dataset: sbd | pascal | pascalpart | lip
--epochs              number of epochs to train
--batch-size          mini-batch size (default: 10)
--lr                  initial learning rate
--momentum            momentum
--momentum_bn         momentum of batch normalization
--wd                  weight decay (default: 1e-4)
--pretrained          use pre-trained model
--manual_seed         manual seed 
--n_classes           number of classes of ground truth annotation
--dprob               dropout probability
```