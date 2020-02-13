# Pre-trained Models

We provide the pre-trained models used in our paper.

## CIFAR-10

| [VGG16]( https://drive.google.com/open?id=1pz-_0CCdL-1psIQ545uJ3xT6S_AAnqet) | [ResNet56](https://drive.google.com/open?id=1pt-LgK3kI_4ViXIQWuOP0qmmQa3p2qW5) | [ResNet110](https://drive.google.com/open?id=1Uqg8_J-q2hcsmYTAlRtknCSrkXDqYDMD) |[GoogLeNet](https://drive.google.com/open?id=1YNno621EuTQTVY2cElf8YEue9J4W5BEd) | 

## ImageNet

| [ResNet18](https://download.pytorch.org/models/resnet18-5c106cde.pth) | [ResNet34](https://download.pytorch.org/models/resnet34-333f7ec4.pth) | [ResNet50](https://download.pytorch.org/models/resnet50-19c8e357.pth) | [ResNet101](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth) | [ResNet152](https://download.pytorch.org/models/resnet152-b121ed2d.pth) |

# Running Code

The code has been tested using Pytorch1.3 and CUDA10.0 on Ubuntu16.04.

Requirements: Sklearn 0.20.1


## APPruner

You can run the following code to search model on CIFAR-10:

```shell
python appruner_cifar10.py 
--dataset cifar10 
--data_path ../data/cifar10/
--pretrain_model ./experiment/pretrain/resne56.pt 
--job_dir ./experiment/resnet56/
--arch resnet 
--cfg resnet56 
--init_method centroids
--preference_beta 0.45
--lr 0.01
--lr_decay_step 50 100
--num_epochs 150 
--gpus 0
```

 You can run the following code to search model on Imagenet: 

```shell
python appruner_imagenet.py 
--dataset imagenet 
--data_path ../data/imagenet/
--sketch_model ./experiment/pretrain/resne50.pth 
--job_dir ./experiment/resnet50/
--arch resnet 
--cfg resnet50 
--init_method centroids
--preference_beta 0.45
--lr 0.1
--lr_decay_step 30 60
--num_epochs 90 
--gpus 0
```

## Other Arguments

```shell
optional arguments:
  -h, --help            show this help message and exit
  --gpus GPUS [GPUS ...]
                        Select gpu_id to use. default:[0]
  --dataset DATASET     Select dataset to train. default:cifar10
  --data_path DATA_PATH
                        The dictionary where the input is stored.
                        default:/home/lishaojie/data/cifar10/
  --job_dir JOB_DIR     The directory where the summaries will be stored.
                        default:./experiments
  --arch ARCH           Architecture of model. default:resnet
  --cfg CFG             Detail architecuture of model. default:resnet56
  --num_epochs NUM_EPOCHS
                        The num of epochs to train. default:150
  --train_batch_size TRAIN_BATCH_SIZE
                        Batch size for training. default:128
  --eval_batch_size EVAL_BATCH_SIZE
                        Batch size for validation. default:100
  --momentum MOMENTUM   Momentum for MomentumOptimizer. default:0.9
  --lr LR               Learning rate for train. default:1e-2
  --lr_decay_step LR_DECAY_STEP [LR_DECAY_STEP ...]
                        the iterval of learn rate. default:50, 100
  --weight_decay WEIGHT_DECAY
                        The weight decay of loss. default:5e-4
  --pretrain_model PRETRAIN_MODEL
                        Path to the pretrain model . default:None
  --init_method INIT_METHOD
                        Initital method of pruned model. default:centroids.
                        optimal:random,centroids,random_project
  --preference_beta PREFERENCE_BETA
                        The coefficient of preference used in
                        AffinityPropagation cluster. default:0.45
  --weight_norm_method WEIGHT_NORM_METHOD
                        Select the weight norm method. default:None
                        Optional:l2
```

## Tips

Any problem, free to contact the authors (lmbxmu@stu.xmu.edu.cn or shaojieli@stu.xmu.edu.cn).