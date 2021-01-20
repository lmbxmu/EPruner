# Non-Parametric Adaptive Network Pruning ![]( https://visitor-badge.glitch.me/badge?page_id=lmbxmu.epruner).

<div align=center><img src="img/framework.png" height = "50%" width = "60%"/></div>


## Tips

Any problem, please contact the first author (Email: lmbxmu@stu.xmu.edu.cn, or WeChat: linmb007 if you are using it) or the third author (Email: shaojieli@stu.xmu.edu.cn). Also, you can post issues with github, but sometimes we could not receive github emails thus may ignore the posted issues (sorry if it happens).


# Pre-trained Models

We provide the pre-trained models used in our paper.

## CIFAR-10

| [VGG16]( https://drive.google.com/open?id=1sAax46mnA01qK6S_J5jFr19Qnwbl1gpm) | [ResNet56](https://drive.google.com/open?id=1pt-LgK3kI_4ViXIQWuOP0qmmQa3p2qW5) | [ResNet110](https://drive.google.com/open?id=1Uqg8_J-q2hcsmYTAlRtknCSrkXDqYDMD) |[GoogLeNet](https://drive.google.com/open?id=1YNno621EuTQTVY2cElf8YEue9J4W5BEd) | 

## ImageNet

| [ResNet18](https://download.pytorch.org/models/resnet18-5c106cde.pth) | [ResNet34](https://download.pytorch.org/models/resnet34-333f7ec4.pth) | [ResNet50](https://download.pytorch.org/models/resnet50-19c8e357.pth) | [ResNet101](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth) | [ResNet152](https://download.pytorch.org/models/resnet152-b121ed2d.pth) |

# Result Models

 We provide our pruned models in the experiments, along with their training loggers and configurations. 

## CIFAR-10

|           | Preference Beta | Inititial Method | FLOPs<br>(Prune Rate) | Params<br>(Prune Rate) | Top-1<br>Accuracy |                           Download                           |
| :-------: | :-------------: | :--------------: | :-------------------: | :--------------------: | :---------------: | :----------------------------------------------------------: |
|   VGG16   |      0.73       |    centroids     |    74.42M(76.34%)     |     1.65M(88.80%)      |      93.08%       | [Link](https://drive.google.com/open?id=1rGhurVIV4jrCMfeFlGwozZNMdhtfkDIn) |
|   VGG16   |      0.73       |      random      |    74.42M(76.34%)     |     1.65M(88.80%)      |      92.61%       | [Link](https://drive.google.com/open?id=17__j51Vjqt6QDTznwdWKnvMZoM_jgxYJ) |
|   VGG16   |      0.73       |  random_project  |    74.42M(76.34%)     |     1.65M(88.80%)      |      92.95%       | [Link](https://drive.google.com/open?id=1DVNQ-fitKq61iqvN2PX17JPitybjt8SP) |
| GoogLeNet |      0.65       |    centroids     |    500.87M(67.36%)    |     2.22M(64.20%)      |      94.99%       | [Link](https://drive.google.com/open?id=1fiDGuN2srZBc2w68B993wx0W1VxZZdH0) |
| GoogLeNet |      0.65       |      random      |    500.87M(67.36%)    |     2.22M(64.20%)      |      94.19%       | [Link](https://drive.google.com/open?id=1fNVhwq5zUnUfn2davLKJQZvPbBEPZZ-M) |
| GoogLeNet |      0.65       |  random_project  |    500.87M(67.36%)    |     2.22M(64.20%)      |      94.49%       | [Link](https://drive.google.com/open?id=1LrE_t4JhNzark1XJDunZRm8ZObmEnsCu) |
| ResNet56  |      0.76       |    centroids     |    49.35M(61.33%)     |     0.39M(54.20%)      |      93.18%       | [Link](https://drive.google.com/open?id=1vcz8N6xwj4yRoEk037eFD0-lWaZuADzN) |
| ResNet56  |      0.76       |      random      |    49.35M(61.33%)     |     0.39M(54.20%)      |      91.45%       | [Link](https://drive.google.com/open?id=1zTL6F1oRtJZyTIalWpER_01IBNdwir4t) |
| ResNet56  |      0.76       |  random_project  |    49.35M(61.33%)     |     0.39M(54.20%)      |      92.44%       | [Link](https://drive.google.com/open?id=1RqCNvQjTPrjP3-3zz0x_AG3nTHJBlI5T) |
| ResNet110 |       0.6       |    centroids     |    87.65M(65.91%)     |     0.41M(76.30%)      |      93.62%       | [Link](https://drive.google.com/open?id=1nMBi60r1JtwhXZ_036kwQawHRP4-EUNM) |
| ResNet110 |       0.6       |      random      |    87.65M(65.91%)     |     0.41M(76.30%)      |      92.44%       | [Link](https://drive.google.com/open?id=1vSMCieWH2Pi-emNwjiuMQcjAu9K0CqvQ) |
| ResNet110 |       0.6       |  random_project  |    87.65M(65.91%)     |     0.41M(76.30%)      |      93.02%       | [Link](https://drive.google.com/open?id=1SMK9PCiPU6waL68LPI3iFoDuaGx4bJ96) |

## ImageNet

|           | Preference Beta | Initial Method | FLOPs<br>(Prune Rate) | Params<br>(Prune Rate) | Top-1<br>Accuracy | Top-5<br>Accuracy |                           Download                           |
| :-------: | :-------------: | :------------: | :-------------------: | :--------------------: | :---------------: | :---------------: | :----------------------------------------------------------: |
| ResNet18  |      0.73       |   centroids    |   1024.01M(43.88%)    |     6.05M(48.52%)      |      67.31%       |      87.70%       | [Link](https://drive.google.com/open?id=1bjarsqD5czJzJ-x5Cc9sYnLajux7FHCY) |
| ResNet18  |      0.73       |     random     |   1024.01M(43.88%)    |     6.05M(48.52%)      |      66.46%       |      87.13%       | [Link](https://drive.google.com/open?id=1KFM7XvMFC0e5qsp8EpVhbFr2xHGS-aqs) |
| ResNet18  |      0.73       | random_project |   1024.01M(43.88%)    |     6.05M(48.52%)      |      66.68%       |      87.45%       | [Link](https://drive.google.com/open?id=1xpOoAmI-76QSpAes-Sgz822-OKVheY1w) |
| ResNet34  |      0.75       |   centroids    |   1853.92M(49.61%)    |     10.24M(53.24%)     |      70.95%       |      89.97%       | [Link](https://drive.google.com/open?id=1s6edp4ec4YQ74TbONd0pI0mF4L99Cz0r) |
| ResNet34  |      0.75       |     random     |   1853.92M(49.61%)    |     10.24M(53.24%)     |      70.71%       |      89.78%       | [Link](https://drive.google.com/open?id=1M8ztRs9jqoxmuHCjPpcFVLgZO3bDdwwQ) |
| ResNet34  |      0.75       | random_project |   1853.92M(49.61%)    |     10.24M(53.24%)     |      70.79%       |      89.91%       | [Link](https://drive.google.com/open?id=1HroILecs0UJE0xMbQtbUfdL2QfOPdpCB) |
| ResNet50  |      0.73       |   centroids    |   1929.15M(53.35%)    |     12.70M(50.31%)     |      74.26%       |      91.88%       | [Link](https://drive.google.com/open?id=1Swng8R9f27M0BXvD4jPwbaiHWRHsXWkO) |
| ResNet50  |      0.73       |     random     |   1929.15M(53.35%)    |     12.70M(50.31%)     |      73.54%       |      91.55%       | [Link](https://drive.google.com/open?id=1znuCEAMv2tOzXgNkdQpZAziz5P8kK31d) |
| ResNet50  |      0.73       | random_project |   1929.15M(53.35%)    |     12.70M(50.31%)     |      73.80%       |      91.83%       | [Link](https://drive.google.com/open?id=1igC-56Cw4Q9q4RUmLwlGsw1EkzsK8UWl) |
| ResNet101 |      0.67       |   centroids    |   2817.27M(64.20%)    |     15.55M(65.10%)     |      75.45%       |      92.70%       | [Link](https://drive.google.com/open?id=1DJuB_hoxc38UA8WkMeA5cHL3wjxelmHo) |
| ResNet101 |      0.67       |     random     |   2817.27M(64.20%)    |     15.55M(65.10%)     |      75.15%       |      92.25%       | [Link](https://drive.google.com/open?id=1LfUpJcc3rclO73LVuB5XI_UcCuvZ8-1S) |
| ResNet101 |      0.67       | random_project |   2817.27M(64.20%)    |     15.55M(65.10%)     |      75.31%       |      92.50%       | [Link](https://drive.google.com/open?id=1JLfdzTBlcFrfGAQ59q8ccHGbdAhlG4g_) |
| ResNet152 |      0.63       |   centroids    |   4047.69M(65.12%)    |     21.56M(64.18%)     |      76.51%       |      93.22%       | [Link](https://drive.google.com/open?id=1FgOJHocaIujKgsrEx63Vg-h1gD3cr9sD) |
| ResNet152 |      0.63       |     random     |   4047.69M(65.12%)    |     21.56M(64.18%)     |      76.15%       |      92.97%       | [Link](https://drive.google.com/open?id=1rjgkEAF19CeCQdt298rhmoL6GsQzISXw) |
| ResNet152 |      0.63       | random_project |   4047.69M(65.12%)    |     21.56M(64.18%)     |      76.43%       |      93.14%       | [Link](https://drive.google.com/open?id=1oIsDp28cdZOXLg-A_IIagQ5Nfly4cnEP) |
| ResNet50  |      0.71       |   centroids    |   2366.80M(42.77%)    |     21.98M(13.99%)     |      74.95%       |         -         | [Link](https://drive.google.com/open?id=1WxCaVliGiHJANqiH-lLt4DPvhjmzPt-e) |
| ResNet50  |      0.81       |   centroids    |   1290.35M(68.63%)    |     14.78M(42.15%)     |      72.73%       |         -         | [Link](https://drive.google.com/open?id=1ZvmqG8k6-APJJNq1SMHz0lertgB6Ht19) |
| ResNet50  |      0.85       |   centroids    |    905.89M(78.10%)    |     8.65M(66.15%)      |      70.34%       |         -         | [Link](https://drive.google.com/open?id=1nVOYAWsJGbxwb4fNXIlDcWWwNEBNCPC5) |

# Running Code

The code has been tested using Pytorch1.3 and CUDA10.0 on Ubuntu16.04.

Requirements: Sklearn 0.20.1


## EPruner

You can run the following code to search model on CIFAR-10:

```shell
python epruner_cifar.py 
--dataset cifar10 
--data_path /data/CIFAR10/ 
--pretrain_model /data/model/resnet56.pt 
--job_dir /data/experiment/resnet56 
--arch resnet 
--cfg resnet56 
--init_method centroids 
--preference_beta 0.76 
--lr 0.01 
--lr_decay_step 50 100 
--num_epochs 150 
--train_batch_size 256 
--weight_decay 5e-3 
--gpus 0
```

 You can run the following code to search model on ImageNet: 

```shell
python epruner_imagenet.py 
--dataset imagenet 
--data_path /data/ImageNet/ 
--pretrain_model /data/model/resnet50.pth 
--job_dir /data/experiment/resnet50 
--arch resnet 
--cfg resnet50 
--init_method centroids 
--preference_beta 0.75 
--lr 0.1 
--lr_decay_step 30 60 
--num_epochs 90 
--train_batch_size 256 
--weight_decay 1e-4 
--gpus 0 1 2 
```

## Test Our Performance

Before you testing our model, please use the following command to install the thop python package which can calculate the flops and params of model:

```shell
pip install thop
```

 Follow the command below to verify our pruned models: 

```shell
python test_flops_params.py
--dataset cifar10 
--data_path /data/CIFAR10 
--arch resnet 
--cfg resnet56
--pruned_model /data/model/pruned_model/resnet56/model_best.pt 
--eval_batch_size 100
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
                        Batch size for training. default:256
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
                        AffinityPropagation cluster. default:0.75
```
