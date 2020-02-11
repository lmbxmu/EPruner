import torch
import torch.nn as nn
import torch.optim as optim
from utils.options import args
from model.resnet_cifar import ResBasicBlock
from model.googlenet import Inception
import utils.common as utils

import os
import time
from data import cifar10
from importlib import import_module
from utils.common import cluster_weight, random_project, direct_project

device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'
checkpoint = utils.checkpoint(args)
logger = utils.get_logger(os.path.join(args.job_dir + 'logger.log'))
loss_func = nn.CrossEntropyLoss()

# Data
print('==> Preparing data..')
loader = cifar10.Data(args)

def cluster_resnet():

    if args.pretrain_model is None or not os.path.exists(args.pretrain_model):
        raise ('pretrain model path should be exist!')
    ckpt = torch.load(args.pretrain_model, map_location=device)
    origin_model = import_module(f'model.{args.arch}_cifar').resnet(args.cfg).to(device)
    origin_model.load_state_dict(ckpt['state_dict'])

    cfg = []
    centroids_state_dict = {}
    prune_state_dict = []

    for name, module in origin_model.named_modules():

        if isinstance(module, ResBasicBlock):

            conv1_weight = module.conv1.weight.data
            _, centroids, indices = cluster_weight(conv1_weight)
            cfg.append(len(centroids))
            centroids_state_dict[name + '.conv1.weight'] = centroids
            if args.init_method == 'random_project':
                centroids_state_dict[name + '.conv2.weight'] = random_project(module.conv2.weight.data, len(centroids))
            else:
                centroids_state_dict[name + '.conv2.weight'] = direct_project(module.conv2.weight.data, indices)

            prune_state_dict.append(name + '.bn1.weight')
            prune_state_dict.append(name + '.bn1.bias')
            prune_state_dict.append(name + '.bn1.running_var')
            prune_state_dict.append(name + '.bn1.running_mean')

    model = import_module(f'model.{args.arch}_cifar').resnet(args.cfg, layer_cfg=cfg).to(device)
    # get_flops_params(origin_model, model, logger)
    if args.init_method == 'random_project' or args.init_method == 'centroids':
        pretrain_state_dict = origin_model.state_dict()
        state_dict = model.state_dict()
        centroids_state_dict_keys = list(centroids_state_dict.keys())
        for k, v in state_dict.items():
            if k in prune_state_dict:
                continue
            elif k in centroids_state_dict_keys:
                state_dict[k] = torch.FloatTensor(centroids_state_dict[k]).view_as(state_dict[k])
            else:
                state_dict[k] = pretrain_state_dict[k]
        model.load_state_dict(state_dict)
    else:
        pass
    return model, cfg

def cluster_vgg():

    if args.pretrain_model is None or not os.path.exists(args.pretrain_model):
        raise ('pretrain model path should be exist!')
    ckpt = torch.load(args.pretrain_model, map_location=device)
    origin_model = import_module(f'model.{args.arch}_cifar').VGG(args.cfg).to(device)
    origin_model.load_state_dict(ckpt['state_dict'])

    cfg = []
    centroids_state_dict = {}
    prune_state_dict = []
    indices = []

    for name, module in origin_model.named_modules():

        if isinstance(module, nn.Conv2d):

            conv_weight = module.weight.data
            _, centroids, indice = cluster_weight(conv_weight)
            cfg.append(len(centroids))
            indices.append(indice)
            # indices[name + '.weight'] = indice
            centroids_state_dict[name + '.weight'] = centroids.reshape((-1, conv_weight.size(1), conv_weight.size(2), conv_weight.size(3)))
            prune_state_dict.append(name + '.bias')

        elif isinstance(module, nn.BatchNorm2d):
            prune_state_dict.append(name + '.weight')
            prune_state_dict.append(name + '.bias')
            prune_state_dict.append(name + '.running_var')
            prune_state_dict.append(name + '.running_mean')

        elif isinstance(module, nn.Linear):
            prune_state_dict.append(name + '.weight')
            prune_state_dict.append(name + '.bias')

    model = import_module(f'model.{args.arch}_cifar').VGG(args.cfg, layer_cfg=cfg).to(device)
    # get_flops_params(origin_model, model, logger)

    if args.init_method == 'random_project' or args.init_method == 'centroids':
        pretrain_state_dict = origin_model.state_dict()
        state_dict = model.state_dict()
        centroids_state_dict_keys = list(centroids_state_dict.keys())

        for i, (k, v) in enumerate(centroids_state_dict.items()):
            if i == 0: #first conv need not to prune channel
                continue
            if args.init_method == 'random_project':
                centroids_state_dict[k] = random_project(torch.FloatTensor(centroids_state_dict[k]),
                                                         len(indices[i - 1]))
            else:
                centroids_state_dict[k] = direct_project(torch.FloatTensor(centroids_state_dict[k]), indices[i - 1])

        for k, v in state_dict.items():
            if k in prune_state_dict:
                continue
            elif k in centroids_state_dict_keys:
                state_dict[k] = torch.FloatTensor(centroids_state_dict[k]).view_as(state_dict[k])
            else:
                state_dict[k] = pretrain_state_dict[k]
        model.load_state_dict(state_dict)
    else:
        pass
    return model, cfg

def cluster_googlenet():

    if args.pretrain_model is None or not os.path.exists(args.pretrain_model):
        raise ('pretrain model path should be exist!')
    ckpt = torch.load(args.pretrain_model, map_location=device)
    origin_model = import_module(f'model.{args.arch}').googlenet().to(device)
    origin_model.load_state_dict(ckpt['state_dict'])

    cfg = []
    centroids_state_dict = {}
    prune_state_dict = []
    indices = []

    for name, module in origin_model.named_modules():

        if isinstance(module, Inception):

            branch3_weight = module.branch3x3[0].weight.data
            _, centroids, indice = cluster_weight(branch3_weight)
            cfg.append(len(centroids))
            centroids_state_dict[name + '.branch3x3.0.weight'] = centroids
            if args.init_method == 'random_project':
                centroids_state_dict[name + '.branch3x3.3.weight'] = random_project(module.branch3x3[3].weight.data, len(centroids))
            else:
                centroids_state_dict[name + '.branch3x3.3.weight'] = direct_project(module.branch3x3[3].weight.data, indice)

            prune_state_dict.append(name + '.branch3x3.0.bias')
            prune_state_dict.append(name + '.branch3x3.1.weight')
            prune_state_dict.append(name + '.branch3x3.1.bias')
            prune_state_dict.append(name + '.branch3x3.1.running_var')
            prune_state_dict.append(name + '.branch3x3.1.running_mean')

            branch5_weight1 = module.branch5x5[0].weight.data
            _, centroids, indice = cluster_weight(branch5_weight1)
            cfg.append(len(centroids))
            indices.append(indice)
            centroids_state_dict[name + '.branch5x5.0.weight'] = centroids

            prune_state_dict.append(name + '.branch5x5.0.bias')
            prune_state_dict.append(name + '.branch5x5.1.weight')
            prune_state_dict.append(name + '.branch5x5.1.bias')
            prune_state_dict.append(name + '.branch5x5.1.running_var')
            prune_state_dict.append(name + '.branch5x5.1.running_mean')

            branch5_weight2 = module.branch5x5[3].weight.data
            _, centroids, indice = cluster_weight(branch5_weight2)
            cfg.append(len(centroids))
            centroids_state_dict[name + '.branch5x5.3.weight'] = centroids.reshape((-1, branch5_weight2.size(1), branch5_weight2.size(2), branch5_weight2.size(3)))

            if args.init_method == 'random_project':
                centroids_state_dict[name + '.branch5x5.6.weight'] = random_project(module.branch5x5[6].weight.data, len(centroids))
            else:
                centroids_state_dict[name + '.branch5x5.6.weight'] = direct_project(module.branch5x5[6].weight.data, indice)

            prune_state_dict.append(name + '.branch5x5.3.bias')
            prune_state_dict.append(name + '.branch5x5.4.weight')
            prune_state_dict.append(name + '.branch5x5.4.bias')
            prune_state_dict.append(name + '.branch5x5.4.running_var')
            prune_state_dict.append(name + '.branch5x5.4.running_mean')

    model = import_module(f'model.{args.arch}').googlenet(layer_cfg=cfg).to(device)
    # get_flops_params(origin_model, model, logger)
    if args.init_method == 'random_project' or args.init_method == 'centroids':
        pretrain_state_dict = origin_model.state_dict()
        state_dict = model.state_dict()
        centroids_state_dict_keys = list(centroids_state_dict.keys())

        index = 0
        for k, v in centroids_state_dict.items():

            if k.endswith('.branch5x5.3.weight'):
                if args.init_method == 'random_project':
                    centroids_state_dict[k] = random_project(torch.FloatTensor(centroids_state_dict[k]),
                                                             len(indices[index]))
                else:
                    centroids_state_dict[k] = direct_project(torch.FloatTensor(centroids_state_dict[k]), indices[index])
                index += 1

        for k, v in state_dict.items():
            if k in prune_state_dict:
                continue
            elif k in centroids_state_dict_keys:
                state_dict[k] = torch.FloatTensor(centroids_state_dict[k]).view_as(state_dict[k])
            else:
                state_dict[k] = pretrain_state_dict[k]
        model.load_state_dict(state_dict)
    else:
        pass
    return model, cfg

def train(model, optimizer, trainLoader, args, epoch, topk=(1,)):

    model.train()
    losses = utils.AverageMeter()
    accuracy = utils.AverageMeter()
    top5_accuracy = utils.AverageMeter()
    print_freq = len(trainLoader.dataset) // args.train_batch_size // 10
    start_time = time.time()
    for batch, (inputs, targets) in enumerate(trainLoader):

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = loss_func(output, targets)
        loss.backward()
        losses.update(loss.item(), inputs.size(0))
        optimizer.step()

        prec1 = utils.accuracy(output, targets, topk=topk)
        accuracy.update(prec1[0], inputs.size(0))
        if len(topk) == 2:
            top5_accuracy.update(prec1[1], inputs.size(0))

        if batch % print_freq == 0 and batch != 0:
            current_time = time.time()
            cost_time = current_time - start_time
            if len(topk) == 1:
                logger.info(
                    'Epoch[{}] ({}/{}):\t'
                    'Loss {:.4f}\t'
                    'Accuracy {:.2f}%\t\t'
                    'Time {:.2f}s'.format(
                        epoch, batch * args.train_batch_size, len(trainLoader.dataset),
                        float(losses.avg), float(accuracy.avg), cost_time
                    )
                )
            else:
                logger.info(
                    'Epoch[{}] ({}/{}):\t'
                    'Loss {:.4f}\t'
                    'Top1 {:.2f}%\t'
                    'Top5 {:.2f}%\t'
                    'Time {:.2f}s'.format(
                        epoch, batch * args.train_batch_size, len(trainLoader.dataset),
                        float(losses.avg), float(accuracy.avg), float(top5_accuracy.avg), cost_time
                    )
                )
            start_time = current_time

def test(model, testLoader, topk=(1,)):
    model.eval()

    losses = utils.AverageMeter()
    accuracy = utils.AverageMeter()
    top5_accuracy = utils.AverageMeter()

    start_time = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testLoader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_func(outputs, targets)

            losses.update(loss.item(), inputs.size(0))
            predicted = utils.accuracy(outputs, targets, topk=topk)
            accuracy.update(predicted[0], inputs.size(0))
            if len(topk) == 2:
                top5_accuracy.update(predicted[1], inputs.size(0))

        current_time = time.time()
        if len(topk) == 1:
            logger.info(
                'Test Loss {:.4f}\tAccuracy {:.2f}%\t\tTime {:.2f}s\n'
                .format(float(losses.avg), float(accuracy.avg), (current_time - start_time))
            )
        else:
            logger.info(
                'Test Loss {:.4f}\tTop1 {:.2f}%\tTop5 {:.2f}%\tTime {:.2f}s\n'
                    .format(float(losses.avg), float(accuracy.avg), float(top5_accuracy.avg), (current_time - start_time))
            )
    if len(topk) == 1:
        return accuracy.avg
    else:
        return top5_accuracy.avg

def main():
    start_epoch = 0
    best_acc = 0.0

    # Model
    print('==> Building model..')
    if args.arch == 'resnet':
        model, cfg = cluster_resnet()
    elif args.arch == 'vgg':
        model, cfg = cluster_vgg()
    elif args.arch == 'googlenet':
        model, cfg = cluster_googlenet()
    else:
        raise('arch not exist!')
    print('==>Search Done!')

    if len(args.gpus) != 1:
        model = nn.DataParallel(model, device_ids=args.gpus)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_decay_step, gamma=0.1)


    for epoch in range(start_epoch, args.num_epochs):
        train(model, optimizer, loader.trainLoader, args, epoch, topk=(1, 5) if args.dataset == 'imagenet' else (1, ))
        scheduler.step()
        test_acc = test(model, loader.testLoader, topk=(1, 5) if args.dataset == 'imagenet' else (1, ))

        is_best = best_acc < test_acc
        best_acc = max(best_acc, test_acc)

        model_state_dict = model.module.state_dict() if len(args.gpus) > 1 else model.state_dict()

        state = {
            'state_dict': model_state_dict,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch + 1,
            'arch': args.cfg,
            'cfg': cfg
        }
        checkpoint.save_model(state, epoch + 1, is_best)

    logger.info('Best accuracy: {:.3f}'.format(float(best_acc)))

if __name__ == '__main__':
    main()