import torch
import torch.nn as nn
import torch.optim as optim
from utils.options import args
from model.resnet_cifar import ResBasicBlock
import utils.common as utils
import numpy as np
from thop import profile

import os
import time
from data import cifar10
from importlib import import_module
from sklearn.cluster import AffinityPropagation
from sklearn.cluster.affinity_propagation_ import euclidean_distances
from sklearn.random_projection import SparseRandomProjection

device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'
checkpoint = utils.checkpoint(args)
logger = utils.get_logger(os.path.join(args.job_dir + 'logger.log'))
loss_func = nn.CrossEntropyLoss()

# Data
print('==> Preparing data..')
loader = cifar10.Data(args)

def cluster_weight(weight):

    A = weight.cpu().clone()
    if weight.dim() == 4:  #Convolution layer
        A = A.view(A.size(0), -1)
    else:
        raise('The weight dim must be 4!!!')

    affinity_matrix = -euclidean_distances(A, squared=True)
    preference = np.median(affinity_matrix, axis=0) * args.preference_beta
    cluster = AffinityPropagation(preference=preference)
    cluster.fit(A)
    return cluster.labels_, cluster.cluster_centers_, cluster.cluster_centers_indices_

def random_project(weight, channel_num):

    A = weight.cpu().clone()
    A = A.view(A.size(0), -1)
    rp = SparseRandomProjection(n_components=channel_num * weight.size(2) * weight.size(3))
    rp.fit(A)
    return rp.transform(A)

def get_flops_params(orimodel, prunemodel):
    input = torch.randn(1, 3, 32, 32).to(device)

    print('--------------UnPruned Model--------------')
    oriflops, oriparams = profile(orimodel, inputs=(input,))
    print('Params: %.2f' % (oriparams))
    print('FLOPS: %.2f' % (oriflops))

    print('--------------Pruned Model--------------')
    print('model\'s cfg', prunemodel.layer_cfg)
    oricfg = []
    oricfg.extend([16] * 9)
    oricfg.extend([32] * 9)
    oricfg.extend([64] * 9)
    print(oricfg)
    flops, params = profile(prunemodel, inputs=(input,))
    print('Params: %.2f' % (params))
    print('FLOPS: %.2f' % (flops))

    print('--------------Retention Ratio--------------')
    print('Params Retention Ratio: %d/%d (%.2f%%)' % (params, oriparams, 100. * params / oriparams))
    print('FLOPS Retention Ratio: %d/%d (%.2f%%)' % (flops, oriflops, 100. * flops / oriflops))

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
            centroids_state_dict[name + '.conv2.weight'] = random_project(module.conv2.weight.data, len(centroids))
            prune_state_dict.append(name + '.bn1.weight')
            prune_state_dict.append(name + '.bn1.bias')
            prune_state_dict.append(name + '.bn1.running_var')
            prune_state_dict.append(name + '.bn1.running_mean')

    model = import_module(f'model.{args.arch}_cifar').resnet(args.cfg, layer_cfg=cfg).to(device)
    get_flops_params(origin_model, model)
    if args.init_method == 'other':
        pass
    elif args.init_method == 'centroids':
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
    origin_model = import_module(f'model.{args.arch}_cifar').resnet(args.cfg).to(device)
    origin_model.load_state_dict(ckpt['state_dict'])

    cfg = []
    centroids_state_dict = {}
    prune_state_dict = []

    for name, module in origin_model.named_modules():

        if isinstance(module, nn.Conv2d):

            conv1_weight = module.weight.data
            grp, centroids = cluster_weight(conv1_weight)
            cfg.append(len(centroids))
            centroids_state_dict[name + '.weight'] = centroids
            centroids_state_dict[name + '.conv2.weight'] = random_project(module.conv2.weight.data, len(centroids))
            prune_state_dict.append(name + '.bn1.weight')
            prune_state_dict.append(name + '.bn1.bias')
            prune_state_dict.append(name + '.bn1.running_var')
            prune_state_dict.append(name + '.bn1.running_mean')

    model = import_module(f'model.{args.arch}').resnet(args.cfg, cfg).to(device)
    # get_flops_params(origin_model, model)
    if args.init_method == 'other':
        pass
    elif args.init_method == 'centroids':
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