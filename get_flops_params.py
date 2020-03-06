import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from utils.options import args
from model.resnet_cifar import ResBasicBlock
from model.resnet_imagenet import BasicBlock, Bottleneck
from model.googlenet import Inception

import os
import time
from thop import profile
import seaborn as sns
from importlib import import_module
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from utils.common import cluster_weight, random_project, direct_project

device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'

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
            label, centroids, indice = cluster_weight(conv1_weight)
            cfg.append(len(centroids))
            # plot_tsne_embedding(conv1_weight.cpu().view(conv1_weight.size(0), -1), label, indice, '', plot_name=name+'.conv1.weight')
            plot_pca_embedding(conv1_weight.cpu().view(conv1_weight.size(0), -1), label, indice, '', plot_name=name+'.conv1.weight')
            centroids_state_dict[name + '.conv1.weight'] = centroids
            if args.init_method == 'random_project':
                centroids_state_dict[name + '.conv2.weight'] = random_project(module.conv2.weight.data, len(centroids))
            else:
                centroids_state_dict[name + '.conv2.weight'] = direct_project(module.conv2.weight.data, indice)

            prune_state_dict.append(name + '.bn1.weight')
            prune_state_dict.append(name + '.bn1.bias')
            prune_state_dict.append(name + '.bn1.running_var')
            prune_state_dict.append(name + '.bn1.running_mean')

    model = import_module(f'model.{args.arch}_cifar').resnet(args.cfg, layer_cfg=cfg).to(device)
    get_flops_params(origin_model, model)
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

    start_alpha_layer_index = 8
    current_conv_layer_index = 0

    for name, module in origin_model.named_modules():

        if isinstance(module, nn.Conv2d):

            conv_weight = module.weight.data
            if current_conv_layer_index >= start_alpha_layer_index:
                label, centroids, indice = cluster_weight(conv_weight, beta=0.92)
            else:
                label, centroids, indice = cluster_weight(conv_weight)

            cfg.append(len(centroids))
            # plot_tsne_embedding(conv_weight.cpu().view(conv_weight.size(0), -1), label, indice, '', plot_name=name)
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
    get_flops_params(origin_model, model)

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
            label, centroids, indice = cluster_weight(branch3_weight)
            # plot_tsne_embedding(branch3_weight.cpu().view(branch3_weight.size(0), -1), label, indice, '', plot_name=name+'branch3.weight')
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
            label, centroids, indice = cluster_weight(branch5_weight1)
            cfg.append(len(centroids))
            # plot_tsne_embedding(branch5_weight1.cpu().view(branch5_weight1.size(0), -1), label, indice, '', plot_name=name+'branch5.weight1')
            indices.append(indice)
            centroids_state_dict[name + '.branch5x5.0.weight'] = centroids

            prune_state_dict.append(name + '.branch5x5.0.bias')
            prune_state_dict.append(name + '.branch5x5.1.weight')
            prune_state_dict.append(name + '.branch5x5.1.bias')
            prune_state_dict.append(name + '.branch5x5.1.running_var')
            prune_state_dict.append(name + '.branch5x5.1.running_mean')

            branch5_weight2 = module.branch5x5[3].weight.data
            _, centroids, indice = cluster_weight(branch5_weight2)
            # plot_tsne_embedding(branch5_weight2.cpu().view(branch5_weight2.size(0), -1), label, indice, '', plot_name=name+'branch5.weight2')
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
    get_flops_params(origin_model, model)
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

def cluster_resnet_imagenet():
    if args.pretrain_model is None or not os.path.exists(args.pretrain_model):
        raise ('pretrain model path should be exist!')
    ckpt = torch.load(args.pretrain_model, map_location=device)
    origin_model = import_module(f'model.{args.arch}_imagenet').resnet(args.cfg).to(device)
    origin_model.load_state_dict(ckpt)

    cfg = []
    centroids_state_dict = {}
    prune_state_dict = []
    indices = []

    for name, module in origin_model.named_modules():

        if isinstance(module, BasicBlock):

            conv1_weight = module.conv1.weight.data
            label, centroids, indice = cluster_weight(conv1_weight)
            # plot_tsne_embedding(conv1_weight.cpu().view(conv1_weight.size(0), -1), label, indice, '', plot_name=name+'.conv1.weight')
            cfg.append(len(centroids))
            cfg.append(0)  # assume baseblock has three conv layer
            centroids_state_dict[name + '.conv1.weight'] = centroids
            if args.init_method == 'random_project':
                centroids_state_dict[name + '.conv2.weight'] = random_project(module.conv2.weight.data, len(centroids))
            else:
                centroids_state_dict[name + '.conv2.weight'] = direct_project(module.conv2.weight.data, indice)

            prune_state_dict.append(name + '.bn1.weight')
            prune_state_dict.append(name + '.bn1.bias')
            prune_state_dict.append(name + '.bn1.running_var')
            prune_state_dict.append(name + '.bn1.running_mean')

        elif isinstance(module, Bottleneck):

            conv1_weight = module.conv1.weight.data
            label, centroids, indice = cluster_weight(conv1_weight)
            cfg.append(len(centroids))
            # plot_tsne_embedding(conv1_weight.cpu().view(conv1_weight.size(0), -1), label, indice, '', plot_name=name+'.conv1.weight')
            indices.append(indice)
            centroids_state_dict[name + '.conv1.weight'] = centroids

            prune_state_dict.append(name + '.bn1.weight')
            prune_state_dict.append(name + '.bn1.bias')
            prune_state_dict.append(name + '.bn1.running_var')
            prune_state_dict.append(name + '.bn1.running_mean')

            conv2_weight = module.conv2.weight.data
            label, centroids, indice = cluster_weight(conv2_weight)
            cfg.append(len(centroids))
            # plot_tsne_embedding(conv2_weight.cpu().view(conv2_weight.size(0), -1), label, indice, '', plot_name=name+'.conv2.weight')
            centroids_state_dict[name + '.conv2.weight'] = centroids.reshape(
                (-1, conv2_weight.size(1), conv2_weight.size(2), conv2_weight.size(3)))

            if args.init_method == 'random_project':
                centroids_state_dict[name + '.conv3.weight'] = random_project(module.conv3.weight.data, len(centroids))
            else:
                centroids_state_dict[name + '.conv3.weight'] = direct_project(module.conv3.weight.data, indice)

            prune_state_dict.append(name + '.bn2.weight')
            prune_state_dict.append(name + '.bn2.bias')
            prune_state_dict.append(name + '.bn2.running_var')
            prune_state_dict.append(name + '.bn2.running_mean')

    model = import_module(f'model.{args.arch}_imagenet').resnet(args.cfg, layer_cfg=cfg).to(device)
    get_flops_params(origin_model, model, dataset='imagenet')

    if args.init_method == 'random_project' or args.init_method == 'centroids':
        pretrain_state_dict = origin_model.state_dict()
        state_dict = model.state_dict()
        centroids_state_dict_keys = list(centroids_state_dict.keys())

        index = 0
        for k, v in centroids_state_dict.items():

            if k.endswith('.conv2.weight') and args.cfg != 'resnet18' and args.cfg != 'resnet34':
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

def get_flops_params(orimodel, prunemodel, dataset='cifar10'):
    ori_cfgs = {
        'vgg16': [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512],
        'resnet56': [16] * 9 + [32] * 9 + [64] * 9,
        'resnet110': [16] * 18 + [32] * 18 + [64] * 18,
        'googlenet': [96, 16, 32, 128, 32, 96, 96, 16, 48,
                      112, 24, 64, 128, 24, 64, 144, 32, 64,
                      160, 32, 128, 160, 32, 128, 192, 48, 128],
        'resnet18': [64] * 4 + [128] * 4 + [256] * 4 + [512] * 4,
        'resnet34': [64] * 6 + [128] * 8 + [256] * 12 + [512] * 6,
        'resnet50': [64] * 6 + [128] * 8 + [256] * 12 + [512] * 6,
        'resnet101': [64] * 6 + [128] * 8 + [256] * 46 + [512] * 6,
        'resnet152': [64] * 6 + [128] * 16 + [256] * 72 + [512] * 6,
    }

    if dataset == 'imagenet':
        input = torch.randn(1, 3 ,224, 224).to(device)
    else:
        input = torch.randn(1, 3, 32, 32).to(device)

    print('--------------UnPruned Model--------------')
    oriflops, oriparams = profile(orimodel, inputs=(input,))
    print('Params: %.2f' % (oriparams))
    print('FLOPS: %.2f' % (oriflops))

    print('--------------Pruned Model--------------')
    flops, params = profile(prunemodel, inputs=(input,))
    print('Params: %.2f' % (params))
    print('FLOPS: %.2f' % (flops))

    print('\n')
    layer_cfg = ''
    for i in range(len(prunemodel.layer_cfg)):
        if prunemodel.layer_cfg[i] == 0:
            continue
        layer_cfg = layer_cfg + str(prunemodel.layer_cfg[i]) + ' '
    print('Pruned model\'s cfg: ' + layer_cfg)

    ori_cfg = ''
    for i in range(len(ori_cfgs[args.cfg])):
        ori_cfg = ori_cfg + str(ori_cfgs[args.cfg][i]) + ' '
    print('Unpruned model\'s cfg: ' + ori_cfg)
    print('--------------Retention Ratio--------------')
    channel_retention = ''
    for i in range(len(prunemodel.layer_cfg)):
        if prunemodel.layer_cfg[i] == 0:
            continue
        channel_retention = channel_retention + str(prunemodel.layer_cfg[i] / ori_cfgs[args.cfg][i]) + ' '
    print('Channel Retentio Ratio: ' + channel_retention)
    print('Params Retention Ratio: %d/%d (%.2f%%)' % (params, oriparams, 100. * params / oriparams))
    print('FLOPS Retention Ratio: %d/%d (%.2f%%)' % (flops, oriflops, 100. * flops / oriflops))

    print('--------------Prune Ratio--------------')
    channel_prune = ''
    for i in range(len(prunemodel.layer_cfg)):
        if prunemodel.layer_cfg[i] == 0:
            continue
        channel_prune = channel_prune + str(1 - prunemodel.layer_cfg[i] / ori_cfgs[args.cfg][i]) + ' '
    print('Channel Prune Ratio:' + channel_prune)
    print('Params Prune Ratio: %d/%d (%.2f%%)' % (oriparams - params, oriparams, 100.0 * (1.0 - params / oriparams)))
    print('FLOPS Prune Ratio: %d/%d (%.2f%%)' % (oriflops - flops, oriflops, 100. * (1.0 - flops / oriflops)))

def get_color_set(color_num):

    colors = ['r', 'g', 'b', 'k', 'c', 'm', 'y']

    if len(colors) < color_num:

        times = color_num - len(colors)
        for i in range(times):
            colors.append(plt.cm.Set1(i / 10.))
    return colors

def plot_tsne_embedding(data, label, indices, title, plot_name):

    t_sne = TSNE(n_components=2)
    data = t_sne.fit(X=data, y=label).fit_transform(data)
    # t_sne.fit_transform(data, label)
    # data = t_sne.fit_transform(data)
    # x_min, x_max = np.min(data, 0), np.max(data, 0)
    # data = (data - x_min) / (x_max - x_min)

    plt.figure()

    centroids = []
    for i in range(data.shape[0]):
        if i in indices:
            centroids.append('centroid')
        else:
            centroids.append('other')
    # print(label)
    tips = {
        'x': data[:, 0],
        'y': data[:, 1],
        'label': label,
        'centroids': centroids
    }
    cmap = {}
    palette = sns.color_palette("Set2", 8) + \
              sns.color_palette("Paired", 12) + \
              sns.color_palette("hls", 8) + \
              sns.color_palette("husl", 8) + \
              sns.color_palette() + \
              sns.color_palette("Set1", 8) + \
              sns.color_palette("Set3", 8)
    for i in range(len(palette)):
        cmap[i] = palette[i]
    # sns.set_palette(palette)
    # print(palette)
    # sns.color_palette(palette)
    # sns.palplot(palette)

    frame = pd.DataFrame(tips)
    ax = sns.scatterplot(x='x', y='y', hue='label', size='centroids', sizes=(100,200),
                         size_order=['centroid', 'other'],
                          palette=cmap, data=frame, legend=False)
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles=handles[-2:], labels=labels[-2:])

    # ax = sns.scatterplot(x='x', y='y', hue='label', size='centroids',
    #                      palette='Set2', data=frame)

    plt.axes = ax
    plt.title(title)

    output_path = './tsne_plot/' + args.cfg
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    plt.savefig(os.path.join(output_path, plot_name + '.png'))
    plt.close()

def plot_pca_embedding(data, label, indices, title, plot_name):

    # x_min, x_max = np.min(data, 0), np.max(data, 0)
    # data = (data - x_min) / (x_max - x_min)
    pca = PCA(n_components=2)
    data = pca.fit(X=data, y=label).transform(data)

    plt.figure()

    centroids = []
    for i in range(data.shape[0]):
        if i in indices:
            centroids.append('centroid')
        else:
            centroids.append('other')
    # print(label)
    tips = {
        'x': data[:, 0],
        'y': data[:, 1],
        'label': label,
        'centroids': centroids
    }
    cmap = {}
    palette = sns.color_palette("Set2", 8) + \
              sns.color_palette("Paired", 12) + \
              sns.color_palette("hls", 8) + \
              sns.color_palette("husl", 8) + \
              sns.color_palette() + \
              sns.color_palette("Set1", 8) + \
              sns.color_palette("Set3", 8)
    for i in range(len(palette)):
        cmap[i] = palette[i]
    # sns.set_palette(palette)
    # print(palette)
    # sns.color_palette(palette)
    # sns.palplot(palette)

    frame = pd.DataFrame(tips)
    ax = sns.scatterplot(x='x', y='y', hue='label', size='centroids', sizes=(100,200),
                         size_order=['centroid', 'other'],
                          palette=cmap, data=frame, legend=False)
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles=handles[-2:], labels=labels[-2:])

    # ax = sns.scatterplot(x='x', y='y', hue='label', size='centroids',
    #                      palette='Set2', data=frame)

    plt.axes = ax
    plt.title(title)

    output_path = './pca_plot/' + args.cfg
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    plt.savefig(os.path.join(output_path, plot_name + '.png'))
    plt.close()


def main():

    # Model
    print('==> Building model..')
    if args.arch == 'resnet' and args.dataset == 'cifar10':
        cluster_resnet()
    elif args.arch == 'vgg':
        cluster_vgg()
    elif args.arch == 'googlenet':
        cluster_googlenet()
    elif args.arch == 'resnet' and args.dataset == 'imagenet':
        cluster_resnet_imagenet()
    else:
        raise('arch not exist!')
    print('==>Search Done!')

if __name__ == '__main__':
    main()