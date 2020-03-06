import torch
import torch.nn as nn

from thop import profile
import utils.common as utils
from data import cifar10, imagenet_dali, imagenet
from importlib import import_module
import argparse
import os
import time

parser = argparse.ArgumentParser('Test Flops & Params')
parser.add_argument(
    '--pruned_model',
    type=str,
    default=None,
    help='The path of pruned model. default:None'
)
parser.add_argument(
    '--arch',
    type=str,
    default='resnet',
    help='Architecture of model. default:resnet')

parser.add_argument(
    '--cfg',
    type=str,
    default='resnet56',
    help='Detail architecuture of model. default:resnet56'
)
parser.add_argument(
    '--dataset',
    type=str,
    default='cifar10',
    help='Select dataset to train. default:cifar10',
)
parser.add_argument(
    '--data_path',
    type=str,
    default='/home/lishaojie/data/cifar10/',
    help='The dictionary where the input is stored. default:/home/lishaojie/data/cifar10/',
)
parser.add_argument(
    '--eval_batch_size',
    type=int,
    default=100,
    help='Batch size for validation. default:100'
)
parser.add_argument(
    '--gpus',
    type=int,
    nargs='+',
    default=[0],
    help='Select gpu_id to use. default:[0]',
)
args = parser.parse_args()

device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'
loss_func = nn.CrossEntropyLoss()

# Data
print('==> Preparing data..')
if args.data_set == 'cifar10':
    testLoader = cifar10.Data(args).testLoader
else: #imagenet
    if device != 'cpu':
        testLoader = imagenet_dali.get_imagenet_iter_dali('val', args.data_path, args.eval_batch_size,
                                             num_threads=4, crop=224, device_id=args.gpus[0], num_gpus=1)
    else:
        testLoader = imagenet.Data(args).testLoader

def get_flops_params(orimodel, prunemodel):

    if args.dataset == 'imagenet':
        input = torch.randn(1, 3 ,224, 224).to(device)
    else:
        input = torch.randn(1, 3, 32, 32).to(device)

    orimodel_channel_num = 0

    for name, module in orimodel.named_modules():

        if isinstance(module, nn.Conv2d):

            orimodel_channel_num += module.weight.data.size(0)

    prunemodel_channel_num = 0

    for name, module in prunemodel.named_modules():

        if isinstance(module, nn.Conv2d):

            prunemodel_channel_num += module.weight.data.size(0)

    print('--------------UnPruned Model--------------')
    oriflops, oriparams = profile(orimodel, inputs=(input,), verbose=False)
    print('Channel num: %d' % orimodel_channel_num)
    print('Params: %.2f' % (oriparams))
    print('FLOPS: %.2f' % (oriflops))

    print('--------------Pruned Model--------------')
    flops, params = profile(prunemodel, inputs=(input,), verbose=False)
    print('Channel num: %d' % prunemodel_channel_num)
    print('Params: %.2f' % (params))
    print('FLOPS: %.2f' % (flops))

    print('--------------Prune Ratio--------------')
    print('Channel num Prune Ratio: %d/%d (%.2f%%)' % (orimodel_channel_num - prunemodel_channel_num, orimodel_channel_num,
                                                      100.0 * (1.0 - prunemodel_channel_num/orimodel_channel_num)))
    print('Params Prune Ratio: %d/%d (%.2f%%)' % (oriparams - params, oriparams, 100.0 * (1.0 - params / oriparams)))
    print('FLOPS Prune Ratio: %d/%d (%.2f%%)' % (oriflops - flops, oriflops, 100. * (1.0 - flops / oriflops)))

def test(model, topk=(1,)):
    model.eval()

    losses = utils.AverageMeter()
    accuracy = utils.AverageMeter()
    top5_accuracy = utils.AverageMeter()

    start_time = time.time()
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(testLoader):
            if len(topk) == 2:
                inputs = batch_data[0]['data'].to(device)
                targets = batch_data[0]['label'].squeeze().long().to(device)
            else:
                inputs = batch_data[0]
                targets = batch_data[1]
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
            print(
                'Test Loss {:.4f}\tAccuracy {:.2f}%\t\tTime {:.2f}s\n'
                .format(float(losses.avg), float(accuracy.avg), (current_time - start_time))
            )
        else:
            print(
                'Test Loss {:.4f}\tTop1 {:.2f}%\tTop5 {:.2f}%\tTime {:.2f}s\n'
                    .format(float(losses.avg), float(accuracy.avg), float(top5_accuracy.avg), (current_time - start_time))
            )

def main():

    if args.pruned_model is None or not os.path.exists(args.pruned_model):
        raise ('Pruned Model path must be exist!')
    # Model
    print('==> Building model..')
    ckpt = torch.load(args.pruned_model, map_location='cpu')
    layer_cfg = ckpt['cfg']

    if args.arch == 'resnet' and args.dataset == 'cifar10':
        origin_model = import_module(f'model.{args.arch}_cifar').resnet(args.cfg).to(device)
        pruned_model = import_module(f'model.{args.arch}_cifar').resnet(args.cfg, layer_cfg=layer_cfg).to(device)

    elif args.arch == 'vgg':
        origin_model = import_module(f'model.{args.arch}_cifar').VGG(args.cfg).to(device)
        pruned_model = import_module(f'model.{args.arch}_cifar').VGG(args.cfg, layer_cfg=layer_cfg).to(device)

    elif args.arch == 'googlenet':
        origin_model = import_module(f'model.{args.arch}').googlenet().to(device)
        pruned_model = import_module(f'model.{args.arch}').googlenet(layer_cfg=layer_cfg).to(device)

    elif args.arch == 'resnet' and args.dataset == 'imagenet':
        origin_model = import_module(f'model.{args.arch}_imagenet').resnet(args.cfg).to(device)
        pruned_model = import_module(f'model.{args.arch}_imagenet').resnet(args.cfg, layer_cfg=layer_cfg).to(device)

    else:
        raise('arch not exist!')
    get_flops_params(origin_model, pruned_model)
    test(pruned_model, topk=(1, 5) if args.data_set == 'imagenet' else (1,))

if __name__ == '__main__':
    main()