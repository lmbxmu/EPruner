from __future__ import absolute_import
import datetime
import shutil
from pathlib import Path
import logging
import torch
from .options import args
import numpy as np

import os
from sklearn.cluster import AffinityPropagation
from sklearn.cluster.affinity_propagation_ import euclidean_distances
from sklearn.random_projection import SparseRandomProjection

device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class checkpoint():
    def __init__(self, args):
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        self.args = args
        self.job_dir = Path(args.job_dir)
        self.ckpt_dir = self.job_dir / 'checkpoint'
        self.run_dir = self.job_dir / 'run'

        def _make_dir(path):
            if not os.path.exists(path):
                os.makedirs(path)

        _make_dir(self.job_dir)
        _make_dir(self.ckpt_dir)
        _make_dir(self.run_dir)

        config_dir = self.job_dir / 'config.txt'
        with open(config_dir, 'w') as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

    def save_model(self, state, epoch, is_best):
        save_path = f'{self.ckpt_dir}/model_{epoch}.pt'
        torch.save(state, save_path)
        if is_best:
            shutil.copyfile(save_path, f'{self.ckpt_dir}/model_best.pt')


def get_logger(file_path):
    logger = logging.getLogger('gal')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def get_prune_rate(sketch_rate):
    import re

    cprate_str = sketch_rate
    cprate_str_list = cprate_str.split('+')
    pat_cprate = re.compile(r'\d+\.\d*')
    pat_num = re.compile(r'\*\d+')
    cprate = []
    for x in cprate_str_list:
        num = 1
        find_num = re.findall(pat_num, x)
        if find_num:
            assert len(find_num) == 1
            num = int(find_num[0].replace('*', ''))
        find_cprate = re.findall(pat_cprate, x)
        assert len(find_cprate) == 1
        cprate += [float(find_cprate[0])] * num

    return cprate


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

def weight_norm(weight):

    if args.weight_norm_method == 'l2':
        norm_func = lambda x: np.sqrt(np.sum(np.power(x, 2)))
    else:
        norm_func = lambda x: 1.0

    weight /= norm_func(weight)

    return weight

def random_project(weight, channel_num):

    A = weight.cpu().clone()
    A = A.view(A.size(0), -1)
    rp = SparseRandomProjection(n_components=channel_num * weight.size(2) * weight.size(3))
    rp.fit(A)
    return rp.transform(A)
    # return weight_norm(rp.transform(A))

def direct_project(weight, indices):

    A = torch.randn(weight.size(0), len(indices), weight.size(2), weight.size(3))
    for i, indice in enumerate(indices):

        A[:, i, :, :] = weight[:, indice, :, :]

    return A