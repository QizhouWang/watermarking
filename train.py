import numpy as np
import os, sys
import pickle
import argparse
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F

from wrn import WideResNet
from utils import get_measures, print_measures

parser = argparse.ArgumentParser(description='Watermarking for OOD Detection', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Experimental Setup
parser.add_argument('dataset', type=str, choices=['cifar10', 'cifar100'])
parser.add_argument('--score', type=str, default='MSP',  choices=['MSP', 'energy'])

# Hyerparameter
parser.add_argument('--epochs',     type=int, default=50,  help='epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size.')
parser.add_argument('--test_bs',    type=int, default=200, help='batch size.')

parser.add_argument('--alpha', type=float, default=0.0, help='learning rate.')
parser.add_argument('--sigma', type=float, default=0.0, help='covaraince.')
parser.add_argument('--rho',   type=float, default=0.0, help='constraint.')
parser.add_argument('--beta',  type=float, default=0.0, help='trade-off.')
parser.add_argument('--T_in',  type=float, default=0.0, help='Scale for energy ID.')
parser.add_argument('--T_out',  type=float, default=0.0, help='Scale for energy OOD.')

parser.add_argument('--T', default=1., type=float, help='temperature: energy')

# WRN Architecture
parser.add_argument('--layers',       default=40,  type=int)
parser.add_argument('--widen-factor', default=2,   type=int)
parser.add_argument('--droprate',     default=0.3, type=float)

# Others
parser.add_argument('--load',       type=str, default='./ckpt')
parser.add_argument('--prefetch',   type=int, default=4)
parser.add_argument('--out_as_pos', action = 'store_true')
parser.add_argument('--num_to_avg', type=int, default=1)

args = parser.parse_args()

# pre defined hyperparameter 
if args.alpha == 0 and args.sigma == 0 and args.rho == 0 and args.beta == 0:
    print('loading predefined hyper parameters')
    args.alpha = 0.01
    if args.dataset == 'cifar10'  and args.score == 'MSP':
        args.sigma = 0.4
        args.rho   = 1.0
        args.beta  = 3.5
    if args.dataset == 'cifar100' and args.score == 'MSP':
        args.sigma = 1.0
        args.rho   = 0.2
        args.beta  = 2.5
    if args.dataset == 'cifar10'  and args.score == 'energy':
        args.sigma = 1.5
        args.rho   = 0.05
        args.beta  = 0.1
        args.T_in  = 0.2
        args.T_out = 0.7
    if args.dataset == 'cifar100'  and args.score == 'energy':
        args.sigma = 0.4
        args.rho   = 0.1
        args.beta  = 0.1
        args.T_in  = 0.9
        args.T_out = 0.1


cudnn.benchmark = True

mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std  = [x / 255 for x in [63.0, 62.1, 66.7]]

train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.ToTensor(), trn.Normalize(mean, std)])
test_transform  = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

if args.dataset == 'cifar10':
    train_data  = dset.CIFAR10('../data/cifarpy', train=True,  transform=train_transform, download=True)
    test_data   = dset.CIFAR10('../data/cifarpy', train=False, transform=test_transform)
    num_classes = 10
else:
    train_data  = dset.CIFAR100('../data/cifarpy', train=True,  transform=train_transform, download=True)
    test_data   = dset.CIFAR100('../data/cifarpy', train=False, transform=test_transform)
    num_classes = 100

train_loader    = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,  num_workers=args.prefetch, pin_memory=False)
test_loader     = torch.utils.data.DataLoader(test_data,  batch_size=args.batch_size, shuffle=False, num_workers=args.prefetch, pin_memory=False)

texture_data    = dset.ImageFolder(root="../watermarking_final/data/dtd/images",    transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32), trn.ToTensor(), trn.Normalize(mean, std)]))
svhn_data       = dset.SVHN(root='../data/svhn/', split="test",  transform=trn.Compose( [trn.ToTensor(), trn.Normalize(mean, std)]), download = True)
places365_data  = dset.ImageFolder(root="../data/places365_standard/", transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32), trn.ToTensor(), trn.Normalize(mean, std)]))
# lsunc_data      = dset.ImageFolder(root="../data/LSUN",          transform=trn.Compose([trn.Resize(32), trn.ToTensor(), trn.Normalize(mean, std)]))
lsunr_data      = dset.ImageFolder(root="../data/LSUN_resize",   transform=trn.Compose([trn.Resize(32), trn.ToTensor(), trn.Normalize(mean, std)]))
isun_data       = dset.ImageFolder(root="../data/iSUN",          transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))

texture_loader  = torch.utils.data.DataLoader(texture_data,    batch_size=args.test_bs, shuffle=True, num_workers=1, pin_memory=False)
svhn_loader     = torch.utils.data.DataLoader(svhn_data,       batch_size=args.test_bs, shuffle=True, num_workers=1, pin_memory=False)
places365_loader = torch.utils.data.DataLoader(places365_data, batch_size=args.test_bs, shuffle=True, num_workers=1, pin_memory=False)
# lsunc_loader    = torch.utils.data.DataLoader(lsunc_data,      batch_size=args.test_bs, shuffle=True, num_workers=1, pin_memory=False)
lsunr_loader    = torch.utils.data.DataLoader(lsunr_data,      batch_size=args.test_bs, shuffle=True, num_workers=1, pin_memory=False)
isun_loader     = torch.utils.data.DataLoader(isun_data,       batch_size=args.test_bs, shuffle=True, num_workers=1, pin_memory=False)


ood_num_examples = len(test_data) // 5
concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()

def get_ood_scores(loader, watermark = None, in_dist=False):
    _score = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if batch_idx >= ood_num_examples // args.test_bs and in_dist is False:
                break

            data, target = data.cuda(), target.cuda()
       
            if watermark is not None: 
                data = data + watermark.cuda()
            output = net(data)
            smax = to_np(F.softmax(output, dim=1))
            if args.score == 'energy':
                _score.append(-to_np((args.T*torch.logsumexp(output / args.T, dim=1))))
            else: 
                _score.append(-np.max(smax, axis=1))

    if in_dist:
        return concat(_score).copy() 
    else:
        return concat(_score)[:ood_num_examples].copy()

def get_and_print_results(ood_loader, in_score, watermark = None, num_to_avg=args.num_to_avg):
    aurocs, auprs, fprs = [], [], []
    for _ in range(num_to_avg):
        out_score = get_ood_scores(ood_loader, watermark)
        if args.out_as_pos:
            measures = get_measures(out_score, in_score)
        else:
            measures = get_measures(-in_score, -out_score)
        aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])
    auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)
    print_measures(auroc, aupr, fpr, '')

    return auroc, aupr, fpr

def cal_loss(x, in_set):
    loss = F.cross_entropy(x[:len(in_set[0])], in_set[1])

    if args.score == 'energy':
        T_out, T_in = args.T_out, args.T_in
        Ec_out = (x[len(in_set[0]):] * T_out).exp().mean(1)
        Ec_in  = (-x[:len(in_set[0])] * T_in).exp().mean(1)
        loss += args.beta * (Ec_out.mean() + Ec_in.mean())
    else:
        loss_oe = -(x[len(in_set[0]):].mean(1) - torch.logsumexp(x[len(in_set[0]):], dim=1)) 
        loss += args.beta * loss_oe.mean()

    return loss

def train(epoch, watermark, alpha):
    loss_avg = 0.0
    for batch_idx, in_set in enumerate(train_loader):
        in_set[0], in_set[1] = in_set[0].cuda(), in_set[1].cuda()
        out_set = torch.ones_like(in_set[0]) * 0 + torch.randn_like(in_set[0]) * args.sigma
        data = torch.cat((in_set[0], out_set), 0)

        # [STAGE 1] for perturbation
        watermark.requires_grad_()
        optim.zero_grad()
        logits = net(data + watermark)
        loss = cal_loss(logits, in_set)
        grad = torch.autograd.grad(loss, [watermark])[0].detach()
        perturb = grad.sign() * grad.abs() / (grad ** 2).sum().sqrt()

        # [STAGE 2] for updating
        watermark.requires_grad_()
        optim.zero_grad()
        logits = net(data + watermark + perturb * args.rho)
        loss = cal_loss(logits, in_set)
        grad = torch.autograd.grad(loss, [watermark])[0].detach()
        watermark = watermark.detach() - alpha * torch.sign(grad)

        # LOGS
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2
        sys.stdout.write('\r E%d [%d / %d] loss %.2f ' %(epoch, batch_idx + 1, len(train_loader), loss_avg))
    return watermark

net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate).cuda()
optim = torch.optim.SGD(net.parameters(), lr = 1)
watermark = torch.zeros(1,3,32,32).cuda()
# Restore model
if args.dataset == 'cifar10':
    model_path = args.load  + '/cifar10_wrn_pretrained_epoch_99.pt'
else:
    model_path = args.load  + '/cifar100_wrn_pretrained_epoch_99.pt'
net.load_state_dict(torch.load(model_path))

net.eval()
print('  Score: ' + args.score)
print('  FPR95 AUROC AUPR')
in_score = get_ood_scores(test_loader, in_dist=True)
metric_ll = []
metric_ll.append(get_and_print_results(isun_loader, in_score))
metric_ll.append(get_and_print_results(places365_loader, in_score))
metric_ll.append(get_and_print_results(texture_loader, in_score))
metric_ll.append(get_and_print_results(svhn_loader, in_score))
metric_ll.append(get_and_print_results(lsunr_loader, in_score))
avg_r = torch.Tensor(metric_ll).mean(0)
print_measures(avg_r[0], avg_r[1], avg_r[2], '')

alpha = args.alpha
for epoch in range(0, args.epochs + 1):
    if epoch == 25: alpha /= 10
    watermark = train(epoch, watermark, alpha)
    if epoch % 5 == 0:
        print('\n  FPR95 AUROC AUPR')
        in_score = get_ood_scores(test_loader, watermark, in_dist=True)
        metric_ll = []
        metric_ll.append(get_and_print_results(isun_loader, in_score, watermark))
        metric_ll.append(get_and_print_results(places365_loader, in_score, watermark))
        metric_ll.append(get_and_print_results(texture_loader, in_score, watermark))
        metric_ll.append(get_and_print_results(svhn_loader, in_score, watermark))
        metric_ll.append(get_and_print_results(lsunr_loader, in_score, watermark))
        avg_r = torch.Tensor(metric_ll).mean(0)
        print_measures(avg_r[0], avg_r[1], avg_r[2], '')

