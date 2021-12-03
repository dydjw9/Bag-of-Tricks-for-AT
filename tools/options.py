#python scripts
__author__='Du Jiawei NUS/IHPC'
__email__='dujiawei@u.nus.edu'
#Descrption:
import argparse
import logging
import sys
import time
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import os

mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
std = torch.tensor(cifar10_std).view(3,1,1).cuda()

def normalize(x):
    return (x - mu)/std

upper_limit, lower_limit = 1,0


def clamp(x, lower_limit, upper_limit):
    return torch.max(torch.min(x, upper_limit), lower_limit)




def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts,
               norm, mixup=False, y_a=None, y_b=None, lam=None, 
               early_stop=False, early_stop_pgd_max=1,
               multitarget=False,
               use_DLRloss=False, use_CWloss=False,
               epoch=0, totalepoch=110, gamma=0.8,
               use_adaptive=False, s_HE=15,
               fast_better=False, BNeval=False):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()

    if BNeval:
        model.eval()

    for _ in range(restarts):
        # early stop pgd counter for each x
        early_stop_pgd_count = early_stop_pgd_max * torch.ones(y.shape[0], dtype=torch.int32).cuda()

        # initialize perturbation
        delta = torch.zeros_like(X).cuda()
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta.normal_()
            d_flat = delta.view(delta.size(0),-1)
            n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r/n*epsilon
        else:
            raise ValueError
        delta = clamp(delta, lower_limit-X, upper_limit-X)
        delta.requires_grad = True

        iter_count = torch.zeros(y.shape[0])

        # craft adversarial examples
        for _ in range(attack_iters):
            output = model(normalize(X + delta))

            # if use early stop pgd
            if early_stop:
                # calculate mask for early stop pgd
                if_success_fool = (output.max(1)[1] != y).to(dtype=torch.int32)
                early_stop_pgd_count = early_stop_pgd_count - if_success_fool
                index = torch.where(early_stop_pgd_count > 0)[0]
                iter_count[index] = iter_count[index] + 1
            else:
                index = slice(None,None,None)
            if not isinstance(index, slice) and len(index) == 0:
                break

            # Whether use mixup criterion
            if fast_better:
                loss_ori = F.cross_entropy(output, y)
                grad_ori = torch.autograd.grad(loss_ori, delta, create_graph=True)[0]
                loss_grad = (alpha / 4.) * (torch.norm(grad_ori.view(grad_ori.shape[0], -1), p=2, dim=1) ** 2)
                loss = loss_ori + loss_grad.mean()
                loss.backward()
                grad = delta.grad.detach()

            elif not mixup:
                if multitarget:
                    random_label = torch.randint(low=0, high=10, size=y.shape).cuda()
                    random_direction = 2*((random_label == y).to(dtype=torch.float32) - 0.5)
                    loss = torch.mean(random_direction * F.cross_entropy(output, random_label, reduction='none'))
                    loss.backward()
                    grad = delta.grad.detach()
                elif use_DLRloss:
                    beta_ = gamma * epoch / totalepoch
                    loss = (1. - beta_) * F.cross_entropy(output, y) + beta_ * dlr_loss(output, y)
                    loss.backward()
                    grad = delta.grad.detach()
                elif use_CWloss:
                    beta_ = gamma * epoch / totalepoch
                    loss = (1. - beta_) * F.cross_entropy(output, y) + beta_ * CW_loss(output, y)
                    loss.backward()
                    grad = delta.grad.detach()
                else:
                    if use_adaptive:
                        loss = F.cross_entropy(s_HE * output, y)
                    else:
                        loss = F.cross_entropy(output, y)
                    loss.backward()
                    grad = delta.grad.detach()
            else:
                criterion = nn.CrossEntropyLoss()
                loss = mixup_criterion(criterion, model(normalize(X+delta)), y_a, y_b, lam)
                loss.backward()
                grad = delta.grad.detach()


            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                scaled_g = g/(g_norm + 1e-10)
                d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()
        if mixup:
            criterion = nn.CrossEntropyLoss(reduction='none')
            all_loss = mixup_criterion(criterion, model(normalize(X+delta)), y_a, y_b, lam)
        else:
            all_loss = F.cross_entropy(model(normalize(X+delta)), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)

    if BNeval:
        model.train()

    return max_delta, iter_count



def get_args():
    parser = argparse.argumentparser()
    parser.add_argument('--model', default='preactresnet18')
    parser.add_argument('--l1', default=0, type=float)
    parser.add_argument('--data-dir', default='../cifar-data', type=str)
    parser.add_argument('--epochs', default=110, type=int)
    parser.add_argument('--lr-schedule', default='piecewise', choices=['superconverge', 'piecewise', 'linear', 'piecewisesmoothed', 'piecewisezoom', 'onedrop', 'multipledecay', 'cosine', 'cyclic'])
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--lr-one-drop', default=0.01, type=float)
    parser.add_argument('--lr-drop-epoch', default=100, type=int)
    parser.add_argument('--attack', default='pgd', type=str, choices=['pgd', 'fgsm', 'free', 'none'])
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--test_epsilon', default=8, type=int)
    parser.add_argument('--attack-iters', default=10, type=int)
    parser.add_argument('--restarts', default=1, type=int)
    parser.add_argument('--pgd-alpha', default=2, type=float)
    parser.add_argument('--test-pgd-alpha', default=2, type=float)
    parser.add_argument('--fgsm-alpha', default=1.25, type=float)
    parser.add_argument('--norm', default='l_inf', type=str, choices=['l_inf', 'l_2'])
    parser.add_argument('--fgsm-init', default='random', choices=['zero', 'random', 'previous'])
    parser.add_argument('--fname', default='cifar_model', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--width-factor', default=10, type=int)
    parser.add_argument('--resume', default=0, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--chkpt-iters', default=100, type=int)
    parser.add_argument('--mixture', action='store_true') # whether use mixture of clean and adv examples in a mini-batch
    parser.add_argument('--mixture_alpha', type=float)
    parser.add_argument('--l2', default=0, type=float)

    # group 1
    parser.add_argument('--earlystoppgd', action='store_true') # whether use early stop in pgd
    parser.add_argument('--earlystoppgdepoch1', default=60, type=int)
    parser.add_argument('--earlystoppgdepoch2', default=100, type=int)

    parser.add_argument('--warmup_lr', action='store_true') # whether warm_up lr from 0 to max_lr in the first n epochs
    parser.add_argument('--warmup_lr_epoch', default=15, type=int)

    parser.add_argument('--weight_decay', default=5e-4, type=float)#weight decay

    parser.add_argument('--warmup_eps', action='store_true') # whether warm_up eps from 0 to 8/255 in the first n epochs
    parser.add_argument('--warmup_eps_epoch', default=15, type=int)

    parser.add_argument('--batch-size', default=128, type=int) #batch size

    parser.add_argument('--labelsmooth', action='store_true') # whether use label smoothing
    parser.add_argument('--labelsmoothvalue', default=0.0, type=float)

    parser.add_argument('--lrdecay', default='base', type=str, choices=['intenselr', 'base', 'looselr', 'lineardecay'])

    # group 2
    parser.add_argument('--use_dlrloss', action='store_true') # whether use dlrloss
    parser.add_argument('--use_cwloss', action='store_true') # whether use cwloss


    parser.add_argument('--use_multitarget', action='store_true') # whether use multitarget

    parser.add_argument('--use_stronger_adv', action='store_true') # whether use mixture of clean and adv examples in a mini-batch
    parser.add_argument('--stronger_index', default=0, type=int)

    parser.add_argument('--use_fnandwn', action='store_true') # whether use fn and wn
    parser.add_argument('--use_adaptive', action='store_true') # whether use s in attack during training
    parser.add_argument('--s_fn', default=15, type=float) # s in fn
    parser.add_argument('--m_fn', default=0.2, type=float) # s in fn

    parser.add_argument('--use_fnonly', action='store_true') # whether use fn only

    parser.add_argument('--fast_better', action='store_true')

    parser.add_argument('--bneval', action='store_true') # whether use eval mode for bn when crafting adversarial examples

    parser.add_argument('--focalloss', action='store_true') # whether use focalloss
    parser.add_argument('--focallosslambda', default=2., type=float)

    parser.add_argument('--activation', default='relu', type=str)
    parser.add_argument('--softplus_beta', default=1., type=float)

    parser.add_argument('--optimizer', default='momentum', choices=['momentum', 'nesterov', 'sgd_gc', 'sgd_gcc', 'adam', 'adamw'])

    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--mixup-alpha', type=float)

    parser.add_argument('--cutout', action='store_true')
    parser.add_argument('--cutout-len', type=int)

    return parser.parse_args()

