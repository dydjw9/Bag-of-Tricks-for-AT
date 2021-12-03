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

from wideresnet import WideResNet
from preactresnet import PreActResNet18, PreActResNet50
from models import *
from tools.options import attack_pgd,get_args
from tools.awp import AdvWeightPerturb 

from utils import *
import torch.distributed as dist

mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
std = torch.tensor(cifar10_std).view(3,1,1).cuda()

def normalize(x):
    return (x - mu)/std

upper_limit, lower_limit = 1,0


def clamp(x, lower_limit, upper_limit):
    return torch.max(torch.min(x, upper_limit), lower_limit)


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=10, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class Batches():
    def __init__(self, dataset, batch_size, shuffle, set_random_choices=False, num_workers=0, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.set_random_choices = set_random_choices
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=shuffle, drop_last=drop_last
        )

    def __iter__(self):
        if self.set_random_choices:
            self.dataset.set_random_choices()
        return ({'input': x.to(device).float(), 'target': y.to(device).long()} for (x,y) in self.dataloader)

    def __len__(self):
        return len(self.dataloader)


def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def dlr_loss(x, y):
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()
    
    loss_value = -(x[np.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] 
        * (1. - ind)) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)
    return loss_value.mean()

def CW_loss(x, y):
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()
    
    loss_value = -(x[np.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind))
    return loss_value.mean()    


def get_auto_fname(args):
    names = args.model + '_' + args.lr_schedule + '_eps' + str(args.epsilon) + '_bs' + str(args.batch_size) + '_maxlr' + str(args.lr_max)
    # Group 1
    if args.earlystopPGD:
        names = names + '_earlystopPGD' + str(args.earlystopPGDepoch1) + str(args.earlystopPGDepoch2)
    if args.warmup_lr:
        names = names + '_warmuplr' + str(args.warmup_lr_epoch)
    if args.warmup_eps:
        names = names + '_warmupeps' + str(args.warmup_eps_epoch)
    if args.weight_decay != 5e-4:
        names = names + '_wd' + str(args.weight_decay)
    if args.labelsmooth:
        names = names + '_ls' + str(args.labelsmoothvalue)

    # Group 2
    if args.use_stronger_adv:
        names = names + '_usestrongeradv#' + str(args.stronger_index)
    if args.use_multitarget:
        names = names + '_usemultitarget'
    if args.use_DLRloss:
        names = names + '_useDLRloss'
    if args.use_CWloss:
        names = names + '_useCWloss'
    if args.use_FNandWN:
        names = names + '_HE' + 's' + str(args.s_FN) + 'm' + str(args.m_FN)
    if args.use_adaptive:
        names = names + 'adaptive'
    if args.use_FNonly:
        names = names + '_FNonly'
    if args.fast_better:
        names = names + '_fastbetter'
    if args.activation != 'ReLU':
        names = names + '_' + args.activation
        if args.activation == 'Softplus':
            names = names + str(args.softplus_beta)
    if args.lrdecay != 'base':
        names = names + '_' + args.lrdecay
    if args.BNeval:
        names = names + '_BNeval'
    if args.focalloss:
        names = names + '_focalloss' + str(args.focallosslambda)
    if args.optimizer != 'momentum':
        names = names + '_' + args.optimizer
    if args.mixup:
        names = names + '_mixup' + str(args.mixup_alpha)
    if args.cutout:
        names = names + '_cutout' + str(args.cutout_len)
    if args.attack != 'pgd':
        names = names + '_' + args.attack

    print('File name: ', names)
    return names


def main():
    args = get_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device


    if args.fname == 'auto':
        names = get_auto_fname(args)
        args.fname = 'trained_models/' + names
    else:
        args.fname = 'trained_models/' + args.fname

    if not os.path.exists(args.fname):
        os.makedirs(args.fname)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(args.fname, 'eval.log' if args.eval else 'output.log')),
            logging.StreamHandler()
        ])

    logger.info(args)


    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


    # Prepare data
    transforms = [Crop(32, 32), FlipLR()]
    if args.cutout:
        transforms.append(Cutout(args.cutout_len, args.cutout_len))
    if args.val:
        try:
            dataset = torch.load("cifar10_validation_split.pth")
        except:
            print("Couldn't find a dataset with a validation split, did you run "
                  "generate_validation.py?")
            return
        val_set = list(zip(transpose(dataset['val']['data']/255.), dataset['val']['labels']))
        val_batches = Batches(val_set, args.batch_size, shuffle=False, num_workers=4)
    else:
        dataset = cifar10(args.data_dir)
    train_set = list(zip(transpose(pad(dataset['train']['data'], 4)/255.),
        dataset['train']['labels']))
    train_set_x = Transform(train_set, transforms)
    train_batches = Batches(train_set_x, args.batch_size, shuffle=True, set_random_choices=True, num_workers=4)

    test_set = list(zip(transpose(dataset['test']['data']/255.), dataset['test']['labels']))
    test_batches = Batches(test_set, args.batch_size, shuffle=False, num_workers=4)


    # Set perturbations
    epsilon = (args.epsilon / 255.)
    test_epsilon = (args.test_epsilon / 255.)
    pgd_alpha = (args.pgd_alpha / 255.)
    test_pgd_alpha = (args.test_pgd_alpha / 255.)


    # Set models
    if args.model == 'VGG':
        model = VGG('VGG19')
    elif args.model == 'ResNet18':
        model = ResNet18()
    elif args.model == 'GoogLeNet':
        model = GoogLeNet()
    elif args.model == 'DenseNet121':    
        model = DenseNet121()
    elif args.model == 'DenseNet201':    
        model = DenseNet201()
    elif args.model == 'ResNeXt29':
        model = ResNeXt29_2x64d()
    elif args.model == 'ResNeXt29L':
        model = ResNeXt29_32x4d()
    elif args.model == 'MobileNet':
        model = MobileNet()
    elif args.model == 'MobileNetV2':
        model = MobileNetV2()
    elif args.model == 'DPN26':
        model = DPN26()
    elif args.model == 'DPN92':
        model = DPN92()
    elif args.model == 'ShuffleNetG2':
        model = ShuffleNetG2()
    elif args.model == 'SENet18':
        model = SENet18()
    elif args.model == 'ShuffleNetV2':
        model = ShuffleNetV2(1)
    elif args.model == 'EfficientNetB0':
        model = EfficientNetB0()
    elif args.model == 'PNASNetA':
        model = PNASNetA()
    elif args.model == 'RegNetX':
        model = RegNetX_200MF()
    elif args.model == 'RegNetLX':
        model = RegNetX_400MF()
    elif args.model == 'PreActResNet50':
        model = PreActResNet50()
    elif args.model == 'PreActResNet18':
        model = PreActResNet18(normalize_only_FN=args.use_FNonly, normalize=args.use_FNandWN, scale=args.s_FN,
            activation=args.activation, softplus_beta=args.softplus_beta)
    elif args.model == 'WideResNet':
        model = WideResNet(34, 10, widen_factor=10, dropRate=0.0, normalize=args.use_FNandWN,
            activation=args.activation, softplus_beta=args.softplus_beta)
    elif args.model == 'WideResNet_20':
        model = WideResNet(34, 10, widen_factor=20, dropRate=0.0, normalize=args.use_FNandWN,
            activation=args.activation, softplus_beta=args.softplus_beta)
    else:
        raise ValueError("Unknown model")


    model.to(args.device)
    # if args.local_rank != -1:
        # model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size())


    model.train()

    # Set training hyperparameters
    if args.l2:
        decay, no_decay = [], []
        for name,param in model.named_parameters():
            if 'bn' not in name and 'bias' not in name:
                decay.append(param)
            else:
                no_decay.append(param)
        params = [{'params':decay, 'weight_decay':args.l2},
                  {'params':no_decay, 'weight_decay': 0 }]
    else:
        params = model.parameters()
    if args.lr_schedule == 'cyclic':
        opt = torch.optim.Adam(params, lr=args.lr_max, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
    else:
        if args.optimizer == 'momentum':
            opt = torch.optim.SGD(params, lr=args.lr_max, momentum=0.9, weight_decay=args.weight_decay)
        elif args.optimizer == 'Nesterov':
            opt = torch.optim.SGD(params, lr=args.lr_max, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
        elif args.optimizer == 'SGD_GC':
            opt = SGD_GC(params, lr=args.lr_max, momentum=0.9, weight_decay=args.weight_decay)
        elif args.optimizer == 'SGD_GCC':
            opt = SGD_GCC(params, lr=args.lr_max, momentum=0.9, weight_decay=args.weight_decay)
        elif args.optimizer == 'Adam':
            opt = torch.optim.Adam(params, lr=args.lr_max, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
        elif args.optimizer == 'AdamW':
            opt = torch.optim.AdamW(params, lr=args.lr_max, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
    
    # Cross-entropy (mean)
    if args.labelsmooth:
        criterion = LabelSmoothingLoss(smoothing=args.labelsmoothvalue)
    else:
        criterion = nn.CrossEntropyLoss()

    # If we use freeAT or fastAT with previous init
    if args.attack == 'free':
        delta = torch.zeros(args.batch_size, 3, 32, 32).cuda()
        delta.requires_grad = True
    elif args.attack == 'fgsm' and args.fgsm_init == 'previous':
        delta = torch.zeros(args.batch_size, 3, 32, 32).cuda()
        delta.requires_grad = True

    if args.attack == 'free':
        epochs = int(math.ceil(args.epochs / args.attack_iters))
    else:
        epochs = args.epochs


    # Set lr schedule
    if args.lr_schedule == 'superconverge':
        lr_schedule = lambda t: np.interp([t], [0, args.epochs * 2 // 5, args.epochs], [0, args.lr_max, 0])[0]
    elif args.lr_schedule == 'piecewise':
        def lr_schedule(t, warm_up_lr = args.warmup_lr):
            if t < 100:
                if  warm_up_lr and t < args.warmup_lr_epoch:
                    return (t + 1.) / args.warmup_lr_epoch * args.lr_max
                else:
                    return args.lr_max
            if args.lrdecay == 'lineardecay':
                if t < 105:
                    return args.lr_max * 0.02 * (105 - t)
                else:
                    return 0.
            elif args.lrdecay == 'intenselr':
                if t < 102:
                    return args.lr_max / 10.
                else:
                    return args.lr_max / 100.
            elif args.lrdecay == 'looselr':
                if t < 150:
                    return args.lr_max / 10.
                else:
                    return args.lr_max / 100.
            elif args.lrdecay == 'base':
                if t < 105:
                    return args.lr_max / 10.
                else:
                    return args.lr_max / 100.
    elif args.lr_schedule == 'linear':
        lr_schedule = lambda t: np.interp([t], [0, args.epochs // 3, args.epochs * 2 // 3, args.epochs], [args.lr_max, args.lr_max, args.lr_max / 10, args.lr_max / 100])[0]
    elif args.lr_schedule == 'onedrop':
        def lr_schedule(t):
            if t < args.lr_drop_epoch:
                return args.lr_max
            else:
                return args.lr_one_drop
    elif args.lr_schedule == 'multipledecay':
        def lr_schedule(t):
            return args.lr_max - (t//(args.epochs//10))*(args.lr_max/10)
    elif args.lr_schedule == 'cosine': 
        def lr_schedule(t): 
            return args.lr_max * 0.5 * (1 + np.cos(t / args.epochs * np.pi))
    elif args.lr_schedule == 'cyclic':
        def lr_schedule(t, stepsize=18, min_lr=1e-5, max_lr=args.lr_max):

            # Scaler: we can adapt this if we do not want the triangular CLR
            scaler = lambda x: 1.

            # Additional function to see where on the cycle we are
            cycle = math.floor(1 + t / (2 * stepsize))
            x = abs(t / stepsize - 2 * cycle + 1)
            relative = max(0, (1 - x)) * scaler(cycle)

            return min_lr + (max_lr - min_lr) * relative




    #### Set stronger adv attacks when decay the lr ####
    def eps_alpha_schedule(t, warm_up_eps = args.warmup_eps, if_use_stronger_adv=args.use_stronger_adv, stronger_index=args.stronger_index): # Schedule number 0
        if stronger_index == 0:
            epsilon_s = [epsilon * 1.5, epsilon * 2]
            pgd_alpha_s = [pgd_alpha, pgd_alpha]
        elif stronger_index == 1:
            epsilon_s = [epsilon * 1.5, epsilon * 2]
            pgd_alpha_s = [pgd_alpha * 1.25, pgd_alpha * 1.5]
        elif stronger_index == 2:
            epsilon_s = [epsilon * 2, epsilon * 2.5]
            pgd_alpha_s = [pgd_alpha * 1.5, pgd_alpha * 2]
        else:
            print('Undefined stronger index')

        if if_use_stronger_adv:
            if t < 100:
                if t < args.warmup_eps_epoch and warm_up_eps:
                    return (t + 1.) / args.warmup_eps_epoch * epsilon, pgd_alpha, args.restarts
                else:
                    return epsilon, pgd_alpha, args.restarts
            elif t < 105:
                return epsilon_s[0], pgd_alpha_s[0], args.restarts
            else:
                return epsilon_s[1], pgd_alpha_s[1], args.restarts
        else:
            if t < args.warmup_eps_epoch and warm_up_eps:
                return (t + 1.) / args.warmup_eps_epoch * epsilon, pgd_alpha, args.restarts
            else:
                return epsilon, pgd_alpha, args.restarts

    #### Set the counter for the early stop of PGD ####
    def early_stop_counter_schedule(t):
        if t < args.earlystopPGDepoch1:
            return 1
        elif t < args.earlystopPGDepoch2:
            return 2
        else:
            return 3





    best_test_robust_acc = 0
    best_val_robust_acc = 0
    if args.resume:
        start_epoch = args.resume
        model.load_state_dict(torch.load(os.path.join(args.fname, f'model_{start_epoch-1}.pth')))
        opt.load_state_dict(torch.load(os.path.join(args.fname, f'opt_{start_epoch-1}.pth')))
        logger.info(f'Resuming at epoch {start_epoch}')

        best_test_robust_acc = torch.load(os.path.join(args.fname, f'model_best.pth'))['test_robust_acc']
        if args.val:
            best_val_robust_acc = torch.load(os.path.join(args.fname, f'model_val.pth'))['val_robust_acc']
    else:
        start_epoch = 0

    if args.eval:
        if not args.resume:
            logger.info("No model loaded to evaluate, specify with --resume FNAME")
            return
        logger.info("[Evaluation mode]")

    # logger.info('Epoch \t Train Time \t Test Time \t LR \t Train Loss \t Train Grad \t Train Acc \t Train Robust Loss \t Train Robust Acc || \t Test Loss \t Test Acc \t Test Robust Loss \t Test Robust Acc')
    logger.info('Epoch \t Train Acc \t Train Robust Acc \t Test Acc \t Test Robust Acc')
    
    # Records per epoch for savetxt
    train_loss_record = []
    train_acc_record = []
    train_robust_loss_record = []
    train_robust_acc_record = []
    train_grad_record = []

    test_loss_record = []
    test_acc_record = []
    test_robust_loss_record = []
    test_robust_acc_record = []
    test_grad_record = []

    for epoch in range(start_epoch, epochs):
        model.train()
        start_time = time.time()

        train_loss = 0
        train_acc = 0
        train_robust_loss = 0
        train_robust_acc = 0
        train_n = 0
        train_grad = 0

        record_iter = torch.tensor([])

        for i, batch in enumerate(train_batches):
            if args.eval:
                break
            X, y = batch['input'], batch['target']

            onehot_target_withmargin_HE = args.m_FN * args.s_FN * torch.nn.functional.one_hot(y, num_classes=10)

            if args.mixup:
                X, y_a, y_b, lam = mixup_data(X, y, args.mixup_alpha)
                X, y_a, y_b = map(Variable, (X, y_a, y_b))
            epoch_now = epoch + (i + 1) / len(train_batches)
            lr = lr_schedule(epoch_now)
            opt.param_groups[0].update(lr=lr)

            if args.attack == 'pgd':
                # Random initialization
                epsilon_sche, pgd_alpha_sche, restarts_sche = eps_alpha_schedule(epoch_now)
                early_counter_max = early_stop_counter_schedule(epoch_now)
                if args.mixup:
                    delta, iter_counts = attack_pgd(model, X, y, epsilon_sche, pgd_alpha_sche, args.attack_iters, restarts_sche, args.norm, 
                        early_stop=args.earlystopPGD, early_stop_pgd_max=early_counter_max,
                        mixup=True, y_a=y_a, y_b=y_b, lam=lam)
                else:
                    delta, iter_counts = attack_pgd(model, X, y, epsilon_sche, pgd_alpha_sche, args.attack_iters, restarts_sche, args.norm, 
                        early_stop=args.earlystopPGD, early_stop_pgd_max=early_counter_max, multitarget=args.use_multitarget,
                        use_DLRloss=args.use_DLRloss, use_CWloss=args.use_CWloss, 
                        epoch=epoch_now, totalepoch=args.epochs, gamma=0.8,
                        use_adaptive=args.use_adaptive, s_HE=args.s_FN,
                        fast_better=args.fast_better, BNeval=args.BNeval)

                record_iter = torch.cat((record_iter, iter_counts))

                delta = delta.detach()
            elif args.attack == 'fgsm':
                delta,_ = attack_pgd(model, X, y, epsilon, args.fgsm_alpha*epsilon, 1, 1, args.norm, fast_better=args.fast_better)
                delta = delta.detach()
            # Standard training
            elif args.attack == 'none':
                delta = torch.zeros_like(X)


            adv_input = normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit))
            adv_input.requires_grad = True
            robust_output = model(adv_input)

            

            
            # Training losses
            if args.mixup:
                clean_input = normalize(X)
                clean_input.requires_grad = True     
                output = model(clean_input)
                robust_loss = mixup_criterion(criterion, robust_output, y_a, y_b, lam)

            elif args.mixture:
                clean_input = normalize(X)
                clean_input.requires_grad = True     
                output = model(clean_input)
                robust_loss = args.mixture_alpha * criterion(robust_output, y) + (1-args.mixture_alpha) * criterion(output, y)

            else:
                clean_input = normalize(X)
                clean_input.requires_grad = True     
                output = model(clean_input)
                if args.focalloss:
                    criterion_nonreduct = nn.CrossEntropyLoss(reduction='none')
                    robust_confidence = F.softmax(robust_output, dim=1)[:, y].detach()
                    robust_loss = (criterion_nonreduct(robust_output, y) * ((1. - robust_confidence) ** args.focallosslambda)).mean()

                elif args.use_DLRloss:
                    beta_ = 0.8 * epoch_now / args.epochs
                    robust_loss = (1. - beta_) * F.cross_entropy(robust_output, y) + beta_ * dlr_loss(robust_output, y)

                elif args.use_CWloss:
                    beta_ = 0.8 * epoch_now / args.epochs
                    robust_loss = (1. - beta_) * F.cross_entropy(robust_output, y) + beta_ * CW_loss(robust_output, y)

                elif args.use_FNandWN:
                    #print('use FN and WN with margin')
                    robust_loss = criterion(args.s_FN * robust_output - onehot_target_withmargin_HE, y)

                else:
                    robust_loss = criterion(robust_output, y)




            if args.l1:
                for name,param in model.named_parameters():
                    if 'bn' not in name and 'bias' not in name:
                        robust_loss += args.l1*param.abs().sum()


            opt.zero_grad()
            robust_loss.backward()
            opt.step()

            
            clean_input = normalize(X)
            clean_input.requires_grad = True     
            output = model(clean_input)
            if args.mixup:
                loss = mixup_criterion(criterion, output, y_a, y_b, lam)
            else:
                loss = criterion(output, y) 

            # Get the gradient norm values
            input_grads = torch.autograd.grad(loss, clean_input, create_graph=False)[0]

            # Record the statstic values
            train_robust_loss += robust_loss.item() * y.size(0)
            train_robust_acc += (robust_output.max(1)[1] == y).sum().item()
            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)
            train_grad += input_grads.abs().sum()

        train_time = time.time()
        if args.earlystopPGD:
            print('Iter mean: ', record_iter.mean().item(), ' Iter std:  ', record_iter.std().item())
        print('Learning rate: ', lr)
        #print('Eps: ', epsilon_sche)
        # Evaluate on test data
        model.eval()
        test_loss = 0
        test_acc = 0
        test_robust_loss = 0
        test_robust_acc = 0
        test_n = 0
        test_grad = 0
        for i, batch in enumerate(test_batches):
            X, y = batch['input'], batch['target']

            # Random initialization
            if args.attack == 'none':
                delta = torch.zeros_like(X)
            else:
                delta, _ = attack_pgd(model, X, y, test_epsilon, test_pgd_alpha, args.attack_iters, args.restarts, args.norm, early_stop=False)
            delta = delta.detach()

            adv_input = normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit))
            adv_input.requires_grad = True
            robust_output = model(adv_input)
            robust_loss = criterion(robust_output, y)

            clean_input = normalize(X)
            clean_input.requires_grad = True     
            output = model(clean_input)
            loss = criterion(output, y)

            # Get the gradient norm values
            input_grads = torch.autograd.grad(loss, clean_input, create_graph=False)[0]

            test_robust_loss += robust_loss.item() * y.size(0)
            test_robust_acc += (robust_output.max(1)[1] == y).sum().item()
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            test_n += y.size(0)
            test_grad += input_grads.abs().sum()

        test_time = time.time()

        if args.val:
            val_loss = 0
            val_acc = 0
            val_robust_loss = 0
            val_robust_acc = 0
            val_n = 0
            for i, batch in enumerate(val_batches):
                X, y = batch['input'], batch['target']

                # Random initialization
                if args.attack == 'none':
                    delta = torch.zeros_like(X)
                else:
                    delta, _ = attack_pgd(model, X, y, test_epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm, early_stop=False)
                delta = delta.detach()

                robust_output = model(normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)))
                robust_loss = criterion(robust_output, y)

                output = model(normalize(X))
                loss = criterion(output, y)

                val_robust_loss += robust_loss.item() * y.size(0)
                val_robust_acc += (robust_output.max(1)[1] == y).sum().item()
                val_loss += loss.item() * y.size(0)
                val_acc += (output.max(1)[1] == y).sum().item()
                val_n += y.size(0)

        if not args.eval:
            # logger.info('%d \t %.1f \t  %.1f \t  %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t  %.4f \t  %.4f  %.4f \t %.4f \t  %.4f',
            #     epoch, train_time - start_time, test_time - train_time, lr,
            #     train_loss/train_n, train_grad/train_n, train_acc/train_n, train_robust_loss/train_n, train_robust_acc/train_n,
            #     test_loss/test_n, test_acc/test_n, test_robust_loss/test_n, test_robust_acc/test_n)
            logger.info('%d \t %.4f \t %.4f \t %.4f \t %.4f',
                epoch, train_acc/train_n, train_robust_acc/train_n, test_acc/test_n, test_robust_acc/test_n)

            # Save results
            train_loss_record.append(train_loss/train_n)
            train_acc_record.append(train_acc/train_n)
            train_robust_loss_record.append(train_robust_loss/train_n)
            train_robust_acc_record.append(train_robust_acc/train_n)
            train_grad_record.append(train_grad/train_n)

            np.savetxt(args.fname+'/train_loss_record.txt', np.array(train_loss_record))
            np.savetxt(args.fname+'/train_acc_record.txt', np.array(train_acc_record))
            np.savetxt(args.fname+'/train_robust_loss_record.txt', np.array(train_robust_loss_record))
            np.savetxt(args.fname+'/train_robust_acc_record.txt', np.array(train_robust_acc_record))
            np.savetxt(args.fname+'/train_grad_record.txt', np.array(train_grad_record))

            test_loss_record.append(test_loss/test_n)
            test_acc_record.append(test_acc/test_n)
            test_robust_loss_record.append(test_robust_loss/test_n)
            test_robust_acc_record.append(test_robust_acc/test_n)
            test_grad_record.append(test_grad/test_n)

            np.savetxt(args.fname+'/test_loss_record.txt', np.array(test_loss_record))
            np.savetxt(args.fname+'/test_acc_record.txt', np.array(test_acc_record))
            np.savetxt(args.fname+'/test_robust_loss_record.txt', np.array(test_robust_loss_record))
            np.savetxt(args.fname+'/test_robust_acc_record.txt', np.array(test_robust_acc_record))
            np.savetxt(args.fname+'/test_grad_record.txt', np.array(test_grad_record))




            if args.val:
                logger.info('validation %.4f \t %.4f \t %.4f \t %.4f',
                    val_loss/val_n, val_acc/val_n, val_robust_loss/val_n, val_robust_acc/val_n)

                if val_robust_acc/val_n > best_val_robust_acc:
                    torch.save({
                            'state_dict':model.state_dict(),
                            'test_robust_acc':test_robust_acc/test_n,
                            'test_robust_loss':test_robust_loss/test_n,
                            'test_loss':test_loss/test_n,
                            'test_acc':test_acc/test_n,
                            'val_robust_acc':val_robust_acc/val_n,
                            'val_robust_loss':val_robust_loss/val_n,
                            'val_loss':val_loss/val_n,
                            'val_acc':val_acc/val_n,
                        }, os.path.join(args.fname, f'model_val.pth'))
                    best_val_robust_acc = val_robust_acc/val_n

            # save checkpoint
            if epoch > 99 or (epoch+1) % args.chkpt_iters == 0 or epoch+1 == epochs:
                torch.save(model.state_dict(), os.path.join(args.fname, f'model_{epoch}.pth'))
                torch.save(opt.state_dict(), os.path.join(args.fname, f'opt_{epoch}.pth'))

            # save best
            if test_robust_acc/test_n > best_test_robust_acc:
                torch.save({
                        'state_dict':model.state_dict(),
                        'test_robust_acc':test_robust_acc/test_n,
                        'test_robust_loss':test_robust_loss/test_n,
                        'test_loss':test_loss/test_n,
                        'test_acc':test_acc/test_n,
                    }, os.path.join(args.fname, f'model_best.pth'))
                best_test_robust_acc = test_robust_acc/test_n
        else:
            logger.info('%d \t %.1f \t \t %.1f \t \t %.4f \t %.4f \t %.4f \t %.4f \t \t %.4f \t \t %.4f \t %.4f \t %.4f \t \t %.4f',
                epoch, train_time - start_time, test_time - train_time, -1,
                -1, -1, -1, -1,
                test_loss/test_n, test_acc/test_n, test_robust_loss/test_n, test_robust_acc/test_n)
            return


if __name__ == "__main__":
    main()
