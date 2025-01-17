import torch
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
from .options import AverageMeter
from copy import deepcopy
EPS = 1E-20


def diff_in_weights(model, proxy):
    diff_dict = OrderedDict()
    model_state_dict = model.state_dict()
    proxy_state_dict = proxy.state_dict()
    for (old_k, old_w), (new_k, new_w) in zip(model_state_dict.items(), proxy_state_dict.items()):
        if len(old_w.size()) <= 1:
            continue
        if 'weight' in old_k:
            diff_w = new_w - old_w
            diff_dict[old_k] = old_w.norm() / (diff_w.norm() + EPS) * diff_w
    return diff_dict


def add_into_weights(model, diff, coeff=1.0):
    names_in_diff = diff.keys()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in names_in_diff:
                param.add_(coeff * diff[name])


def cos_sim(upg1,upg2):
    cos_sim = AverageMeter()
    cos_fct = nn.CosineSimilarity(dim=0)
    for (old_k, old_w), (new_k, new_w) in zip(upg1.items(),upg2.items()):
        current_cossim = cos_fct(old_w.view(-1)*-1,new_w.view(-1))
        cos_sim.update(current_cossim)
    return cos_sim.val

class AdvWeightPerturb(object):
    def __init__(self, model,gamma):
        super(AdvWeightPerturb, self).__init__()
        self.model = model
        self.proxy = deepcopy(model)
        self.proxy_optim = torch.optim.SGD(proxy.parameters(), lr=0.01)
        self.gamma = gamma

    def calc_awp(self, inputs_adv, targets,defined_loss):
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()
        
        loss = - defined_loss(self.proxy(inputs_adv), targets)

        self.proxy_optim.zero_grad()
        loss.backward()
        self.proxy_optim.step()

        # the adversary weight perturb
        diff = diff_in_weights(self.model, self.proxy)
        return diff

    def perturb(self, diff):
        add_into_weights(self.model, diff, coeff=1.0 * self.gamma)

    def restore(self, diff):
        add_into_weights(self.model, diff, coeff=-1.0 * self.gamma)




