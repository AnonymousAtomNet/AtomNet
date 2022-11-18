# Standard Library
import copy
import math
import warnings
# Import from third library
import torch
from torch import nn


def init_weights_constant(module, val=0):
    for m in module.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(
                m, nn.ConvTranspose2d):
            nn.init.constant_(m.weight.data, val)
            nn.init.constant_(m.bias.data, val)


def init_weights_normal(module, std=0.01):
    for m in module.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(
                m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight.data, std=std)
            if m.bias is not None:
                m.bias.data.zero_()


def init_weights_xavier(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(
                m, nn.ConvTranspose2d):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()


def init_weights_msra(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(
                m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight.data, a=1)
            if m.bias is not None:
                m.bias.data.zero_()


def init_bias_focal(module, cls_loss_type, init_prior, num_classes):
    if cls_loss_type == 'sigmoid':
        for m in module.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # to keep the torch random state
                m.bias.data.normal_(-math.log(1.0 / init_prior - 1.0), init_prior)
                torch.nn.init.constant_(m.bias, -math.log(1.0 / init_prior - 1.0))

    elif cls_loss_type == 'softmax':
        for m in module.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.bias.data.normal_(0, 0.01)
                for i in range(0, m.bias.data.shape[0], num_classes):
                    fg = m.bias.data[i + 1:i + 1 + num_classes - 1]
                    mu = torch.exp(fg).sum()
                    m.bias.data[i] = math.log(mu * (1.0 - init_prior) / init_prior)
    else:
        raise NotImplementedError(f'{cls_loss_type} is not supported')


def trunc_normal(tensor, mean=0., std=1., a=-2., b=2.):
    def _no_grad_trunc_normal_(tensor, mean, std, a, b):
        def norm_cdf(x):
            return (1. + math.erf(x / math.sqrt(2.))) / 2.

        if (mean < a - 2 * std) or (mean > b + 2 * std):
            warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                          "The distribution of values may be incorrect.",
                          stacklevel=2)

        with torch.no_grad():
            v = norm_cdf((a - mean) / std)
            u = norm_cdf((b - mean) / std)
            tensor.uniform_(2 * v - 1, 2 * u - 1)
            tensor.erfinv_()
            tensor.mul_(std * math.sqrt(2.))
            tensor.add_(mean)
            tensor.clamp_(min=a, max=b)
            return tensor

    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def initialize(model, method, **kwargs):
    # initialize BN
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    # initialize Conv & FC
    if method == 'normal':
        init_weights_normal(model, **kwargs)
    elif method == 'msra':
        init_weights_msra(model)
    elif method == 'xavier':
        init_weights_xavier(model)
    else:
        raise NotImplementedError(f'{method} not supported')


def initialize_from_cfg(model, cfg):
    if cfg is None:
        initialize(model, 'normal', std=0.01)
        return

    cfg = copy.deepcopy(cfg)
    method = cfg.pop('method')
    initialize(model, method, **cfg)
