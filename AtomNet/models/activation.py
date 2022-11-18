import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_activation(act_func, inplace=False):
    act_func = act_func.lower()
    if act_func == 'relu':
        return nn.ReLU(inplace=inplace)
    elif act_func == 'leaky_relu':
        return nn.LeakyReLU()
    elif act_func == 'relu6':
        return nn.ReLU6(inplace=inplace)
    elif act_func == 'tanh':
        return nn.Tanh()
    elif act_func == 'sigmoid':
        return nn.Sigmoid()
    elif act_func == 'h_swish' or act_func == 'hard_swish' or act_func == 'hswish':
        return Hswish(inplace=inplace)
    elif act_func == 'h_sigmoid' or act_func == 'hard_sigmoid' or act_func == 'hsigmoid':
        return Hsigmoid(inplace=inplace)
    elif act_func == 'silu':
        if hasattr(nn, 'SiLU'):
            return nn.SiLU()
        else:
            return Swish()
    elif act_func == 'swish':
        return Swish()
    elif act_func == 'memoryefficientswish':
        return MemoryEfficientMish()
    elif act_func == 'Mish':
        return Mish()
    elif act_func == 'memoryefficientmish':
        return MemoryEfficientMish()
    elif act_func == 'frelu':
        return FReLU()
    elif act_func == 'gelu':
        return GELU()
    elif act_func == 'qgelu' or act_func == 'quick_gelu':
        return QGELU()
    elif act_func is None:
        return None
    else:
        raise ValueError('do not support: %s' % act_func)


class GELU(nn.Module):
    """
    Gaussian Error Linear Units, based on
    `"Gaussian Error Linear Units (GELUs)" <https://arxiv.org/abs/1606.08415>`
    """

    def __init__(self, approximate=True):
        super(GELU, self).__init__()
        self.approximate = approximate

    def forward(self, x):
        if self.approximate:
            cdf = 0.5 * (1.0 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
            return x * cdf
        else:
            return x * (torch.erf(x / math.sqrt(2)) + 1) / 2


class QGELU(nn.Module):
    """
    Gaussian Error Linear Units, based on
    `"Gaussian Error Linear Units (GELUs)" <https://arxiv.org/abs/1606.08415>`
    """

    def __init__(self, approximate=True):
        super(QGELU, self).__init__()
        self.approximate = approximate

    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class Hswish(nn.Module):

    def __init__(self, inplace=False):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class Hsigmoid(nn.Module):

    def __init__(self, inplace=False):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class Swish(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class MemoryEfficientSwish(nn.Module):
    class F(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x * torch.sigmoid(x)

        @staticmethod
        def backward(ctx, grad_output):
            x = ctx.saved_tensors[0]
            sx = torch.sigmoid(x)
            return grad_output * (sx * (1 + x * (1 - sx)))

    def forward(self, x):
        return self.F.apply(x)


# Mish https://github.com/digantamisra98/Mish ---------------------------
class Mish(nn.Module):
    @staticmethod
    def forward(x):
        return x * F.softplus(x).tanh()


class MemoryEfficientMish(nn.Module):
    class F(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x.mul(torch.tanh(F.softplus(x)))  # x * tanh(ln(1 + exp(x)))

        @staticmethod
        def backward(ctx, grad_output):
            x = ctx.saved_tensors[0]
            sx = torch.sigmoid(x)
            fx = F.softplus(x).tanh()
            return grad_output * (fx + x * sx * (1 - fx * fx))

    def forward(self, x):
        return self.F.apply(x)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# FReLU https://arxiv.org/abs/2007.11824 -------------------------------
class FReLU(nn.Module):
    def __init__(self, c1, k=3):  # ch_in, kernel
        super().__init__()
        self.conv = nn.Conv2d(c1, c1, k, 1, 1, groups=c1)
        self.bn = nn.BatchNorm2d(c1)

    def forward(self, x):
        return torch.max(x, self.bn(self.conv(x)))
