from decimal import Decimal
from turtle import pd
import torch.nn as nn
from torch.nn import Conv2d, Linear
import torch.nn.functional as F
from collections import OrderedDict

from models.utils import export_onnx_by_module, merge_block, merge_stage
from models.initializer import initialize_from_cfg
from models.normalize import build_norm_layer
from models.normal_blocks import get_same_padding, \
    build_activation, SEBlock, get_same_length, ConvBlock, \
    MBConvBlock, MBConvNextBlock, RegBottleneckBlock

from tools.query import query_model
from tools.flops import get_conv_and_linear_module_list
from tools.plot_flops import plot_model


__all__ = ['mobile']


class Conv(nn.Module):
    def __init__(self, in_channel, out_channel,
                 kernel_size=3, stride=1, branches=1, padding=1, bias=False):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size,
                               stride=stride, padding=padding, groups=1, bias=bias)),
            ('bn', nn.BatchNorm2d(out_channel))]))

    def forward(self, x):
        return self.conv(x)


class DepthwiseConv(nn.Module):
    def __init__(self, in_channel, out_channel,
                 kernel_size=3, stride=1, branches=1, padding=1, bias=False):
        super(DepthwiseConv, self).__init__()
        self.conv = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size,
                               stride=stride, padding=padding, groups=in_channel, bias=bias)),
            ('bn', nn.BatchNorm2d(out_channel))]))

    def forward(self, x):
        return self.conv(x)


class RepDepthwiseConv(nn.Module):
    def __init__(self, in_channel, out_channel,
                 kernel_size=3, stride=1, branches=4, padding=1, bias=False):
        super(RepDepthwiseConv, self).__init__()
        convs = []
        for _ in range(branches):
            convs.append(nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channel, bias=bias)),
                ('bn', nn.BatchNorm2d(out_channel))])))

        self.convs = nn.ModuleList(convs)
        self.conv1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, padding=0, bias=False)),
            ('bn', nn.BatchNorm2d(out_channel))]))
        self.stride = stride
        if stride != 2:
            self.bn = nn.BatchNorm2d(in_channel)

    def forward(self, x):
        if self.stride != 2:
            y = self.bn(x) + self.conv1(x)
            for conv in self.convs:
                y += conv(x)
        else:
            y = self.conv1(x)
            for conv in self.convs:
                y += conv(x)
        return y


class PointwiseConv(nn.Module):
    def __init__(self, in_channel, out_channel,
                 kernel_size=1, stride=1, branches=4, padding=0, bias=False):
        super(PointwiseConv, self).__init__()
        self.conv = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=bias)),
            ('bn', nn.BatchNorm2d(out_channel))]))

    def forward(self, x):
        return self.conv(x)


class RepPointwiseConv(nn.Module):
    def __init__(self, in_channel, out_channel,
                 kernel_size=1, stride=1, branches=4, padding=0, bias=False):
        super(RepPointwiseConv, self).__init__()
        convs = []
        for _ in range(branches):
            convs.append(nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size,
                                   stride=stride, padding=padding, bias=bias)),
                ('bn', nn.BatchNorm2d(out_channel))])))

        self.convs = nn.ModuleList(convs)
        if in_channel == out_channel:
            self.bn = nn.BatchNorm2d(out_channel)
        else:
            self.bn = None

    def forward(self, x):
        if self.bn:
            y = self.bn(x)
        else:
            y = 0
        for conv in self.convs:
            y += conv(x)
        return y


class MobileOneBlock(nn.Module):

    def __init__(self, in_channel, out_channel,
                 kernel_size=3, expand_ratio=1, stride=1, act_func='relu6',
                 use_se=False, act_func1='relu', act_func2='h_sigmoid', use_normal=False,
                 divisor=8, branches=4):
        super(MobileOneBlock, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel

        self.kernel_size = kernel_size
        self.expand_ratio = expand_ratio

        self.stride = stride
        self.act_func = act_func
        self.use_se = use_se
        self.divisor = divisor

        # build modules
        middle_channel = int(self.in_channel * self.expand_ratio)
        padding = get_same_padding(self.kernel_size)

        DepthConv = RepDepthwiseConv if branches > 0 else DepthwiseConv
        DepthConv = DepthConv if not use_normal else Conv
        PointConv = RepPointwiseConv if branches > 0 else PointwiseConv
        self.depth_conv = nn.Sequential(OrderedDict([
            ('conv', DepthConv(in_channel, middle_channel, self.kernel_size, stride=self.stride,
                               padding=padding, bias=False, branches=branches)),
            ('act', build_activation(self.act_func, inplace=False))
        ]))
        if self.use_se:
            self.depth_conv.add_module('se', SEBlock(middle_channel, act_func1=act_func1, act_func2=act_func2))

        self.point_linear = nn.Sequential(OrderedDict([
            ('conv', PointConv(middle_channel, self.out_channel, kernel_size=1, bias=False, branches=branches)),
            ('act', build_activation(self.act_func, inplace=False))
        ]))

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_linear(x)
        return x


class BigNAS_MobileOne(nn.Module):

    def __init__(self,
                 # search settings
                 out_channel=[64, 128, 256, 512],
                 depth=[2, 8, 10, 1],
                 kernel_size=[3, 3, 3, 3],
                 expand_ratio=[1, 1, 1, 1],
                 branches=0,
                 # other settings
                 act_stages=['relu', 'relu', 'relu', 'relu'],
                 se_stages=[False, False, False, False],
                 stride_stages=[4, 2, 2, 2],
                 patch_embed=True,
                 block_type_stage=['mobile_one', 'mobile_one', 'mobile_one', 'mobile_one'],
                 se_act_func1='relu', se_act_func2='hard_sigmoid',
                 dropout_rate=0,
                 divisor=8,
                 # bn and initializer
                 normalize={'type': 'solo_bn'},
                 initializer={'method': 'msra'},
                 # configuration for task
                 frozen_layers=[],
                 out_layers=[],
                 out_strides=[],
                 last_conv=True,
                 last_block=False,
                 pool_first=True,
                 last_bn=True,
                 last_channel=1280,
                 task='classification',
                 se_reduction='mid_channel',
                 num_classes=1000,
                 group_width=24):
        super(BigNAS_MobileOne, self).__init__()

        r"""
        Arguments:
        - out_channel (:obj:`list` of 9 (stages+3) ints): channel list
        - depth (:obj:`list` of 9 (stages+3) ints): depth list for stages
        - kernel_size (:obj:`list` of 9 (stages+3) or 24 (blocks+3) ints): kernel size list for blocks
        - expand_ratio (:obj:`list` of 9 (stages+3) or 24 (blocks+3) ints): expand ratio list for blocks
        - act_stages(:obj:`list` of 9 (stages+3) ints): activation list for blocks
        - stride_stages (:obj:`list` of 9 (stages+3) ints): stride list for stages
        - se_stages (:obj:`list` of 9 (stages+3) ints): se list for stages
        - se_act_func1(:obj:`str`: first activation function for se block)
        - se_act_func1(:obj:`str`: second activation function for se block)
        - dropout_rate (:obj:`float`): dropout rate
        - divisor(:obj:`int`): divisor for channels
        - normalize (:obj:`dict`): configurations for normalize
        - initializer (:obj:`dict`): initializer method
        - frozen_layers (:obj:`list` of :obj:`int`): Index of frozen layers, [0,4]
        - out_layers (:obj:`list` of :obj:`int`): Index of output layers, [0,4]
        - out_strides (:obj:`list` of :obj:`int`): Stride of outputs
        - task (:obj:`str`): Type of task, 'classification' or object 'detection'
        - num_classes (:obj:`int`): number of classification classes
        """

        global NormLayer

        NormLayer = build_norm_layer(normalize)

        self.depth = depth
        self.out_channel = out_channel
        self.kernel_size = get_same_length(kernel_size, depth)
        self.expand_ratio = get_same_length(expand_ratio, depth)

        self.act_stages = act_stages
        self.stride_stages = stride_stages
        self.se_stages = se_stages
        self.se_act_func1 = se_act_func1
        self.se_act_func2 = se_act_func2
        self.dropout_rate = dropout_rate
        self.divisor = divisor

        self.frozen_layers = frozen_layers
        self.out_layers = out_layers
        self.out_strides = out_strides
        self.task = task

        input_channel = 3
        self.stage_out_idx = []

        blocks = []

        stage_num = 0
        _block_idx = 0
        for s, act_func, use_se, n_block in zip(self.stride_stages, self.act_stages,
                                                self.se_stages, self.depth):
            output_channel = self.out_channel[stage_num]
            block_type = block_type_stage[stage_num]
            stage_num += 1
            for i in range(n_block):
                ks_list = self.kernel_size[_block_idx]
                expand_ratio = self.expand_ratio[_block_idx]
                if i == 0:
                    stride = s
                else:
                    stride = 1
                if block_type == 'conv':
                    mobile_inverted_conv = ConvBlock(
                        in_channel=input_channel, out_channel=output_channel, kernel_size=ks_list,
                        stride=stride, act_func=act_func, same_pad=True
                    )
                elif block_type == 'mbconv':
                    mobile_inverted_conv = MBConvBlock(
                        in_channel=input_channel, out_channel=output_channel, kernel_size=ks_list,
                        expand_ratio=expand_ratio, stride=stride, act_func=act_func, act_func1=self.se_act_func1,
                        act_func2=self.se_act_func2, use_se=use_se, divisor=self.divisor, se_reduction=se_reduction)
                elif block_type == 'mbconvnext':
                    mobile_inverted_conv = MBConvNextBlock(
                        in_channel=input_channel, out_channel=output_channel, kernel_size=ks_list,
                        expand_ratio=expand_ratio, stride=stride, act_func=act_func, act_func1=self.se_act_func1,
                        act_func2=self.se_act_func2, use_se=use_se, divisor=self.divisor, se_reduction=se_reduction)
                elif block_type == 'mobile_one':
                    mobile_inverted_conv = MobileOneBlock(
                        in_channel=input_channel, out_channel=output_channel, kernel_size=ks_list,
                        expand_ratio=expand_ratio, stride=stride, act_func=act_func, act_func1=self.se_act_func1,
                        act_func2=self.se_act_func2, use_se=use_se, divisor=self.divisor, branches=branches)
                elif block_type == 'mobile_one_normal':
                    mobile_inverted_conv = MobileOneBlock(
                        in_channel=input_channel, out_channel=output_channel, kernel_size=ks_list,
                        expand_ratio=expand_ratio, stride=stride, act_func=act_func, act_func1=self.se_act_func1,
                        act_func2=self.se_act_func2, use_se=use_se, divisor=self.divisor,
                        branches=branches, use_normal=True)
                elif block_type == 'regblock':
                    mobile_inverted_conv = RegBottleneckBlock(
                        in_channel=input_channel, out_channel=output_channel, kernel_size=ks_list,
                        expand_ratio=expand_ratio, stride=stride, act_func=act_func, act_func1=self.se_act_func1,
                        act_func2=self.se_act_func2, use_se=use_se, divisor=self.divisor, group_width=group_width)
                else:
                    raise ValueError

                blocks.append(mobile_inverted_conv)
                input_channel = output_channel
                _block_idx += 1
            self.stage_out_idx.append(_block_idx - 1)

        self.blocks = nn.ModuleList(blocks)

        # # building last several layers
        self.last_conv = last_conv if not last_block else False
        self.pool_first = pool_first
        self.last_block = last_block 
        if self.last_conv:
            self.fc = ConvBlock(
                in_channel=input_channel, out_channel=last_channel, kernel_size=1,
                stride=1, act_func=act_func, same_pad=True, use_bn=last_bn, bias=True if not last_bn else False
            )
            self.dropout = nn.Dropout(p=0.2)

        if self.last_block:
            self.fc = nn.Sequential(OrderedDict([
            ('expand', ConvBlock(
                in_channel=input_channel, out_channel=last_channel, kernel_size=1,
                stride=1, act_func=act_func, same_pad=True, use_bn=last_bn, bias=True if not last_bn else False
            )),
            ('squeeze', ConvBlock(
                in_channel=last_channel, out_channel=input_channel, kernel_size=1,
                stride=1, act_func=act_func, same_pad=True, use_bn=last_bn, bias=True if not last_bn else False
            ))
            ]))
            self.dropout = nn.Dropout(p=0.2)
        self.task = task
        if self.task == 'classification':
            self.classifier = nn.Linear(last_channel if last_conv else input_channel, num_classes)

        self.out_planes = [self.out_channel[i] for i in self.out_layers]

        if initializer is not None:
            initialize_from_cfg(self, initializer)

        # It's IMPORTANT when you want to freeze part of your backbone.
        # ALWAYS remember freeze layers in __init__ to avoid passing freezed params to optimizer
        self.freeze_layer()

    def forward(self, input):
        x = input['image'] if isinstance(input, dict) else input
        outs = []

        # blocks
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.stage_out_idx:
                outs.append(x)

        if self.last_conv or self.last_block:
            if self.pool_first:
                x = F.avg_pool2d(x, kernel_size=int(x.size(2)))
            x = self.fc(x)
            x = self.dropout(x)
            if self.pool_first:
                x = x.view(x.size(0), -1)
            outs[-1] = x

        if self.task == 'classification':
            if not self.pool_first:
                x = F.avg_pool2d(x, kernel_size=int(x.size(2)))
                x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
        features = [outs[i] for i in self.out_layers]
        return {'features': features, 'strides': self.get_outstrides()}

    def freeze_layer(self):
        """
        Freezing layers during training.
        """
        layers = []
        start_idx = 0
        for stage_out_idx in self.stage_out_idx:
            end_idx = stage_out_idx + 1
            stage = [self.blocks[i] for i in range(start_idx, end_idx)]
            layers.append(nn.Sequential(*stage))
            start_idx = end_idx

        for layer_idx in self.frozen_layers:
            layer = layers[layer_idx]
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        """
        Sets the module in training mode.
        This has any effect only on modules such as Dropout or BatchNorm.

        Returns:
            - module: self
        """
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.freeze_layer()
        return self

    def get_outplanes(self):
        """
        Get dimensions of the output tensors.

        Returns:
            - out (:obj:`list` of :obj:`int`)
        """
        return self.out_planes

    def get_outstrides(self):
        """
        Get strides of output tensors w.r.t inputs.

        Returns:
            - out (:obj:`list` of :obj:`int`)
        """
        return self.out_strides

    def analyse(self, input, name):
        x = input['image'] if isinstance(input, dict) else input
        meta_dict = plot_model(self, image_size=None, print_info=False, name=name, gray=False, input=x)
        block_dict = merge_block(meta_dict, self)
        stage_dict = merge_stage(block_dict, self)
        return stage_dict

    def get_conv_and_linear_module(self, input):
        module_list = get_conv_and_linear_module_list(self, input)
        module_table = {}
        for k in module_list:
            module_table[k['expr']] = k['input'] 
        return module_table

    def get_module_list(self, input, print_info=False, name='model'):
        # list of [expr, input_size, output_size, flops, params, mac, mac/flops]
        module_list = get_conv_and_linear_module_list(self, input, print_info=print_info, name=name)
        return [[i['expr'], i['input']] for i in module_list]

    def export_onnx_by_module(self, input):
        module_table = self.get_conv_and_linear_module(input)
        export_onnx_by_module(module_table)

    def get_latency_by_query_model(self, input):
        module_list = self.get_module_list(input)
        time = query_model(module_list)
        return time
    


def mobile(**kwargs):
    return BigNAS_MobileOne(**kwargs)


def pplcnet_025(**kwargs):
    net_kwargs = {
        'depth': [1, 1, 2, 2, 6, 2],
        'out_channel': [8, 8, 16, 32, 64, 128],
        'block_type_stage': ['conv', 'mobile_one', 'mobile_one', 'mobile_one', 'mobile_one', 'mobile_one'],
        'act_stages': ['hswish', 'hswish', 'hswish', 'hswish', 'hswish', 'hswish'],
        'se_stages': [False, False, False, False, False, True],
        'stride_stages': [2, 1, 2, 2, 2, 2],
        'kernel_size': [3, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 5, 5],
        'expand_ratio': [1, 1, 1, 1, 1, 1],
        'last_conv': True,
        'pool_first': True,
        'se_act_func1': 'relu',
        'se_act_func2': 'hard_sigmoid',
        'dropout_rate': 0.2,
        'branches': 0
    }
    net_kwargs.update(**kwargs)
    return mobile(**net_kwargs)


def pplcnet_035(**kwargs):
    net_kwargs = {
        'depth': [1, 1, 2, 2, 6, 2],
        'out_channel': [8, 16, 24, 48, 88, 176],
        'block_type_stage': ['conv', 'mobile_one', 'mobile_one', 'mobile_one', 'mobile_one', 'mobile_one'],
        'act_stages': ['hswish', 'hswish', 'hswish', 'hswish', 'hswish', 'hswish'],
        'se_stages': [False, False, False, False, False, True],
        'stride_stages': [2, 1, 2, 2, 2, 2],
        'kernel_size': [3, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 5, 5],
        'expand_ratio': [1, 1, 1, 1, 1, 1],
        'last_conv': True,
        'pool_first': True,
        'se_act_func1': 'relu',
        'se_act_func2': 'hard_sigmoid',
        'dropout_rate': 0.2,
        'branches': 0
    }
    net_kwargs.update(**kwargs)
    return mobile(**net_kwargs)


def pplcnet_050(**kwargs):
    net_kwargs = {
        'depth': [1, 1, 2, 2, 6, 2],
        'out_channel': [8, 16, 32, 64, 128, 256],
        'block_type_stage': ['conv', 'mobile_one', 'mobile_one', 'mobile_one', 'mobile_one', 'mobile_one'],
        'act_stages': ['hswish', 'hswish', 'hswish', 'hswish', 'hswish', 'hswish'],
        'se_stages': [False, False, False, False, False, True],
        'stride_stages': [2, 1, 2, 2, 2, 2],
        'kernel_size': [3, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 5, 5],
        'expand_ratio': [1, 1, 1, 1, 1, 1],
        'last_conv': True,
        'pool_first': True,
        'se_act_func1': 'relu',
        'se_act_func2': 'hard_sigmoid',
        'dropout_rate': 0.2,
        'branches': 0
    }
    net_kwargs.update(**kwargs)
    return mobile(**net_kwargs)


def pplcnet_075(**kwargs):
    net_kwargs = {
        'depth': [1, 1, 2, 2, 6, 2],
        'out_channel': [16, 24, 48, 96, 192, 384],
        'block_type_stage': ['conv', 'mobile_one', 'mobile_one', 'mobile_one', 'mobile_one', 'mobile_one'],
        'act_stages': ['hswish', 'hswish', 'hswish', 'hswish', 'hswish', 'hswish'],
        'se_stages': [False, False, False, False, False, True],
        'stride_stages': [2, 1, 2, 2, 2, 2],
        'kernel_size': [3, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 5, 5],
        'expand_ratio': [1, 1, 1, 1, 1, 1],
        'last_conv': True,
        'pool_first': True,
        'se_act_func1': 'relu',
        'se_act_func2': 'hard_sigmoid',
        'dropout_rate': 0.2,
        'branches': 0
    }
    net_kwargs.update(**kwargs)
    return mobile(**net_kwargs)


def pplcnet_100(**kwargs):
    net_kwargs = {
        'depth': [1, 1, 2, 2, 6, 2],
        'out_channel': [16, 32, 64, 128, 256, 512],
        'block_type_stage': ['conv', 'mobile_one', 'mobile_one', 'mobile_one', 'mobile_one', 'mobile_one'],
        'act_stages': ['hswish', 'hswish', 'hswish', 'hswish', 'hswish', 'hswish'],
        'se_stages': [False, False, False, False, False, True],
        'stride_stages': [2, 1, 2, 2, 2, 2],
        'kernel_size': [3, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 5, 5],
        'expand_ratio': [1, 1, 1, 1, 1, 1],
        'last_conv': True,
        'se_act_func1': 'relu',
        'se_act_func2': 'hard_sigmoid',
        'dropout_rate': 0.2,
        'branches': 0
    }
    net_kwargs.update(**kwargs)
    return mobile(**net_kwargs)


def mcunet_10fps(**kwargs):
    net_kwargs = {
        'depth': [1, 1, 3, 3, 2, 3, 2, 1],
        'out_channel': [24, 16, 16, 24, 48, 56, 112, 192],
        'kernel_size': [3, 3, 5, 3, 3, 7, 5, 5, 7, 5, 3, 5, 3, 3, 5, 3],
        'expand_ratio': [0, 1, 4, 3, 4, 5, 4, 5, 5, 4, 5, 4, 5, 6, 4, 6],
        'act_stages': ['relu6', 'relu6', 'relu6', 'relu6', 'relu6', 'relu6', 'relu6', 'relu6'],
        'se_stages': [False, False, False, False, False, False, False, False],
        'stride_stages': [2, 1, 2, 2, 2, 1, 2, 1],
        'last_conv': False,
        'block_type_stage': ['conv', 'mbconv', 'mbconv', 'mbconv', 'mbconv', 'mbconv', 'mbconv', 'mbconv']
    }
    net_kwargs.update(**kwargs)
    return mobile(**net_kwargs)


def mcunet_5fps(**kwargs):
    net_kwargs = {
        'depth': [1, 1, 2, 2, 2, 3, 3, 1],
        'out_channel': [16, 8, 16, 24, 40, 48, 96, 160],
        'kernel_size': [3, 3, 3, 3, 7, 3, 3, 7, 5, 3, 3, 7, 5, 5, 3],
        'expand_ratio': [0, 1, 4, 3, 3, 5, 5, 4, 4, 3, 4, 5, 4, 4, 6],
        'act_stages': ['relu6', 'relu6', 'relu6', 'relu6', 'relu6', 'relu6', 'relu6', 'relu6'],
        'se_stages': [False, False, False, False, False, False, False, False],
        'stride_stages': [2, 1, 2, 2, 2, 1, 2, 1],
        'last_conv': False,
        'block_type_stage': ['conv', 'mbconv', 'mbconv', 'mbconv', 'mbconv', 'mbconv', 'mbconv', 'mbconv']
    }
    net_kwargs.update(**kwargs)
    return mobile(**net_kwargs)


def mcunet_256kb(**kwargs):
    net_kwargs = {
        'depth': [1, 1, 4, 3, 2, 3, 4, 1],
        'out_channel': [16, 8, 16, 24, 40, 48, 96, 160],
        'block_type_stage': ['conv', 'mbconv', 'mbconv', 'mbconv', 'mbconv', 'mbconv', 'mbconv', 'mbconv'],
        'act_stages': ['relu6', 'relu6', 'relu6', 'relu6', 'relu6', 'relu6', 'relu6', 'relu6'],
        'se_stages': [False, False, False, False, False, False, False, False],
        'stride_stages': [2, 1, 2, 2, 2, 1, 2, 1],
        'kernel_size': [3, 3, 5, 7, 3, 5, 3, 7, 5, 7, 5, 3, 5, 3, 5, 5, 5, 3, 5],
        'expand_ratio': [0, 1, 3, 6, 5, 5, 5, 6, 6, 4, 5, 5, 5, 4, 6, 4, 3, 4, 5],
        'last_conv': False,
    }
    net_kwargs.update(**kwargs)
    return mobile(**net_kwargs)


def mcunet_320kb(**kwargs):
    net_kwargs = {
        'depth': [1, 1, 4, 3, 3, 3, 3, 1],
        'out_channel': [16, 8, 16, 24, 40, 48, 96, 160],
        'kernel_size': [3, 3, 7, 3, 7, 5, 5, 5, 5, 3, 7, 5, 5, 7, 3, 3, 7, 3, 7],
        'expand_ratio': [0, 1, 3, 5, 5, 4, 5, 5, 5, 5, 6, 4, 5, 5, 5, 6, 5, 4, 5],
        'act_stages': ['relu6', 'relu6', 'relu6', 'relu6', 'relu6', 'relu6', 'relu6', 'relu6'],
        'se_stages': [False, False, False, False, False, False, False, False],
        'stride_stages': [2, 1, 2, 2, 2, 1, 2, 1],
        'last_conv': False,
        'block_type_stage': ['conv', 'mbconv', 'mbconv', 'mbconv', 'mbconv', 'mbconv', 'mbconv', 'mbconv']
    }
    net_kwargs.update(**kwargs)
    return mobile(**net_kwargs)


def mcunet_512kb(**kwargs):
    net_kwargs = {
        'depth': [1, 1, 3, 3, 3, 3, 3, 1],
        'out_channel': [32, 16, 24, 40, 80, 96, 192, 320],
        'kernel_size': [3, 3, 7, 3, 5, 7, 3, 7, 7, 3, 7, 3, 5, 5, 7, 7, 5, 5],
        'expand_ratio': [0, 1, 3, 5, 4, 5, 4, 4, 3, 3, 3, 4, 3, 3, 4, 3, 3, 4],
        'act_stages': ['relu6', 'relu6', 'relu6', 'relu6', 'relu6', 'relu6', 'relu6', 'relu6'],
        'se_stages': [False, False, False, False, False, False, False, False],
        'stride_stages': [2, 1, 2, 2, 2, 1, 2, 1],
        'last_conv': False,
        'block_type_stage': ['conv', 'mbconv', 'mbconv', 'mbconv', 'mbconv', 'mbconv', 'mbconv', 'mbconv']
    }
    net_kwargs.update(**kwargs)
    return mobile(**net_kwargs)


def tinynet_a(**kwargs):
    net_kwargs = {
        'depth': [1, 1, 2, 2, 4, 4, 5, 1],
        'out_channel': [32, 16, 24, 40, 80, 112, 192, 320],
        'kernel_size': [3, 3, 3, 5, 3, 5, 5, 3],
        'expand_ratio': [0, 1, 6, 6, 6, 6, 6, 6],
        'act_stages': ['silu', 'silu', 'silu', 'silu', 'silu', 'silu', 'silu', 'silu'],
        'se_stages': [False, True, True, True, True, True, True, True],
        'stride_stages': [2, 1, 2, 2, 2, 1, 2, 1],
        'block_type_stage': ['conv', 'mbconv', 'mbconv', 'mbconv', 'mbconv', 'mbconv', 'mbconv', 'mbconv'],
        'last_conv': True,
        'pool_first': False,
        'last_channel': 1280,
        'last_bn': True,
        'se_act_func1': 'swish',
        'se_act_func2': 'sigmoid',
        'se_reduction': 'in_channel',
        'dropout_rate': 0.2,
    }
    net_kwargs.update(**kwargs)
    return mobile(**net_kwargs)


def tinynet_b(**kwargs):
    # image_size = 106
    net_kwargs = {
        'depth': [1, 1, 2, 2, 3, 3, 4, 1],
        'out_channel': [32, 16, 24, 32, 64, 88, 144, 240],
        'kernel_size': [3, 3, 3, 5, 3, 5, 5, 3],
        'expand_ratio': [0, 1, 6, 6, 6, 6, 6, 6],
        'act_stages': ['silu', 'silu', 'silu', 'silu', 'silu', 'silu', 'silu', 'silu'],
        'se_stages': [False, True, True, True, True, True, True, True],
        'stride_stages': [2, 1, 2, 2, 2, 1, 2, 1],
        'block_type_stage': ['conv', 'mbconv', 'mbconv', 'mbconv', 'mbconv', 'mbconv', 'mbconv', 'mbconv'],
        'last_conv': True,
        'pool_first': False,
        'last_bn': True,
        'last_channel': 1280,
        'se_act_func1': 'swish',
        'se_act_func2': 'sigmoid',
        'se_reduction': 'in_channel',
        'dropout_rate': 0.2,
    }
    net_kwargs.update(**kwargs)
    return mobile(**net_kwargs)


def tinynet_c(**kwargs):
    # image_size = 106
    net_kwargs = {
        'depth': [1, 1, 2, 2, 3, 3, 3, 1],
        'out_channel': [32, 8, 16, 24, 40, 64, 104, 176],
        'kernel_size': [3, 3, 3, 5, 3, 5, 5, 3],
        'expand_ratio': [0, 1, 6, 6, 6, 6, 6, 6],
        'act_stages': ['silu', 'silu', 'silu', 'silu', 'silu', 'silu', 'silu', 'silu'],
        'se_stages': [False, True, True, True, True, True, True, True],
        'stride_stages': [2, 1, 2, 2, 2, 1, 2, 1],
        'block_type_stage': ['conv', 'mbconv', 'mbconv', 'mbconv', 'mbconv', 'mbconv', 'mbconv', 'mbconv'],
        'last_conv': True,
        'pool_first': False,
        'last_channel': 1280,
        'last_bn': True,
        'se_act_func1': 'silu',
        'se_act_func2': 'sigmoid',
        'se_reduction': 'in_channel',
        'dropout_rate': 0.2,
    }
    net_kwargs.update(**kwargs)
    return mobile(**net_kwargs)


def tinynet_d(**kwargs):
    # image_size = 106
    net_kwargs = {
        'depth': [1, 1, 1, 1, 2, 2, 3, 1],
        'out_channel': [32, 8, 16, 24, 40, 64, 104, 176],
        'kernel_size': [3, 3, 3, 5, 3, 5, 5, 3],
        'expand_ratio': [0, 1, 6, 6, 6, 6, 6, 6],
        'act_stages': ['silu', 'silu', 'silu', 'silu', 'silu', 'silu', 'silu', 'silu'],
        'se_stages': [False, True, True, True, True, True, True, True],
        'stride_stages': [2, 1, 2, 2, 2, 1, 2, 1],
        'block_type_stage': ['conv', 'mbconv', 'mbconv', 'mbconv', 'mbconv', 'mbconv', 'mbconv', 'mbconv'],
        'last_conv': True,
        'pool_first': False,
        'last_channel': 1280,
        'last_bn': True,
        'se_act_func1': 'silu',
        'se_act_func2': 'sigmoid',
        'se_reduction': 'in_channel',
        'dropout_rate': 0.2,
    }
    net_kwargs.update(**kwargs)
    return mobile(**net_kwargs)


def tinynet_e(**kwargs):
    # image_size = 106
    net_kwargs = {
        'depth': [1, 1, 1, 1, 2, 2, 2, 1],
        'out_channel': [32, 8, 16, 24, 40, 56, 96, 160],
        'kernel_size': [3, 3, 3, 5, 3, 5, 5, 3],
        'expand_ratio': [0, 1, 6, 6, 6, 6, 6, 6],
        'act_stages': ['silu', 'silu', 'silu', 'silu', 'silu', 'silu', 'silu', 'silu'],
        'se_stages': [False, True, True, True, True, True, True, True],
        'stride_stages': [2, 1, 2, 2, 2, 1, 2, 1],
        'block_type_stage': ['conv', 'mbconv', 'mbconv', 'mbconv', 'mbconv', 'mbconv', 'mbconv', 'mbconv'],
        'last_conv': True,
        'pool_first': False,
        'last_channel': 1280,
        'last_bn': True,
        'se_act_func1': 'swish',
        'se_act_func2': 'sigmoid',
        'se_reduction': 'in_channel',
        'dropout_rate': 0.2,
    }
    net_kwargs.update(**kwargs)
    return mobile(**net_kwargs)


def mobilenetv2_035(**kwargs):
    # image_size = 106
    net_kwargs = {
        'depth': [1, 1, 2, 3, 4, 3, 3, 1],
        'out_channel': [16, 8, 8, 16, 24, 32, 56, 112],
        'kernel_size': [3, 3, 3, 3, 3, 3, 3, 3],
        'expand_ratio': [0, 1, 6, 6, 6, 6, 6, 6],
        'act_stages': ['relu6' for _ in range(8)],
        'se_stages': [False for _ in range(8)],
        'stride_stages': [2, 1, 2, 2, 2, 1, 2, 1],
        'block_type_stage': ['conv', 'mbconv', 'mbconv', 'mbconv', 'mbconv', 'mbconv', 'mbconv', 'mbconv'],
        'last_conv': True,
        'pool_first': False,
        'last_channel': 1280,
        'last_bn': True,
        'se_act_func1': 'silu',
        'se_act_func2': 'sigmoid',
        'se_reduction': 'in_channel',
        'dropout_rate': 0.2,
    }
    net_kwargs.update(**kwargs)
    return mobile(**net_kwargs)


def mobilenetv2_050(**kwargs):
    # image_size = 106
    net_kwargs = {
        'depth': [1, 1, 2, 3, 4, 3, 3, 1],
        'out_channel': [16, 8, 16, 16, 32, 48, 80, 160],
        'kernel_size': [3, 3, 3, 3, 3, 3, 3, 3],
        'expand_ratio': [0, 1, 6, 6, 6, 6, 6, 6],
        'act_stages': ['relu6' for _ in range(8)],
        'se_stages': [False for _ in range(8)],
        'stride_stages': [2, 1, 2, 2, 2, 1, 2, 1],
        'block_type_stage': ['conv', 'mbconv', 'mbconv', 'mbconv', 'mbconv', 'mbconv', 'mbconv', 'mbconv'],
        'last_conv': True,
        'pool_first': False,
        'last_channel': 1280,
        'last_bn': True,
        'se_act_func1': 'silu',
        'se_act_func2': 'sigmoid',
        'se_reduction': 'in_channel',
        'dropout_rate': 0.2,
    }
    net_kwargs.update(**kwargs)
    return mobile(**net_kwargs)


def mobilenetv2_075(**kwargs):
    # image_size = 106
    net_kwargs = {
        'depth': [1, 1, 2, 3, 4, 3, 3, 1],
        'out_channel': [24, 16, 24, 24, 48, 72, 120, 240],
        'kernel_size': [3, 3, 3, 3, 3, 3, 3, 3],
        'expand_ratio': [0, 1, 6, 6, 6, 6, 6, 6],
        'act_stages': ['relu6' for _ in range(8)],
        'se_stages': [False for _ in range(8)],
        'stride_stages': [2, 1, 2, 2, 2, 1, 2, 1],
        'block_type_stage': ['conv', 'mbconv', 'mbconv', 'mbconv', 'mbconv', 'mbconv', 'mbconv', 'mbconv'],
        'last_conv': True,
        'pool_first': False,
        'last_channel': 1280,
        'last_bn': True,
        'se_act_func1': 'silu',
        'se_act_func2': 'sigmoid',
        'se_reduction': 'in_channel',
        'dropout_rate': 0.2,
    }
    net_kwargs.update(**kwargs)
    return mobile(**net_kwargs)


def mobilenetv2_100(**kwargs):
    # image_size = 106
    net_kwargs = {
        'depth': [1, 1, 2, 3, 4, 3, 3, 1],
        'out_channel': [32, 16, 24, 32, 64, 96, 160, 320],
        'kernel_size': [3, 3, 3, 3, 3, 3, 3, 3],
        'expand_ratio': [0, 1, 6, 6, 6, 6, 6, 6],
        'act_stages': ['relu6' for _ in range(8)],
        'se_stages': [False for _ in range(8)],
        'stride_stages': [2, 1, 2, 2, 2, 1, 2, 1],
        'block_type_stage': ['conv', 'mbconv', 'mbconv', 'mbconv', 'mbconv', 'mbconv', 'mbconv', 'mbconv'],
        'last_conv': True,
        'pool_first': False,
        'last_bn': True,
        'last_channel': 1280,
        'se_act_func1': 'silu',
        'se_act_func2': 'sigmoid',
        'se_reduction': 'in_channel',
        'dropout_rate': 0.2,
    }
    net_kwargs.update(**kwargs)
    return mobile(**net_kwargs)


def mobilenetv2_140(**kwargs):
    # image_size = 106
    net_kwargs = {
        'depth': [1, 1, 2, 3, 4, 3, 3, 1],
        'out_channel': [48, 24, 32, 48, 88, 136, 224, 448],
        'kernel_size': [3, 3, 3, 3, 3, 3, 3, 3],
        'expand_ratio': [0, 1, 6, 6, 6, 6, 6, 6],
        'act_stages': ['relu6' for _ in range(8)],
        'se_stages': [False for _ in range(8)],
        'stride_stages': [2, 1, 2, 2, 2, 1, 2, 1],
        'block_type_stage': ['conv', 'mbconv', 'mbconv', 'mbconv', 'mbconv', 'mbconv', 'mbconv', 'mbconv'],
        'last_conv': True,
        'pool_first': False,
        'last_channel': 1792,
        'last_bn': True,
        'se_act_func1': 'silu',
        'se_act_func2': 'sigmoid',
        'se_reduction': 'in_channel',
        'dropout_rate': 0.2,
    }
    net_kwargs.update(**kwargs)
    return mobile(**net_kwargs)

def atomnet_base(**kwargs):
    net_kwargs = {
        'depth': [1, 1, 2, 2, 6, 2],
        'out_channel': [16, 16, 64, 128, 128, 256],
        'block_type_stage': ['conv', 'mobile_one', 'mobile_one', 'mbconvnext', 'mbconv', 'mbconv'],
        'act_stages': ['relu', 'relu', 'relu', 'relu', 'relu', 'relu'],
        'se_stages': [False, False, False, False, False, False],
        'stride_stages': [2, 1, 2, 2, 2, 2],
        'kernel_size': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        'expand_ratio': [1, 1, 1, 1, 1, 1],
        'last_conv': False,
        'pool_first': False,
        'se_act_func1': 'relu',
        'se_act_func2': 'hard_sigmoid',
        'dropout_rate': 0.2,
        'branches': 0
    }
    net_kwargs.update(**kwargs)
    return mobile(**net_kwargs)

if __name__ == '__main__':
    import torch
    resolution_dict = {'tinynet_a': 192,
                       'tinynet_b': 188,
                       'tinynet_c': 184,
                       'tinynet_d': 152,
                       'tinynet_e': 106,
                       'pplcnet_025': 224,
                       'pplcnet_035': 224,
                       'pplcnet_050': 224,
                       'pplcnet_075': 224,
                       'pplcnet_100': 224,
                       'mcunet_10fps': 48,
                       'mcunet_5fps': 96,
                       'mcunet_256kb': 160,
                       'mcunet_320kb': 172,
                       'mcunet_512kb': 160,
                       'mobilenetv2_035': 224,
                       'mobilenetv2_050': 224,
                       'mobilenetv2_075': 224,
                       'mobilenetv2_100': 224,
                       'mobilenetv2_140': 224,
                       }

    import os
    for k, v in resolution_dict.items():
        #if k != 'tinynet_e':
        #    continue
        print(k)
        name = k
        cls_model = eval(name)()
        input = torch.randn([1, 3, v, v])
        cls_model.analyse(input, name)
        torch.onnx.export(cls_model, input, name+'.onnx')
        os.system('python -m onnxsim {} {}'.format(name+'.onnx', name+'.onnx'))
        cls_model.get_module_list(input, print_info=True, name=name)
        #print('query time', cls_model.get_latency_by_query_model(input))
        #torch.onnx.export(cls_model, input, name+'.onnx')
        #cls_model.export_onnx_by_module(input)
