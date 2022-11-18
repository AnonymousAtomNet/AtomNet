# Standard Library
import numpy as np
import copy
# Import from third library
import torch


class FrozenBatchNorm2d(torch.nn.BatchNorm2d):
    def __init__(self, *args, **kwargs):
        super(FrozenBatchNorm2d, self).__init__(*args, **kwargs)
        self.training = False

    def train(self, mode=False):
        self.training = False
        for module in self.children():
            module.train(False)
        return self


def build_norm_layer(config):
    _norm_cfg = {
        'solo_bn': torch.nn.BatchNorm2d,
        'freeze_bn': FrozenBatchNorm2d,
        # 'gn': torch.nn.GroupNorm,
    }
    """
    Build normalization layer according to configurations.

    solo_bn (original bn): torch.nn.BatchNorm2d
    freeze_bn (frozen bn): torch.nn.BatchNorm2d with training type of False
    gn (group normalization): torch.nn.GroupNorm
    """
    assert isinstance(config, dict) and 'type' in config
    config = copy.deepcopy(config)
    norm_type = config.pop('type')
    config_kwargs = config.get('kwargs', {})

    def NormLayer(*args, **kwargs):
        return _norm_cfg[norm_type](*args, **kwargs, **config_kwargs)

    return NormLayer
