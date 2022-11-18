import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from fvcore.nn import FlopCountAnalysis

def get_colors(type_dict):
    colors = []
    for k, v in type_dict.items():
        if v == 'Channel Mixing':
            colors.append('red')
        elif v == 'Token Mixing':
            colors.append('blue')
        else:
            colors.append('purple')
    return colors


def get_attention_flops(flops_dict, key):
    mhsa_flops = sum(flops_dict[key.replace('.proj', '')].values()) - sum(flops_dict[key].values()) - sum(
        flops_dict[key.replace('.proj', '.qkv')].values())

    return mhsa_flops


def get_module_type(cls_model, module_names):
    type_dict = {}
    for k in module_names:
        if k not in dict(cls_model.named_modules()).keys():
            type_dict[k] = 'Token Mixing'
            continue
        module = dict(cls_model.named_modules())[k]
        if isinstance(module, nn.Conv2d):
            if module.groups == 1 and module.kernel_size == (1, 1):
                type_dict[k] = 'Channel Mixing'
            elif module.groups != module.in_channels and module.kernel_size[0] > 1:
                type_dict[k] = 'Token & Channel Mixing'
            else:
                type_dict[k] = 'Token Mixing'
        elif isinstance(module, nn.Linear):
                type_dict[k] = 'Channel Mixing'
        if 'linear_token' in k:
            type_dict[k] = 'Token Mixing'
    return type_dict


def activation_table(subnet, input, module_names):
    max_channel = []
    module_table = {}
    handles = []

    def make_hook(module_name):
        def hook(module, data, out):
            module_table[module_name] = round(out.numel()/10**3, 2)
            if out.dim() == 4:
                max_channel.append(out.size()[1])

        return hook

    for n, m in subnet.named_modules():
        if n in module_names:
            handle = m.register_forward_hook(make_hook(n))
            handles.append(handle)
    subnet(input)
    for handle in handles:
        handle.remove()

    new_dict = {}
    for k in module_names:
        if k in module_table.keys():
            new_dict[k] = module_table[k]
        else:
            new_dict[k] = 0

    print('max channel', max(max_channel))
    return new_dict


def find_keys(table, key):
    for i in table.keys():
        if i == key[0:len(i)]:
            return i
    return None


def get_flops(cls_model, input, gray=False):
    flops_dict = {}
    flops = FlopCountAnalysis(cls_model, input)
    print('total', flops.total() / 10 ** 6)
    print('operator', flops.by_operator())
    for k, v in flops.by_module_and_operator().items():
        if len(v.keys()) != 0:
            upper_key = find_keys(flops_dict, k)
            if upper_key is not None:
                del flops_dict[upper_key]
            if list(v.keys()) == ['batch_norm'] or list(v.keys()) == ['layer_norm'] or list(v.keys()) == [
                'adaptive_avg_pool2d']:
                continue
            if 'attn.proj' in k:
                flops_dict[k.replace('.proj', '.mhsa')] = get_attention_flops(dict(flops.by_module_and_operator()),
                                                                                  k)
            flops_dict[k] = round(sum(v.values())/10**6, 5)

    return flops_dict


def get_params(cls_model, module_names):
    params_dict = {}
    for k in module_names:
        if k not in dict(cls_model.named_modules()).keys():
            params_dict[k] = 0
            continue
        module = dict(cls_model.named_modules())[k]
        total = 0
        for p in module.parameters():
            total += p.numel()
        params_dict[k] = round(total/10**6, 5)
    return params_dict


def plot_model(cls_model, image_size=None, print_info=False, name='test', gray=False, input=None):
    input = input if input is not None else torch.randn(1, 3 if not gray else 1, image_size, image_size)
    flops_dict = get_flops(cls_model, input, gray)
    param_dict = get_params(cls_model, list(flops_dict.keys()))
    type_dict = get_module_type(cls_model, list(flops_dict.keys()))
    activate_dict = activation_table(cls_model, input, list(flops_dict.keys()))
    meta_dict = {}
    for k in flops_dict.keys():
        if print_info:
            print(k, 'type', type_dict[k], 'flops', flops_dict[k], 'params', param_dict[k], 'activation', activate_dict[k])
        meta_dict[k] = {'type': type_dict[k], 'flops': flops_dict[k], 'params': param_dict[k], 'activation': activate_dict[k]}

    print('total layers ', len(flops_dict.keys()))
    print('sum flops', round(sum(flops_dict.values()), 2), 'm')
    plt.bar(np.arange(len(flops_dict.keys())), list(flops_dict.values()), color=get_colors(type_dict))
    plt.title('FLOPs Distribution with total ' + str(round(sum(flops_dict.values()), 2)) + 'M')
    plt.ylabel('FLOPs')
    plt.xlabel('Layer index')
    plt.savefig(name+'_flops')
    if print_info:
        plt.show()
    plt.cla()

    print('sum params', round(sum(param_dict.values()), 2), 'm')
    plt.bar(np.arange(len(param_dict.keys())), list(param_dict.values()), color=get_colors(type_dict))
    plt.title('Params Distribution with total ' + str(round(sum(param_dict.values()), 2)) + 'M')
    plt.ylabel('Params')
    plt.xlabel('Layer index')
    plt.savefig(name+'_params')
    if print_info:
        plt.show()
    plt.cla()

    print('input size', round(int(input.numel()), 2), 'k')
    print('max activation', round(max(activate_dict.values()), 2), 'k')
    print('sum activation', round(sum(activate_dict.values())/10**3, 2), 'm')
    plt.bar(np.arange(len(activate_dict.keys())), list(activate_dict.values()), color=get_colors(type_dict))
    plt.title('Activation Distribution with total ' + str(round(sum(activate_dict.values())/10**3, 2)) + 'M')
    plt.ylabel('Activation')
    plt.xlabel('Layer index')
    plt.savefig(name+'_mac')
    if print_info:
        plt.show()
    plt.cla()
    return meta_dict


if __name__ == '__main__':
    from modeldb.models.common import MODELDB_MODELS_REGISTRY
    cls_model = MODELDB_MODELS_REGISTRY['mcunet_256kb']()
    plot_model(cls_model, 160, print_info=True, name='mcunet_256kb')

    from modeldb.models.common import MODELDB_MODELS_REGISTRY
    cls_model = MODELDB_MODELS_REGISTRY['mcunet_320kb']()
    plot_model(cls_model, 176, print_info=True, name='mcunet_320kb')

    from modeldb.models.common import MODELDB_MODELS_REGISTRY
    cls_model = MODELDB_MODELS_REGISTRY['mcunet_512kb']()
    plot_model(cls_model, 160, print_info=True, name='mcunet_512kb')

