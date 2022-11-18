import os
import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

from tools.query import query
from fvcore.nn import FlopCountAnalysis, parameter_count


def get_flops(cls_model, input_size):
    input = torch.randn(1, 3, input_size, input_size)
    flops = FlopCountAnalysis(cls_model, input)
    print(flops.total() / 10 ** 6, 'M')
    print(parameter_count(cls_model)[''] / 10 ** 6, 'M')
    print(flops.by_operator())
    print((sum(flops.by_operator().values()) - flops.by_operator()['batch_norm']) / 10 ** 6, 'M')
    return parameter_count(cls_model)[''] / 10 ** 6, flops.total() / 10 ** 6


def get_onnx(cls_model, input_size, name):
    print(name)
    input = torch.randn(1, 3, input_size, input_size)
    torch.onnx.export(cls_model, input, name + '.onnx')
    os.system('python -m onnxsim {} {}'.format(name + '.onnx', name + '.onnx'))


def get_flops_and_onnx(cls_model, input_size, name):
    flash,flops = get_flops(cls_model, input_size)
    get_onnx(cls_model, input_size, name)
    return flash, flops


def compute_flops(module, output_height, output_width):
    flops = module.in_channels // module.groups * module.out_channels * output_height * output_width * module.kernel_size[0] * module.kernel_size[1] + module.out_channels
    return flops 


def compute_params(module):
    total = module.weight.numel() + module.bias.numel() if module.bias is not None else module.weight.numel()
    return total


def get_type(module):
    if isinstance(module, nn.Linear):
        return 'Dense'
    elif isinstance(module, nn.Conv2d):
        if module.kernel_size[0] > 1 and module.groups != 1:
            return 'DepthWise'
        elif module.kernel_size[0] > 1 and module.groups == 1:
            return 'Ordinary'
        elif module.kernel_size[0] == 1 and module.groups == 1:
            return 'PointWise'
    else:
        raise ValueError

def print_info_by_type(module_list, types=['PointWise', 'DepthWise', 'Ordinary', 'Dense']):
    table = {}
    for module_type in types:
        table[module_type] = {'flops': sum([k['flops'] for k in module_list if k['type'] == module_type]) / 10 ** 6,
                              'time': sum([k['query_time'] for k in module_list if k['type'] == module_type]),
                              'params': sum([k['params'] for k in module_list if k['type'] == module_type]) / 10 ** 6,
                              #'memory_access':sum([k['memory_access'] for k in module_list if k['type'] == module_type]) / 10 ** 6,
                              #'flops/memory_access': sum([k['flops/memory_access'] for k in module_list if k['type'] == module_type]),
                              'count': len([k['flops/memory_access'] for k in module_list if k['type'] == module_type])}
        print(module_type, table[module_type])
    return table



def get_conv_and_linear_module_list(model, input, print_info=False, name='model'):
    handles = []
    module_list = []

    def make_hook(module_name):
        def hook(module, data, out):
            input_feature_map = list(data[0].size())
            output_feature_map = list(out.size())
            params = compute_params(module)
            flops = compute_flops(module, output_feature_map[2], output_feature_map[3]) if isinstance(module, nn.Conv2d) else module.in_features * m.out_features
            memory_access = np.prod(np.array(input_feature_map)) + np.prod(np.array(output_feature_map)) + params
            time = query(module.__repr__(), input_feature_map) if print_info else 0
            module_list.append({'expr': module.__repr__(), 
                                'input': input_feature_map, # input size
                                'output': output_feature_map, # output size
                                'flops': flops,
                                'params': params,
                                'memory_access': memory_access,
                                'flops/memory_access': flops/memory_access,
                                'query_time': time, # query time
                                'type': get_type(module),
                                'flops/time': flops/time if time != 0 else None
                                #2.50481175e-05 * mac + 1.99952128e-06 * flops # predict time
                                #3.71461591e-05 * mac + 2.14122882e-06 * flops -0.62                # predict time 
                                })
        return hook

    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            handle = m.register_forward_hook(make_hook(n))
            handles.append(handle)
    
    model(input)
    for handle in handles:
        handle.remove()

    if print_info:
        flops_list = []
        mac_flops_list = []
        for i, k in enumerate(module_list):
            if k not in module_list[0:i]:
                flops_list.append(k['flops'])
                mac_flops_list.append(k['flops/memory_access'])
        plot_twins([i for i in range(len(flops_list))], flops_list, mac_flops_list, name=name)
        #sorted_module = sorted(module_list, key=lambda x:(1 / x[-2]), reverse=True)
        for i, k in enumerate(module_list):
            print(k)
        print('query time', sum([i['query_time'] for i in module_list]))
        print('total flops', sum([i['flops'] for i in module_list])/10**6)
        print('totoal params', sum([i['params'] for i in module_list])/10**6)
        print_info_by_type(module_list)
    return module_list


def plot_twins(x, y1, y2, labels=['flops', 'mac/flops'], xlabel='Layer Index', name='model'):
    fig = plt.figure()

    ax1 = fig.add_subplot(111)
    ax1.bar(x, y1, label=labels[0], width=0.3)
    ax1.set_ylim(min(y1)*0.9, max(y1)*1.2)
    ax1.set_ylabel(labels[0])
    ax1.set_title(labels[0] + '  Vs  ' + labels[1])

    ax2 = ax1.twinx()  # this is the important function
    ax2.bar([i+0.5 for i in x], y2, color='r', label=labels[1], width=0.3)
    ax2.set_ylim(min(y2)*0.9, max(y2)*1.2)
    ax2.set_ylabel(labels[1])
    ax2.set_xlabel(xlabel)
    plt.legend()
    plt.savefig(name+'_flops_vs_mac_div_flops')
    plt.clf()