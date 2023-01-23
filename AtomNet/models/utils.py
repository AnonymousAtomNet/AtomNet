import os
import torch
from torch.nn import Conv2d, Linear # noqa
from decimal import Decimal


def export_onnx_by_module(module_table):
    for name, size in module_table.items():
        onnx_name = name.replace(' ', '').replace('(', '_').replace(',', '')+'_r_'+str(size[-1])+'.onnx'
        if not os.path.exists(onnx_name):
            name = name.replace('bias=False', 'bias=True')
            module = decode_extra_repr(name)
            if query(module.__repr__(), size) == 0:
                torch.onnx.export(module, torch.randn(*size), onnx_name)


def decode_extra_repr(repr='Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))'):
    module = eval(repr)
    return module


def merge_block(meta_dict, model):
    new_dict = {}
    for k, v in meta_dict.items():
        if 'blocks' in k:
            block_number = int(k.split('.')[1])
            if 'blocks.' + str(block_number) in new_dict.keys():
                continue
            if getattr(model.blocks[block_number], 'middle_channel', 0):
                max_channel = max([model.blocks[block_number].in_channel, model.blocks[block_number].out_channel,
                                   model.blocks[block_number].middle_channel])
            else:
                max_channel = max([model.blocks[block_number].in_channel, model.blocks[block_number].out_channel])
            new_dict['blocks' + str(block_number)] = {'type': type(model.blocks[block_number]),
                           'flops': sum([v['flops'] for k, v in meta_dict.items() if 'blocks.' + str(block_number) + '.' in k]),
                           'params': sum([v['params'] for k, v in meta_dict.items() if 'blocks.' + str(block_number) + '.' in k]),
                           'max_activation': max([v['activation'] for k, v in meta_dict.items() if 'blocks.' + str(block_number) + '.' in k]),
                           'max_channel': max_channel,
                           'block_number': block_number
            }
        else:
            module = dict(model.named_modules())[k]
            new_dict[k] = {'type': type(module),
                           'flops': v['flops'],
                           'params': v['params'],
                           'max_activation': v['activation'],
                           'max_channel': module.out_channels if getattr(module, 'out_channels', 0) else module.out_features,
                           }
    return new_dict


def merge_stage(meta_dict, model):
    stage_out_idxs = model.stage_out_idx
    stride_stages = model.stride_stages
    new_stage_out_idxs = []
    for i in range(len(stride_stages)-1):
        if stride_stages[i+1] == 2:
            new_stage_out_idxs.append(stage_out_idxs[i])
    new_stage_out_idxs.append(stage_out_idxs[-1])

    block_numbers = []
    for i, d in enumerate(new_stage_out_idxs):
        if i == 0:
            block_numbers.append([i for i in range(d+1)])
        else:
            block_numbers.append([i for i in range(new_stage_out_idxs[i-1]+1, d+1)])

    new_dict = {}

    stage_all = sum([1 for i in stride_stages if i == 2])
    for stage_idx in range(stage_all):
        new_dict['stage.'+str(stage_idx)] = {
            'flops': sum([v['flops'] for k, v in meta_dict.items() if 'block_number' in v and v['block_number'] in block_numbers[stage_idx]]),
            'params': sum([v['params'] for k, v in meta_dict.items() if 'block_number' in v and v['block_number'] in block_numbers[stage_idx]]),
            'max_activation': max([v['max_activation'] for k, v in meta_dict.items() if 'block_number' in v and v['block_number'] in block_numbers[stage_idx]]),
            'max_channel': max([v['max_channel'] for k, v in meta_dict.items() if 'block_number' in v and v['block_number'] in block_numbers[stage_idx]]),
            'block_number': len(block_numbers[stage_idx]),
        }

    for k, v in meta_dict.items():
        if 'blocks' not in k:
            new_dict[k] = v

    all_flops = sum([v['flops'] for v in new_dict.values()])
    all_params = sum([v['params'] for v in new_dict.values()])
    print('Flops:', Decimal(all_flops).quantize(Decimal('0.00')))
    print('Params:', Decimal(all_params).quantize(Decimal('0.00')))

    for _, v in new_dict.items():
        print('--------------{}-----------'.format(_))
        for i, j in v.items():
            if i == 'flops':
                new_j = str(Decimal(j / all_flops * 100).quantize(Decimal('0.000'))) + '%'
            elif i == 'params':
                new_j = str(Decimal(j / all_params * 100).quantize(Decimal('0.000'))) + '%'
            else:
                new_j = j
            print(i, ':', new_j)

    return new_dict