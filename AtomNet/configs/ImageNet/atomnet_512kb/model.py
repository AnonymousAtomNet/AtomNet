from AtomNet.models.mobile import atomnet_base

if __name__ == "__main__":
    config = {
        'depth': [1, 1, 2, 2, 6, 2],
        'out_channel': [8, 16, 48, 96, 112, 192],
        'block_type_stage': ['conv', 'mbconv', 'mbconv', 'mbconv', 'mbconv', 'mbconv'],
        'act_stages': ['relu', 'hswish', 'hswish', 'hswish', 'hswish', 'hswish'],
        'se_stages': [False, False, False, False, True, True],
        'stride_stages': [2, 1, 2, 2, 2, 2],
        'kernel_size': [3, 3, 3, 3, 3, 3],
        'expand_ratio': [1, 1, 3, 3, 3, 6]
    }
    input_resolution = 160
    atomnet_512kb = atomnet_base(**config)
