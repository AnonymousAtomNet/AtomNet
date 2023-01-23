from AtomNet.models.mobile import atomnet_base

if __name__ == "__main__":
    config = {
        'depth': [1, 1, 2, 2, 6, 2],
        'out_channel': [24, 24, 48, 64, 80, 128],
        'block_type_stage': ['conv', 'mobile_one', 'mobile_one', 'mbconvnext', 'mbconvnext', 'mbconvnext'],
        'act_stages': ['relu', 'hswish', 'hswish', 'hswish', 'hswish', 'hswish'],
        'se_stages': [False, False, False, True, True, True],
        'stride_stages': [2, 1, 2, 2, 2, 2],
        'kernel_size': [3, 3, 3, 3, 3, 3],
        'expand_ratio': [1, 1, 1, 6, 3, 6]
    }
    input_resolution = 160
    atomnet_256kb = atomnet_base(**config)
