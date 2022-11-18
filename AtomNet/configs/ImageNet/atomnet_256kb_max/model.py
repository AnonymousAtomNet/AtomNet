from AtomNet.models.mobile import atomnet_base

if __name__ == "__main__":
    config = {
        'depth': [1, 1, 2, 2, 6, 2],
        'out_channel': [16, 16, 32, 64, 80, 144],
        'block_type_stage': ['conv', 'mobile_one', 'mobile_one', 'mbconvnext', 'mbconvnext', 'mbconvnext'],
        'act_stages': ['relu', 'hswish', 'hswish', 'hswish', 'hswish', 'hswish'],
        'se_stages': [False, False, False, True, True, True],
        'stride_stages': [2, 1, 2, 2, 2, 2],
        'kernel_size': [3, 3, 3, 5, 5, 5],
        'expand_ratio': [1, 1, 1, 3, 3, 6]
    }
    input_resolution = 224
    atomnet_256kb_max = atomnet_base(config)
