from AtomNet.models.mobile import atomnet_base

if __name__ == "__main__":
    config = {
        'depth': [1, 1, 2, 2, 6, 2],
        'out_channel': [8, 16, 64, 96, 192, 320],
        'block_type_stage': ['conv', 'mobile_one', 'mobile_one', 'mobile_one', 'mobile_one', 'mobile_one'],
        'act_stages': ['relu', 'hswish', 'hswish', 'hswish', 'hswish', 'hswish'],
        'se_stages': [False, False, False, False, True, True],
        'stride_stages': [2, 1, 2, 2, 2, 2],
        'kernel_size': [3, 3, 3, 3, 3, 5],
        'expand_ratio': [1, 1, 1, 1, 1, 1]
    }
    input_resolution = 96
    atomnet_5fps = atomnet_base(**config)
