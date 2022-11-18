from AtomNet.models.mobile import atomnet_base

if __name__ == "__main__":
    config = {
        'depth': [1, 1, 2, 4, 4, 2],
        'out_channel': [24, 32, 96, 256, 320, 512],
        'block_type_stage': ['conv', 'mobile_one', 'mobile_one', 'mobile_one', 'mobile_one', 'mobile_one'],
        'act_stages': ['relu', 'hswish', 'hswish', 'hswish', 'hswish', 'hswish'],
        'se_stages': [False, False, False, False, True, True],
        'stride_stages': [2, 1, 2, 2, 2, 2],
        'kernel_size': [3, 3, 3, 5, 5, 5],
        'expand_ratio': [1, 1, 1, 1, 1, 1]
    }
    input_resolution = 224
    atomnet_512_max = atomnet_base(config)