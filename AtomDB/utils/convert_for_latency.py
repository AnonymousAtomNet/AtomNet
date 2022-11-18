import onnx
import torch
import argparse

import numpy as np
import tensorflow as tf

from onnx2keras import onnx_to_keras


def generate_tflite_with_weight(
    onnx_model_path, tflite_fname, calib_loader, input_name, n_calibrate_sample=100
):
    # 1. convert onnx format to keras model, to get nhwc model by change_ordering
    onnx_model = onnx.load(onnx_model_path)
    k_model = onnx_to_keras(onnx_model, input_name, change_ordering=True)

    # 2. convert to tflite (with int8 quantization)
    converter = tf.lite.TFLiteConverter.from_keras_model(k_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.inference_output_type = tf.int8
    converter.inference_input_type = tf.int8

    def representative_dataset_gen():
        for i_b, (data, _) in enumerate(calib_loader):
            if i_b == n_calibrate_sample:
                break
            print(data.shape)
            data = data.numpy().transpose(0, 2, 3, 1).astype(np.float32)
            yield [data]

    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    tflite_buffer = converter.convert()
    with open(tflite_fname, "wb") as f:
        f.write(tflite_buffer)


parser = argparse.ArgumentParser(description="STM32CUBEMX tflite transer shell")
parser.add_argument(
    "-d",
    "--dir",
    help="data dir",
    type=str,
    default="/path-of-imagenet/",
)
parser.add_argument(
    "-o", "--onnx", help="onnx path", type=str, default="./test_log/test.onnx"
)
parser.add_argument(
    "-t",
    "--tflite",
    help="output tflite path",
    type=str,
    default="./tflites/test.tflite",
)
parser.add_argument(
    "-n", "--n_calibrate_sample", help="n_calibrate_sample", type=int, default=100
)

args = parser.parse_args()

if __name__ == "__main__":
    onnx_path = args.onnx

    # get onnx description
    model = onnx.load(onnx_path)
    #! assume just one input in here
    inputs = model.graph.input
    input_name = inputs[0].name
    try:
        input_c = inputs[0].type.tensor_type.shape.dim[1].dim_value
        input_hr = inputs[0].type.tensor_type.shape.dim[2].dim_value
        input_wr = inputs[0].type.tensor_type.shape.dim[3].dim_value
        input_size = (input_c, input_hr, input_wr)
    except:
        input_features = inputs[0].type.tensor_type.shape.dim[1].dim_value
        input_size = (input_features)

    dataset_path = args.dir
    # process export tflite path
    if args.tflite == "./tflites/test.tflite":
        # export right side with origin onnx
        tflite_path = args.onnx[:-5] + ".tflite"
    else:
        tflite_path = args.tflite

    # prepare calib loader
    # calibrate the model for quantization
    from torchvision import datasets, transforms
    class ReshapeTransform:
        def __init__(self, new_size):
            self.new_size = new_size

        def __call__(self, img):
            return torch.rand(self.new_size)

    train_dataset = datasets.ImageFolder(
        dataset_path,
        transform=transforms.Compose(
            [
                ReshapeTransform(input_size)
            ]
        ),
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=4
    )

    generate_tflite_with_weight(
        onnx_path,
        tflite_path,
        train_loader,
        [input_name],
        n_calibrate_sample=args.n_calibrate_sample,
    )
