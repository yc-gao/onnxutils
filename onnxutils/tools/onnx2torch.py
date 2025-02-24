#!/usr/bin/env python3
import argparse

import numpy as np
import onnx
import torch

from onnxutils.common.random import rand_numpy
from onnxutils.onnx import OnnxModel, apply_optimizers
from onnxutils.onnx2torch import convert


def tensor_from_vinfo(vinfo):
    dtype_mapping = {
        onnx.TensorProto.DataType.FLOAT16: np.float16,
        onnx.TensorProto.DataType.FLOAT: np.float32,
        onnx.TensorProto.DataType.DOUBLE: np.float64,
    }
    shape = tuple(x.dim_value
                  for x in vinfo.type.tensor_type.shape.dim)
    dtype = vinfo.type.tensor_type.elem_type
    return torch.from_numpy(rand_numpy(shape, dtype_mapping[dtype]))


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output')
    parser.add_argument('model')
    return parser.parse_args()


def main():
    options = parse_options()

    onnx_model = OnnxModel.from_file(options.model)
    with onnx_model.session() as sess:
        for node in onnx_model.proto().graph.node:
            if node.name == '':
                node.name = sess.unique_name()

    onnx_model = apply_optimizers(
        onnx_model, [
            'eliminate-identity',
            'convert-constant-to-initializer',
            'fold-constant',

            'convert-shape-to-initializer',
            'eliminate-expand',
            'fold-constant',
        ])

    torch_model = convert(onnx_model)
    torch_model.print_readable()

    example_inputs = tuple(tensor_from_vinfo(x) for x in onnx_model.inputs())

    if options.output:
        torch.onnx.export(
            torch_model,
            example_inputs,
            options.output,
            input_names=torch_model.onnx_mapping['inputs'],
            output_names=torch_model.onnx_mapping['outputs'],
        )


if __name__ == "__main__":
    main()
