#!/usr/bin/env python3
import argparse

from onnxutils.onnx import OnnxModel
from onnxutils.onnx2torch import convert


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    return parser.parse_args()


def main():
    options = parse_options()

    onnx_model = OnnxModel.from_file(options.model)
    with onnx_model.session() as sess:
        for node in onnx_model.proto().graph.node:
            if node.name == '':
                node.name = sess.unique_name()

    torch_model = convert(onnx_model, False)
    print(torch_model)


if __name__ == "__main__":
    main()
