#!/usr/bin/env python3
import argparse

from onnxutils.onnx import OnnxModel


def dfscmp(onnx_model0, onnx_model1, output_name0, output_name1):
    onnx_node0 = onnx_model0.get_node_by_output(output_name0)
    onnx_node1 = onnx_model1.get_node_by_output(output_name1)

    if onnx_node0 is None and onnx_node1 is None:
        return

    if onnx_node0 is None or onnx_node1 is None:
        print('node diff')
        print(f"    {onnx_node0.name() if onnx_node0 else onnx_node0}")
        print(f"    {onnx_node1.name() if onnx_node1 else onnx_node1}")
        return

    if onnx_node0.op_type() != onnx_node1.op_type():
        print('node diff')
        print(f"    {onnx_node0.name() if onnx_node0 else onnx_node0}")
        print(f"    {onnx_node1.name() if onnx_node1 else onnx_node1}")
        return

    if len(onnx_node0.inputs()) != len(onnx_node1.inputs()):
        print('node diff')
        print(f"    {onnx_node0.name() if onnx_node0 else onnx_node0}")
        print(f"    {onnx_node1.name() if onnx_node1 else onnx_node1}")
        return

    for input_name0, input_name1 in zip(onnx_node0.inputs(), onnx_node1.inputs()):  # noqa
        dfscmp(onnx_model0, onnx_model1, input_name0, input_name1)


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('model0')
    parser.add_argument('model1')
    return parser.parse_args()


def main():
    options = parse_options()
    onnx_model0 = OnnxModel.from_file(options.model0)
    onnx_model1 = OnnxModel.from_file(options.model1)

    if len(onnx_model0.output_names()) != len(onnx_model1.output_names()):  # noqa
        print('output diff')
        print(f'    {onnx_model0.output_names()}')
        print(f'    {onnx_model1.output_names()}')

    for output_name0, output_name1 in zip(onnx_model0.output_names(), onnx_model1.output_names()):  # noqa
        dfscmp(onnx_model0, onnx_model1, output_name0, output_name1)


if __name__ == "__main__":
    main()
