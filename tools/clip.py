#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path

from onnxutils.onnx import OnnxModel


def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('--inode', dest='inodes', type=str,
                        default=[], action='append')
    parser.add_argument('--onode', dest='onodes', type=str,
                        default=[], action='append')
    parser.add_argument('model', type=str)
    return parser.parse_args(args)


def main():
    options = parse_args()

    onnx_model = OnnxModel.from_file(options.model)
    onnx_model = onnx_model.extract(
        options.inodes, options.onodes or onnx_model.output_names())

    output = Path(options.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx_model.save(output)


if __name__ == '__main__':
    main()
