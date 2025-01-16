#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path

import onnx


def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('--iomap', dest='iomaps', type=str,
                        default=[], action='append')
    parser.add_argument('model0', type=str)
    parser.add_argument('model1', type=str)
    return parser.parse_args(args)


def main():
    options = parse_args()

    model0 = onnx.load(options.model0)
    model1 = onnx.load(options.model1)

    model = onnx.compose.merge_models(model0, model1, options.iomaps)

    output = Path(options.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output)


if __name__ == '__main__':
    main()
