#!/usr/bin/env python3
import argparse
from pathlib import Path

from onnxutils.onnx import OnnxModel
from onnxutils.onnx import apply_optimizers, list_optimizers


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output')
    parser.add_argument('-l', '--ls-optim', action='store_true', default=False)
    parser.add_argument('--optim', action='append', default=[])
    parser.add_argument('model', nargs='?')
    return parser.parse_args()


def main():
    options = parse_options()

    if options.ls_optim:
        print('\n'.join(list_optimizers()))

    if options.model:
        onnx_model = OnnxModel.from_file(options.model)
        onnx_model = apply_optimizers(onnx_model, options.optim)
        onnx_model.topological_sort()

        if options.output:
            output = Path(options.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            onnx_model.save(output)


if __name__ == "__main__":
    main()
