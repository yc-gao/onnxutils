#!/usr/bin/env python3
import argparse
from pathlib import Path

import onnxruntime as ort
import numpy as np


dtype_mapping = {
    'tensor(float)': np.float32,
    'tensor(int32)': np.int32,
}


def load_data(inputs, imaps):
    def load(arg):
        return np.fromfile(
            imaps[arg.name],
            dtype=dtype_mapping[arg.type]).reshape(arg.shape)

    return {x.name: load(x) for x in inputs}


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='args', action='append', default=[])
    parser.add_argument('-o', '--output')
    parser.add_argument('model')
    return parser.parse_args()


def main():
    options = parse_options()
    imaps = {
        arg[0]: arg[1] for arg in map(lambda x: x.split(':'), options.args)
    }

    ort_sess = ort.InferenceSession(options.model)
    vals = ort_sess.run(None, load_data(ort_sess.get_inputs(), imaps))

    if options.output:
        output_dir = Path(options.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        for val, output in zip(vals, ort_sess.get_outputs()):
            val.tofile(output_dir/f"{output.name.replace('/', '_').replace('.', '_')}.bin")  # noqa


if __name__ == "__main__":
    main()
