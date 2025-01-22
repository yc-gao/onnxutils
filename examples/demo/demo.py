#!/usr/bin/env python3
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization.observer import PerChannelMinMaxObserver

from onnxutils.quantization.modules import QuantizedLinear
from onnxutils.quantization.fake_quantize import FakeQuantize


class M(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 3)

    def forward(self, x):
        x = self.linear(x)
        return x


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, default='output.onnx')
    return parser.parse_args()


def main():
    options = parse_options()
    torch_input = (torch.randn(1, 3), )

    fq_cls = FakeQuantize.with_args(
        observer=PerChannelMinMaxObserver)
    m = M()
    m.linear = QuantizedLinear.from_float(m.linear, fq_cls)
    m(*torch_input)

    for module in m.modules():
        if isinstance(module, FakeQuantize):
            module.disable_observer()

    torch.onnx.export(
        m,
        torch_input,
        options.output,
        input_names=["x"],
        output_names=["y"]
    )


if __name__ == "__main__":
    main()
