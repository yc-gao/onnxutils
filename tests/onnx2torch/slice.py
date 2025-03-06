#!/usr/bin/env python3
import unittest

import os
import tempfile

import torch

import numpy as np
import onnx
import onnxruntime as ort

from onnxutils.onnx import OnnxModel, apply_optimizers
from onnxutils.onnx2torch import convert


class SliceTests(unittest.TestCase):
    def test_slice0(self):
        torch.set_printoptions(precision=8)

        class SliceModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                return x[2:5]

        torch_module = SliceModule()
        with tempfile.TemporaryDirectory() as workdir:
            onnx_fpath = os.path.join(workdir, 'output.onnx')
            torch.onnx.export(
                torch_module,
                (torch.rand(10, 128, 72, 120),),
                onnx_fpath,
                input_names=['x'],
            )
            del torch_module

            onnx.checker.check_model(onnx_fpath)
            onnx_model = OnnxModel.from_file(onnx_fpath)
            onnx_model = apply_optimizers(
                onnx_model, ['convert-constant-to-initializer'])
            onnx_module = convert(onnx_model)
            sess = ort.InferenceSession(
                onnx_fpath,
                providers=['CPUExecutionProvider'])
        for _ in range(100):
            x = np.random.rand(10, 128, 72, 120).astype(np.float32)
            y, = sess.run(None, {'x': x})
            pred = onnx_module(torch.from_numpy(x))
            self.assertTrue(
                np.allclose(
                    y,
                    pred.detach().cpu().numpy(),
                    1e-5,
                    1e-5
                )
            )


if __name__ == '__main__':
    unittest.main()
