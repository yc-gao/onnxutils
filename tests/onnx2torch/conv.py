#!/usr/bin/env python3
import unittest

import os
import tempfile

import torch
from torchvision import models

import numpy as np
import onnx
import onnxruntime as ort

from onnxutils.onnx import OnnxModel
from onnxutils.onnx2torch import convert


class ConvTests(unittest.TestCase):
    def test_conv0(self):
        torch.set_printoptions(precision=8)
        torch_module = torch.nn.Conv2d(
            128,
            128,
            kernel_size=(1, 3),
            stride=(1, 1),
            padding=(0, 1),
            dilation=(1, 1),
            groups=1,
        )
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


class Resnet50Tests(unittest.TestCase):
    def test_resnet50(self):
        torch.set_printoptions(precision=8)

        torch_model = models.resnet50(models.ResNet50_Weights.IMAGENET1K_V2)
        torch_model.eval()

        with tempfile.TemporaryDirectory() as workdir:
            onnx_fpath = os.path.join(workdir, 'output.onnx')
            torch.onnx.export(
                torch_model,
                (torch.rand(1, 3, 224, 224),),
                onnx_fpath,
                input_names=['data'],
            )

            onnx.checker.check_model(onnx_fpath)
            onnx_model = OnnxModel.from_file(onnx_fpath)
            onnx_model = convert(onnx_model)

        for _ in range(100):
            data = torch.rand(1, 3, 224, 224)
            gt = torch_model(data)
            pred = onnx_model(data)

            self.assertTrue(torch.allclose(pred, gt, atol=1e-4))


if __name__ == '__main__':
    unittest.main()
