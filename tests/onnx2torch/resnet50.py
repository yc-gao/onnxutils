#!/usr/bin/env python3
import unittest

import os
import tempfile

import torch
from torchvision import models

import onnx

from onnxutils.onnx import OnnxModel
from onnxutils.onnx2torch import convert


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
