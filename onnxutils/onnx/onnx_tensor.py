import numpy as np
import onnx
from onnx import TensorProto


class OnnxTensor:
    @staticmethod
    def from_numpy(tensor_np: np.ndarray, name: str):
        return OnnxTensor(onnx.numpy_helper.from_array(tensor_np, name))

    def __init__(self, tensor_pb: TensorProto) -> None:
        self._proto = tensor_pb

    @property
    def proto(self):
        return self._proto

    @property
    def name(self):
        return self.proto.name

    def to_numpy(self):
        return onnx.numpy_helper.to_array(self.proto)
