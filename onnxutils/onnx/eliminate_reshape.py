from .onnx_model import OnnxModel
from .pass_manager import optimizer


@optimizer('eliminate-reshape')
class _:
    @staticmethod
    def apply(onnx_model: OnnxModel) -> OnnxModel:
        import onnxoptimizer
        return OnnxModel(
            onnxoptimizer.optimize(
                onnx_model.proto(),
                passes=['eliminate_nop_reshape']))
