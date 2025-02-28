from ..onnx_model import OnnxModel
from ..pass_registry import add_optimizer


@add_optimizer('eliminate-transpose')
class _:
    @staticmethod
    def apply(onnx_model: OnnxModel) -> OnnxModel:
        import onnxoptimizer
        return OnnxModel(
            onnxoptimizer.optimize(
                onnx_model.proto,
                passes=['eliminate_nop_transpose']))
