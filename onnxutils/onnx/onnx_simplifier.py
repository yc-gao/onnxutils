from .onnx_model import OnnxModel

from .pass_manager import optimizer


@optimizer('onnx-simplifier')
class _:
    @staticmethod
    def apply(onnx_model: OnnxModel) -> OnnxModel:
        from onnxsim.onnx_simplifier import simplify
        model, ret = simplify(onnx_model.proto())
        if ret:
            return OnnxModel(model)
        return onnx_model
