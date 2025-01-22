from .onnx_model import OnnxModel
from .pass_manager import optimizer


@optimizer('convert-constant-to-initializer')
class _:
    @staticmethod
    def apply(onnx_model: OnnxModel) -> OnnxModel:
        import onnxoptimizer
        return OnnxModel(
            onnxoptimizer.optimize(
                onnx_model.proto(),
                passes=['extract_constant_to_initializer']))

