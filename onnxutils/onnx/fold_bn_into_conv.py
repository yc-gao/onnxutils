from .onnx_model import OnnxModel
from .pass_manager import optimizer


@optimizer('fold-bn-into-conv')
class _:
    @staticmethod
    def apply(onnx_model: OnnxModel) -> OnnxModel:
        import onnxoptimizer
        return OnnxModel(
            onnxoptimizer.optimize(
                onnx_model.proto(),
                passes=['fuse_bn_into_conv']))

