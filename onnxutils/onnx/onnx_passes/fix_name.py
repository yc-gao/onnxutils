from ..onnx_model import OnnxModel
from ..pass_registry import add_optimizer


@add_optimizer('fix-name')
class _:
    @staticmethod
    def apply(onnx_model: OnnxModel) -> OnnxModel:
        with onnx_model.modify() as sess:
            for node in onnx_model.proto.graph.node:
                if node.name == '':
                    node.name = sess.unique_name()
        return onnx_model
