from ..onnx_model import OnnxModel

from ..pass_registry import add_optimizer


@add_optimizer('eliminate-identity')
class _:
    @staticmethod
    def apply(onnx_model: OnnxModel) -> OnnxModel:
        output_names = onnx_model.output_names
        with onnx_model.modify() as sess:
            for node in onnx_model.nodes:
                if node.op_type != 'Identity':
                    continue
                if node.output_names[0] in output_names:
                    continue
                sess.remap_node_inputs(
                    {node.output_names[0]: node.input_names[0]})
        return onnx_model
