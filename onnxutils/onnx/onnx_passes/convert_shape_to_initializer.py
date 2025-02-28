import onnx

from ..onnx_model import OnnxModel
from ..pass_registry import add_optimizer


@add_optimizer('convert-shape-to-initializer')
class _:
    @staticmethod
    def apply(onnx_model: OnnxModel) -> OnnxModel:
        output_names = onnx_model.output_names
        with onnx_model.modify() as sess:
            for node in onnx_model.nodes:
                if node.op_type != 'Shape':
                    continue
                if node.output_names[0] in output_names:
                    continue

                vinfo = onnx_model.get_info_by_name(node.input_names[0])
                if vinfo is None:
                    continue

                shape = [
                    x.dim_value if x.HasField('dim_value') else -1
                    for x in vinfo.type.tensor_type.shape.dim
                ]
                if any(x == -1 for x in shape):
                    continue

                sess.remove_node(node)
                sess.add_initializer(
                    onnx.helper.make_tensor(
                        node.output_names[0],
                        onnx.TensorProto.INT64,
                        [len(shape)],
                        shape,
                    )
                )

        return onnx_model
