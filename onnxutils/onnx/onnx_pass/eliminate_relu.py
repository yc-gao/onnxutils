from ..onnx_model import OnnxModel
from ..pass_registry import add_optimizer

from .dag_matcher import DagMatcher

dag_pattern = DagMatcher({
    'id': 0,
    'op_type': 'DequantizeLinear',
    'inputs': [
        {
            'id': 1,
            'op_type': 'QuantizeLinear',
            'inputs': [
                {
                    'id': 2,
                    'op_type': 'Relu'
                }
            ]
        },
    ]
})


@add_optimizer('eliminate-relu')
class _:
    @staticmethod
    def apply(onnx_model: OnnxModel) -> OnnxModel:
        with onnx_model.session() as sess:
            for dag in dag_pattern.MatchAllDags(onnx_model):
                relu_node = dag_pattern.GetNode(dag, 2)

                sess.remap_node_inputs({
                    relu_node.outputs()[0]: relu_node.inputs()[0]
                })
        return onnx_model
