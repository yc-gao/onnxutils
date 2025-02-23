import onnx

from ..onnx_model import OnnxModel
from ..pass_registry import add_optimizer

from .dag_matcher import DagMatcher

dag_pattern = DagMatcher({
    'id': 0,
    'op_type': 'DequantizeLinear',
    'inputs': [
        {
            'id': 1,
            'op_type': 'QuantizeLinear'
        },
    ]
})


@add_optimizer('convert-qdq-to-fq')
class _:
    @staticmethod
    def apply(onnx_model: OnnxModel) -> OnnxModel:
        with onnx_model.session() as sess:
            for dag in dag_pattern.MatchAllDags(onnx_model):
                dq_node = dag_pattern.GetNode(dag, 0)
                q_node = dag_pattern.GetNode(dag, 1)
                sess.remove_nodes([dq_node, q_node])

                qdq_node = onnx.helper.make_node(
                    'FakeQdq',
                    q_node.inputs(),
                    dq_node.outputs(),
                    sess.unique_name(),
                )

                sess.add_node(qdq_node)
        return onnx_model
