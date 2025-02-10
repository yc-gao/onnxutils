from .onnx_model import OnnxModel

from .pass_manager import optimizer
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


@optimizer('eliminate-qdq')
class _:
    @staticmethod
    def apply(onnx_model: OnnxModel) -> OnnxModel:
        output_names = set(onnx_model.output_names())
        with onnx_model.session() as sess:
            for dag in dag_pattern.MatchAllDags(onnx_model):
                dq_node = dag_pattern.GetNode(dag, 0)
                q_node = dag_pattern.GetNode(dag, 1)

                if dq_node.outputs()[0] in output_names:
                    raise NotImplementedError
                else:
                    sess.remap_node_inputs({
                        dq_node.outputs()[0]: q_node.inputs()[0]
                    })

        return onnx_model
