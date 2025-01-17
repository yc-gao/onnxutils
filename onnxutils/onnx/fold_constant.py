import onnx

from .onnx_model import OnnxModel

from .pass_manager import optimizer


def eval_const_vals(onnx_model: OnnxModel, const_vals):
    onnx_model = onnx_model.extract([], const_vals)
    import onnxruntime as ort
    sess = ort.InferenceSession(
        onnx_model.proto().SerializeToString(),
        providers=[
            x for x in ['CUDAExecutionProvider', 'CPUExecutionProvider']
            if x in ort.get_available_providers()
        ])
    outputs = sess.run(None, {})
    return [
        onnx.numpy_helper.from_array(val, output.name)
        for (val, output) in zip(outputs, sess.get_outputs())
    ]


@optimizer('fold-constant')
class _:

    @staticmethod
    def apply(onnx_model: OnnxModel) -> OnnxModel:
        output_names = set(onnx_model.output_names())
        with onnx_model.session() as sess:
            mutable_tensors = onnx_model.input_names()
            mutable_nodes = set()
            for node in onnx_model.nodes():
                if any(x in mutable_tensors for x in node.inputs()):
                    mutable_tensors.update(node.outputs())
                    mutable_nodes.add(node.name())

            nodes_to_fold = [
                x for x in onnx_model.nodes()
                if x.name() not in mutable_nodes and all(
                    output_name not in output_names
                    for output_name in x.outputs())]

            if nodes_to_fold:
                const_vals = [
                    output
                    for node in nodes_to_fold
                    for output in node.outputs()]
                const_vals = eval_const_vals(
                    onnx_model, const_vals)
                sess.add_initializers(const_vals)
                sess.remove_nodes(nodes_to_fold)

        return onnx_model
