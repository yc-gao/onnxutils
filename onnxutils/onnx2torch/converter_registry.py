import onnx


_converter_registry = {}


def add_converter(
    op_type: str,
    version: int,
    domain: str = onnx.defs.ONNX_DOMAIN
):
    def deco(converter):
        op_key = (op_type, version, domain)
        if op_key in _converter_registry:
            raise RuntimeError(f"converter '{op_key}' already registered")
        _converter_registry[op_key] = converter
        return converter
    return deco


def find_converter(
    op_type: str,
    version: int,
    domain: str = onnx.defs.ONNX_DOMAIN,
):

    if schema := onnx.defs.get_schema(op_type, version, domain):
        version = schema.since_version

    op_key = (op_type, version, domain)
    return _converter_registry.get(op_key, None)
