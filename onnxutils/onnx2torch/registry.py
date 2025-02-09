import warnings

from onnx import defs


_converter_registry = {}


def converter(
    operation_type: str,
    version: int,
    domain: str = defs.ONNX_DOMAIN,
    force=False,
):
    def deco(converter):
        op_key = (domain, operation_type, version)
        if op_key in _converter_registry:
            if force:
                warnings.warn(
                    f"Operation '{op_key}' already registered, overwrite")
            else:
                raise ValueError(
                    f"Operation '{op_key}' already registered")
        _converter_registry[op_key] = converter
        return converter
    return deco


def find_converter(
    operation_type: str,
    version: int,
    domain: str = defs.ONNX_DOMAIN,
):

    try:
        version = defs.get_schema(
            operation_type,
            domain=domain,
            max_inclusive_version=version,
        ).since_version
    except (RuntimeError, defs.SchemaError):
        pass

    op_key = (domain, operation_type, version)
    converter = _converter_registry.get(op_key, None)
    if converter is None:
        raise NotImplementedError(
            f'Converter is not implemented {op_key}')

    return converter
