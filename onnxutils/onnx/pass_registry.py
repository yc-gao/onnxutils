from .onnx_model import OnnxModel

_optimizer_registry = {}


def add_optimizer(name):
    def wrapper(cls):
        if name in _optimizer_registry:
            raise RuntimeError(f"optimizer '{name}' already registered")
        _optimizer_registry[name] = cls
        return cls
    return wrapper


def find_optimizer(name):
    return _optimizer_registry.get(name, None)


def list_optimizers():
    return tuple(_optimizer_registry.keys())


def apply_optimizers(onnx_model: OnnxModel, optimizers):
    for name in optimizers:
        onnx_model = find_optimizer(name).apply(onnx_model)
    return onnx_model
