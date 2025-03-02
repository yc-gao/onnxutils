from typing import Any, Union

from .onnx_model import OnnxModel

_optimizer_registry = {}


def add_optimizer(name: str):
    def wrapper(cls):
        if name in _optimizer_registry:
            raise RuntimeError(f"optimizer '{name}' already registered")
        _optimizer_registry[name] = cls
        return cls
    return wrapper


def find_optimizer(name: str):
    return _optimizer_registry.get(name, None)


def list_optimizers() -> tuple[str, ...]:
    return tuple(_optimizer_registry.keys())


def apply_optimizers(onnx_model: OnnxModel, optimizers: list[Union[str, Any]]) -> OnnxModel:
    for optim in optimizers:
        if isinstance(optim, str):
            optim = find_optimizer(optim)
        onnx_model = optim.apply(onnx_model)
    return onnx_model
