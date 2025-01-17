_optimizer_registry = {}


def optimizer(name):
    def wrapper(cls):
        if name in _optimizer_registry:
            raise RuntimeError(f"optimizer '{name}' already registered")
        _optimizer_registry[name] = cls
        return cls
    return wrapper


def find_optimizer(name):
    return _optimizer_registry.get(name, None)


def apply_optimizers(onnx_model, optimizers):
    for name in optimizers:
        optimizer = find_optimizer(name)
        onnx_model = optimizer.apply(onnx_model)
    return onnx_model


def list_optimizers():
    return _optimizer_registry.keys()
