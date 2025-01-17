__name2optimizer = {}


def optimizer(name):
    def wrapper(cls):
        if name in __name2optimizer:
            raise RuntimeError(f"optimizer '{name}' already registered")
        __name2optimizer[name] = cls
        return cls
    return wrapper


def find_optimizer(name):
    return __name2optimizer.get(name, None)


def apply_optimizers(onnx_model, optimizers):
    for name in optimizers:
        optimizer = find_optimizer(name)
        onnx_model = optimizer.apply(onnx_model)
    return onnx_model
