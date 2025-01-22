import torch
import torch.nn.functional as F


class QuantizedLinear(torch.nn.Linear):
    FLOAT_MODULE = torch.nn.Linear

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 device=None,
                 dtype=None,
                 fq_cls=None):
        super().__init__(
            in_features,
            out_features,
            bias,
            device,
            dtype
        )
        self.weight_fake_quant = fq_cls()

    def forward(self, X):
        return F.linear(X, self.weight_fake_quant(self.weight), self.bias)

    @staticmethod
    def from_float(float_module, fq_cls):
        new_mod = QuantizedLinear(
            float_module.in_features,
            float_module.out_features,
            float_module.bias is not None,
            float_module.weight.device,
            float_module.weight.dtype,
            fq_cls
        )

        new_mod.weight = float_module.weight
        if new_mod.bias is not None:
            new_mod.bias = float_module.bias

        return new_mod
