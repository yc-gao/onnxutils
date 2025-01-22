import torch


class QuantizedConv:
    def to_float(self):
        new_mod = self.FLOAT_MODULE(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.bias is not None,
            self.padding_mode,
            self.weight.device,
            self.weight.dtype
        )

        new_mod.weight = self.weight
        if new_mod.bias is not None:
            new_mod.bias = self.bias

        return new_mod

    @classmethod
    def from_float(cls, float_module, fq_cls):
        new_mod = cls(
            fq_cls,
            float_module.in_channels,
            float_module.out_channels,
            float_module.kernel_size,
            float_module.stride,
            float_module.padding,
            float_module.dilation,
            float_module.groups,
            float_module.bias is not None,
            float_module.padding_mode,
            float_module.weight.device,
            float_module.weight.dtype
        )

        new_mod.weight = float_module.weight
        if new_mod.bias is not None:
            new_mod.bias = float_module.bias

        return new_mod


class QuantizedConv1d(torch.nn.Conv1d, QuantizedConv):
    FLOAT_MODULE = torch.nn.Conv1d

    def __init__(self,
                 fq_cls,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 device=None,
                 dtype=None):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype
        )
        self.weight_fake_quant = fq_cls()

    def forward(self, X):
        return self._conv_forward(
            X,
            self.weight_fake_quant(self.weight),
            self.bias)


class QuantizedConv2d(torch.nn.Conv2d, QuantizedConv):
    FLOAT_MODULE = torch.nn.Conv2d

    def __init__(self,
                 fq_cls,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 device=None,
                 dtype=None):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype
        )
        self.weight_fake_quant = fq_cls()

    def forward(self, X):
        return self._conv_forward(
            X,
            self.weight_fake_quant(self.weight),
            self.bias)


class QuantizedConv3d(torch.nn.Conv3d, QuantizedConv):
    FLOAT_MODULE = torch.nn.Conv3d

    def __init__(self,
                 fq_cls,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 device=None,
                 dtype=None):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype
        )
        self.weight_fake_quant = fq_cls()

    def forward(self, X):
        return self._conv_forward(
            X,
            self.weight_fake_quant(self.weight),
            self.bias)
