from typing import Optional, List

import torch
import torch.nn.functional as F


class QuantizedConvTranspose:
    def to_float(self):
        new_mod = self.FLOAT_MODULE(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.output_padding,
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
            float_module.output_padding,
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


class QuantizedConvTranspose1d(torch.nn.ConvTranspose1d, QuantizedConvTranspose):
    FLOAT_MODULE = torch.nn.Conv1d

    def __init__(self,
                 fq_cls,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 output_padding=0,
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
            output_padding,
            groups,
            bias,
            padding_mode,
            device,
            dtype
        )
        self.weight_fake_quant = fq_cls()

    def forward(self, input: torch.Tensor, output_size: Optional[List[int]] = None) -> torch.Tensor:
        if self.padding_mode != "zeros":
            raise ValueError(
                "Only `zeros` padding mode is supported for ConvTranspose1d"
            )

        assert isinstance(self.padding, tuple)
        # One cannot replace List by Tuple or Sequence in "_output_padding" because
        # TorchScript does not support `Sequence[T]` or `Tuple[T, ...]`.
        num_spatial_dims = 1
        output_padding = self._output_padding(
            input,
            output_size,
            self.stride,  # type: ignore[arg-type]
            self.padding,  # type: ignore[arg-type]
            self.kernel_size,  # type: ignore[arg-type]
            num_spatial_dims,
            self.dilation,  # type: ignore[arg-type]
        )
        return F.conv_transpose1d(
            input,
            self.weight_fake_quant(self.weight),
            self.bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )


class QuantizedConvTranspose2d(torch.nn.ConvTranspose2d, QuantizedConvTranspose):
    FLOAT_MODULE = torch.nn.ConvTranspose2d

    def __init__(self,
                 fq_cls,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 output_padding=0,
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
            output_padding,
            groups,
            bias,
            padding_mode,
            device,
            dtype
        )
        self.weight_fake_quant = fq_cls()

    def forward(self, input: torch.Tensor, output_size: Optional[List[int]] = None) -> torch.Tensor:
        if self.padding_mode != "zeros":
            raise ValueError(
                "Only `zeros` padding mode is supported for ConvTranspose2d"
            )

        assert isinstance(self.padding, tuple)
        # One cannot replace List by Tuple or Sequence in "_output_padding" because
        # TorchScript does not support `Sequence[T]` or `Tuple[T, ...]`.
        num_spatial_dims = 2
        output_padding = self._output_padding(
            input,
            output_size,
            self.stride,  # type: ignore[arg-type]
            self.padding,  # type: ignore[arg-type]
            self.kernel_size,  # type: ignore[arg-type]
            num_spatial_dims,
            self.dilation,  # type: ignore[arg-type]
        )

        return F.conv_transpose2d(
            input,
            self.weight_fake_quant(self.weight),
            self.bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )


class QuantizedConvTranspose3d(torch.nn.ConvTranspose3d, QuantizedConvTranspose):
    FLOAT_MODULE = torch.nn.ConvTranspose3d

    def __init__(self,
                 fq_cls,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 output_padding=0,
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
            output_padding,
            groups,
            bias,
            padding_mode,
            device,
            dtype
        )
        self.weight_fake_quant = fq_cls()

    def forward(self, input: torch.Tensor, output_size: Optional[List[int]] = None) -> torch.Tensor:
        if self.padding_mode != "zeros":
            raise ValueError(
                "Only `zeros` padding mode is supported for ConvTranspose3d"
            )

        assert isinstance(self.padding, tuple)
        # One cannot replace List by Tuple or Sequence in "_output_padding" because
        # TorchScript does not support `Sequence[T]` or `Tuple[T, ...]`.
        num_spatial_dims = 3
        output_padding = self._output_padding(
            input,
            output_size,
            self.stride,  # type: ignore[arg-type]
            self.padding,  # type: ignore[arg-type]
            self.kernel_size,  # type: ignore[arg-type]
            num_spatial_dims,
            self.dilation,  # type: ignore[arg-type]
        )

        return F.conv_transpose3d(
            input,
            self.weight_fake_quant(self.weight),
            self.bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )
