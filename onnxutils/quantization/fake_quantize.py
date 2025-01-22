from abc import ABC, abstractmethod
import functools

import torch


class FakeQuantizeBase(ABC, torch.nn.Module):
    per_channel_qscheme = (torch.per_tensor_symmetric, torch.per_tensor_affine)
    per_tensor_qscheme = (torch.per_channel_symmetric,
                          torch.per_channel_affine)
    symmetric_qscheme = (torch.per_tensor_symmetric,
                         torch.per_channel_symmetric)

    supported_qscheme = (torch.per_channel_symmetric, torch.per_channel_affine,
                         torch.per_tensor_symmetric, torch.per_tensor_affine)

    @classmethod
    def with_args(cls, *args, **kwargs):
        return functools.partial(cls, *args, **kwargs)

    def __init__(self, observer_enabled=True, fake_quant_enabled=True) -> None:
        super().__init__()
        self.observer_enabled = observer_enabled
        self.fake_quant_enabled = fake_quant_enabled

    def enable_observer_enabled(self):
        self.observer_enabled = True

    def disable_observer_enabled(self):
        self.observer_enabled = False

    def enable_fake_quant(self):
        self.fake_quant_enabled = True

    def disable_fake_quant(self):
        self.fake_quant_enabled = False

    @abstractmethod
    def calculate_qparams(self):
        pass

    @abstractmethod
    def forward(self, X):
        pass


class FakeQuantize(FakeQuantizeBase):
    def __init__(self, observer) -> None:
        super().__init__()

        self.activation_post_process = observer()
        self.quant_min = self.activation_post_process.quant_min
        self.quant_max = self.activation_post_process.quant_max
        self.dtype = self.activation_post_process.dtype
        self.qscheme = self.activation_post_process.qscheme
        self.ch_axis = (
            self.activation_post_process.ch_axis
            if hasattr(self.activation_post_process, "ch_axis")
            else -1
        )
        self.is_per_channel = self.qscheme in self.per_channel_qscheme

        assert self.qscheme in self.supported_qscheme

        self.register_buffer("scale", torch.tensor([1.0], dtype=torch.float))
        self.register_buffer("zero_point", torch.tensor([0], dtype=torch.int))

    def calculate_qparams(self):
        return self.activation_post_process.calculate_qparams()

    def forward(self, X):
        if self.observer_enabled:
            self.activation_post_process(X.detach())
            _scale, _zero_point = self.calculate_qparams()
            _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(
                self.zero_point.device
            )
            if self.scale.shape != _scale.shape:
                self.scale.resize_(_scale.shape)
                self.zero_point.resize_(_zero_point.shape)
            self.scale.copy_(_scale)
            self.zero_point.copy_(_zero_point)

        if self.fake_quant_enabled:
            if self.is_per_channel:
                X = torch.fake_quantize_per_channel_affine(
                    X,
                    self.scale,
                    self.zero_point,
                    self.ch_axis,
                    self.activation_post_process.quant_min,
                    self.activation_post_process.quant_max,
                )
            else:
                X = torch.fake_quantize_per_tensor_affine(
                    X,
                    self.scale,
                    self.zero_point,
                    self.activation_post_process.quant_min,
                    self.activation_post_process.quant_max,
                )
        return X


class FixedFakeQuantize(FakeQuantize):
    def __init__(self, observer) -> None:
        super().__init__(observer)

        self.scale = self.activation_post_process.scale
        self.zero_point = self.activation_post_process.zero_point

    def calculate_qparams(self):
        return self.scale, self.zero_point
