import torch


class QuantedModel(torch.nn.Module):
    def __init__(self, parameter_model: torch.nn.Module):
        super().__init__()
        self.register_buffer("scale", torch.Tensor([1.0]))
        self.register_buffer("zero_point", torch.Tensor([0]))
        self.quant = torch.quantization.QuantStub()
        self.module = parameter_model
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = x / self.scale
        x = x + self.zero_point
        x = self.quant(x)
        x = self.module(x)
        x = self.dequant(x)
        return x
