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
<<<<<<< HEAD


class QuantedModel(torch.nn.Module):
    def __init__(self, parameter_model: torch.nn.Module):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.submodule = parameter_model
        # for name, module in parameter_model.named_modules():
        #     if not name:
        #         continue
        #     if "." in name:
        #         continue

        #     print("add name is ", name)
        #     self.add_module(name, module)
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.submodule(x)
        x = self.dequant(x)
        return x
        # for name, module in self.named_modules():
        #     if not name:
        #         continue
        #     if "." in name:
        #         continue
        #     print("name is ", name)
        #     print(x.shape)
        #     x = module(x)
        # return x
=======
>>>>>>> main
