import torch
from cyy_naive_pytorch_lib.model_util import ModelUtil


class QuantModel(torch.nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.hidden_features = 100
        self.linear1 = torch.nn.Linear(
            in_features=in_features, out_features=self.hidden_features, bias=True
        )
        self.relu1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(
            in_features=self.hidden_features, out_features=in_features, bias=True
        )
        self.relu2 = torch.nn.ReLU()
        self.quant = torch.quantization.QuantStub()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.quant(x)
        return x


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
