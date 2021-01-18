import torch
import torch.nn as nn
from cyy_naive_pytorch_lib.model_util import ModelUtil


class QuantModel(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.hidden_features = 100
        self.linear1 = nn.Linear(
            in_features=in_features, out_features=self.hidden_features, bias=True
        )
        self.linear2 = nn.Linear(
            in_features=self.hidden_features, out_features=in_features, bias=True
        )

    def forward(self, x):
        x = self.linear1(x)
        x = torch.tanh(x)
        x = self.linear2(x)
        x = torch.tanh(x)
        return torch.quantize_per_tensor(x, 0.1, 10, torch.quint8)


class local_grad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        return i

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clamp_(-1, 1)


class QuantedModel(nn.Module):
    def __init__(self, parameter_model, quant_model):
        super().__init__()
        self.parameter_model = parameter_model
        self.quant_model = quant_model
        self.parameter_names = sorted(
            ModelUtil(parameter_model).get_parameter_dict().keys()
        )
        print(self.parameter_names)

    def forward(self, x):
        quanted_parameters = self.quant_model(
            ModelUtil(self.parameter_model).get_parameter_list()
        ).int_repr()

        print(quanted_parameters.shape)
        quanted_parameters = local_grad.apply(quanted_parameters)
        print(quanted_parameters.shape)

        ModelUtil(self.parameter_model).load_parameter_list(quanted_parameters)
        return self.parameter_model(x)
