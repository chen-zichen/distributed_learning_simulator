import torch
import torch.nn as nn
from cyy_naive_pytorch_lib.model_util import ModelUtil


class QuantModel(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear1 = nn.Linear(
            in_features=in_features, out_features=in_features, bias=True
        )

    def forward(self, x):
        x = self.linear1(x)
        x = torch.tanh(x)
        return torch.quantize_per_tensor(x, 0.1, 10, torch.quint8)


class QuantedModel(nn.Module):
    def __init__(self, parameter_model, quant_model):
        super().__init()
        self.parameter_model = parameter_model
        self.quant_model = quant_model
        self.parameter_names = list(
            sorted(ModelUtil(parameter_model).get_parameter_dict().keys())
        )

    def forward(self, x):
        quanted_parameters = self.quant_model(
            ModelUtil(self.parameter_model).get_parameter_list()
        ).int_repr()
        assert len(self.parameter_names) == quanted_parameters.shape[0]
        ModelUtil(self.parameter_model).load_parameter_dict(
            dict(zip(self.parameter_names, quanted_parameters))
        )
        return self.parameter_model(x)
