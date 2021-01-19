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
    def __init__(self, parameter_model, quant_model):
        super().__init__()
        self.parameter_model = parameter_model
        # self.quant_model = quant_model

    def forward(self, x):
        # quanted_parameters = self.quant_model(
        #     ModelUtil(self.parameter_model).get_parameter_list()
        # )
        # ModelUtil(self.parameter_model).load_parameter_list(quanted_parameters)
        return self.parameter_model(x)
