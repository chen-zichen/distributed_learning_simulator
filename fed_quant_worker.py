import copy

import torch
from cyy_naive_lib.log import get_logger
from cyy_naive_pytorch_lib.device import get_cpu_device
# from cyy_naive_pytorch_lib.inference import Inferencer
from cyy_naive_pytorch_lib.ml_types import MachineLearningPhase
from cyy_naive_pytorch_lib.model_util import ModelUtil
from cyy_naive_pytorch_lib.trainer import Trainer
from torch.nn.intrinsic.quantized.modules.conv_relu import ConvReLU2d
from torch.nn.quantized.modules.linear import Linear
from torch.optim.sgd import SGD

from fed_quant_server import FedQuantServer
from quant_model import QuantedModel
from worker import Worker


class FedQuantWorker(Worker):
    def __init__(self, trainer: Trainer, server: FedQuantServer, **kwargs):
        super().__init__(trainer, server)
        assert isinstance(trainer.get_optimizer(), SGD)

        self.local_epoch = kwargs.get("local_epoch")
        assert self.local_epoch
        self.original_model = trainer.model
        self.quantized_model = None

    def train(self, device):
        self.__prepare_quantization()
        self.trainer.train(
            device=device, after_epoch_callbacks=[self.__send_parameters]
        )

    def __prepare_quantization(self):
        quant_model = QuantedModel(copy.deepcopy(self.original_model))
        quant_model.cpu()
        quant_model.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")
        torch.quantization.fuse_modules(
            quant_model,
            [
                ["module.convnet.c1", "module.convnet.relu1"],
                ["module.convnet.c3", "module.convnet.relu3"],
                ["module.convnet.c5", "module.convnet.relu5"],
                ["module.fc.f6", "module.fc.relu6"],
            ],
            inplace=True,
        )
        torch.quantization.prepare_qat(quant_model, inplace=True)
        self.trainer.set_model(quant_model)

    def __get_parameter_list(self):
        return ModelUtil(self.trainer.model).get_parameter_list()

    def __get_quantized_model(self) -> torch.nn.Module:
        self.trainer.model.cpu()
        self.trainer.model.eval()
        if self.quantized_model is None:
            self.quantized_model = torch.quantization.convert(self.trainer.model)
        return self.quantized_model

    def __get_quantized_parameters(self) -> dict:
        quantized_model = self.__get_quantized_model()
        processed_modules = set()
        state_dict = quantized_model.state_dict()
        quantized_model_util = ModelUtil(quantized_model)
        parameter_dict: dict = dict()
        for k in state_dict:
            if k in ("scale", "zero_point", "quant.scale", "quant.zero_point"):
                continue
            if "." not in k:
                continue
            module_name = ".".join(k.split(".")[:-1])
            if module_name in processed_modules:
                continue
            if not quantized_model_util.has_attr(module_name):
                continue
            sub_module = quantized_model_util.get_attr(module_name)
            module_name = module_name[len("module."):]
            if isinstance(sub_module, (ConvReLU2d, Linear)):
                weight, bias = sub_module._weight_bias()
                assert weight.is_quantized
                scale = weight.q_per_channel_scales()
                zero_point = weight.q_per_channel_zero_points()
                weight = weight.detach().int_repr()
                bias = bias.detach()
                parameter_dict[module_name + ".weight"] = weight
                parameter_dict[module_name + ".weight.scale"] = scale
                parameter_dict[module_name + ".weight.zero_point"] = zero_point
                parameter_dict[module_name + ".bias"] = bias
                processed_modules.add(module_name)
                continue
            get_logger().warn("unsupported sub_module type " + str(type(sub_module)))

        return parameter_dict

    def __load_quantized_parameters(self, parameter_dict: dict) -> dict:
        model_util = ModelUtil(self.original_model)
        quantized_model = self.__get_quantized_model()
        processed_modules = set()
        state_dict = quantized_model.state_dict()
        quantized_model_util = ModelUtil(quantized_model)
        for k in state_dict:
            if k in ("scale", "zero_point", "quant.scale", "quant.zero_point"):
                continue
            if "." not in k:
                continue
            module_name = ".".join(k.split(".")[:-1])
            if module_name in processed_modules:
                continue
            if not quantized_model_util.has_attr(module_name):
                continue
            sub_module = quantized_model_util.get_attr(module_name)
            module_name = module_name[len("module."):]
            if isinstance(sub_module, (ConvReLU2d, Linear)):
                processed_modules.add(module_name)
                weight = parameter_dict[module_name + ".weight"]
                weight = weight.float()
                scale = parameter_dict[module_name + ".weight.scale"]
                zero_point = parameter_dict[module_name + ".weight.zero_point"]
                for idx, v in enumerate(weight):
                    weight[idx] = (v - zero_point[idx]) * scale[idx]

                model_util.set_attr(module_name + ".weight", weight)
                model_util.set_attr(
                    module_name + ".bias", parameter_dict[module_name + ".bias"]
                )
                continue

        return parameter_dict

    def __send_parameters(self, trainer: Trainer, epoch, **kwargs):
        if epoch % self.local_epoch != 0:
            return

        parameter_dict = self.__get_quantized_parameters()

        self.server.add_parameter_dict(parameter_dict)
        parameter_dict = copy.deepcopy(self.server.get_parameter_dict())
        self.__load_quantized_parameters(parameter_dict)

        device = kwargs.get("device")
        self.trainer.set_model(self.quantized_model)
        res = self.trainer.get_inferencer(
            MachineLearningPhase.Test, copy_model=False
        ).inference(device=get_cpu_device())

        self.__prepare_quantization()

        res = self.trainer.get_inferencer(
            MachineLearningPhase.Test, copy_model=False
        ).inference(device=device)
        get_logger().info("after aggregating res is %s", res)
        optimizer: torch.optim.Optimizer = kwargs.get("optimizer")
        optimizer.param_groups.clear()
        optimizer.add_param_group({"params": self.trainer.model.parameters()})
        self.quantized_model = None
        return
