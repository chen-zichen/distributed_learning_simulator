import copy

import torch
from cyy_naive_lib.log import get_logger
from cyy_naive_pytorch_lib.algorithm.quantization.trainer import \
    QuantizationTrainer
from cyy_naive_pytorch_lib.device import get_cpu_device
from cyy_naive_pytorch_lib.ml_types import MachineLearningPhase
from cyy_naive_pytorch_lib.model_util import ModelUtil
from cyy_naive_pytorch_lib.trainer import Trainer

from fed_quant_server import FedQuantServer
from worker import Worker


class FedQuantWorker(Worker):
    def __init__(self, trainer: Trainer, server: FedQuantServer, **kwargs):
        super().__init__(QuantizationTrainer(trainer, replace_layer=False), server)

        self.local_epoch = kwargs.get("local_epoch")
        assert self.local_epoch

    def train(self, device):
        self.trainer.prepare_quantization()
        self.trainer.trainer.set_device(device)
        self.trainer.train(after_epoch_callbacks=[self.__send_parameters])

    def __get_quantized_parameters(self) -> dict:
        quantized_model = self.trainer.get_quantized_model()
        get_logger().info("quantized_model  is %s", quantized_model)
        processed_modules = set()
        state_dict = quantized_model.state_dict()
        quantized_model_util = ModelUtil(quantized_model)
        parameter_dict: dict = dict()
        for k in state_dict:
            get_logger().debug("attribute is %s", k)
            if k in ("scale", "zero_point", "quant.scale", "quant.zero_point"):
                continue
            if "." not in k:
                get_logger().debug("skip attribute is %s", k)
                continue
            module_name = ".".join(k.split(".")[:-1])
            if module_name in processed_modules:
                continue
            if not quantized_model_util.has_attr(module_name):
                continue
            sub_module = quantized_model_util.get_attr(module_name)
            if module_name.startswith("module."):
                module_name = module_name[len("module."):]
            if isinstance(
                sub_module,
                (
                    torch.nn.intrinsic.quantized.modules.conv_relu.ConvReLU2d,
                    torch.nn.quantized.modules.linear.Linear,
                    torch.nn.quantized.modules.conv.Conv2d,
                ),
            ):
                weight, bias = sub_module._weight_bias()
                assert weight.is_quantized
                scale = weight.q_per_channel_scales()
                zero_point = weight.q_per_channel_zero_points()
                weight = weight.detach().int_repr()
                parameter_dict[module_name + ".weight"] = (weight, scale, zero_point)
                if bias is not None:
                    bias = bias.detach()
                    parameter_dict[module_name + ".bias"] = bias
                processed_modules.add(module_name)
                continue
            if isinstance(
                sub_module,
                (torch.nn.quantized.modules.batchnorm.BatchNorm2d),
            ):
                get_logger().info("process BatchNorm2d %s", k)
                weight = sub_module.weight.detach()
                assert not weight.is_quantized
                bias = sub_module.bias.detach()
                assert not bias.is_quantized
                running_mean = sub_module.running_mean.detach()
                assert not running_mean.is_quantized
                running_var = sub_module.running_var.detach()
                assert not running_var.is_quantized

                parameter_dict[module_name + ".weight"] = weight
                parameter_dict[module_name + ".bias"] = bias
                parameter_dict[module_name + ".running_mean"] = running_mean
                parameter_dict[module_name + ".running_var"] = running_var
                processed_modules.add(module_name)
                continue
            get_logger().warning("unsupported sub_module type %s", type(sub_module))

        return parameter_dict

    def __load_quantized_parameters(self, parameter_dict: dict) -> dict:
        model_util = ModelUtil(self.trainer.original_model)
        quantized_model = self.trainer.get_quantized_model()
        processed_modules = set()
        state_dict = quantized_model.state_dict()
        quantized_model_util = ModelUtil(quantized_model)
        for name, module in self.trainer.original_model.named_modules():
            if isinstance(module, torch.nn.modules.BatchNorm2d):
                get_logger().info("ignore BatchNorm2d %s", name)
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.zeros_(module.running_mean)
                torch.nn.init.ones_(module.running_var)
                # module.eps = 0

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
            if module_name.startswith("module."):
                module_name = module_name[len("module."):]
            if isinstance(
                sub_module,
                (
                    torch.nn.intrinsic.quantized.modules.conv_relu.ConvReLU2d,
                    torch.nn.quantized.modules.linear.Linear,
                    torch.nn.quantized.modules.conv.Conv2d,
                    torch.nn.quantized.modules.batchnorm.BatchNorm2d,
                ),
            ):
                processed_modules.add(module_name)
                weight = parameter_dict[module_name + ".weight"]
                if isinstance(weight, tuple):
                    (weight, scale, zero_point) = weight
                    weight = weight.float()
                    for idx, v in enumerate(weight):
                        weight[idx] = (v - zero_point[idx]) * scale[idx]
                model_util.set_attr(module_name + ".weight", weight)

                # for suffix in [".bias"]:
                for suffix in [".bias", ".running_mean", ".running_var"]:
                    attr_name = module_name + suffix
                    if attr_name in parameter_dict:
                        model_util.set_attr(attr_name, parameter_dict[attr_name])
                continue
            get_logger().warning("unsupported sub_module type %s", type(sub_module))
        return parameter_dict

    def __send_parameters(self, trainer: Trainer, epoch, **kwargs):
        if epoch % self.local_epoch != 0:
            return

        # self.trainer.set_model(self.quantized_model)

        # inferencer = self.trainer.get_inferencer(
        #     MachineLearningPhase.Test, copy_model=False
        # )
        # inferencer.set_device(get_cpu_device())

        # res = inferencer.inference()
        # get_logger().info("quantized res is %s", res)

        parameter_dict = self.__get_quantized_parameters()
        self.server.add_parameter_dict(parameter_dict)
        parameter_dict = copy.deepcopy(self.server.get_parameter_dict())
        self.__load_quantized_parameters(parameter_dict)

        self.trainer.prepare_quantization()
        inferencer = self.trainer.trainer.get_inferencer(
            MachineLearningPhase.Test, copy_model=False
        )
        inferencer.set_device(get_cpu_device())
        res = inferencer.inference()

        get_logger().info("after aggregating res is %s", res)
        optimizer: torch.optim.Optimizer = kwargs.get("optimizer")
        optimizer.param_groups.clear()
        optimizer.add_param_group({"params": self.trainer.trainer.model.parameters()})
        self.trainer.reset_quantized_model()
        return
