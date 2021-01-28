import copy

import torch
from cyy_naive_lib.log import get_logger
from cyy_naive_pytorch_lib.device import get_cpu_device
from cyy_naive_pytorch_lib.ml_types import MachineLearningPhase
from cyy_naive_pytorch_lib.model_util import ModelUtil
from cyy_naive_pytorch_lib.trainer import Trainer
from torch.quantization.fuser_method_mappings import OP_LIST_TO_FUSER_METHOD

from fed_quant_server import FedQuantServer
# from quant_model import QuantedModel
from worker import Worker


class FedQuantWorker(Worker):
    def __init__(self, trainer: Trainer, server: FedQuantServer, **kwargs):
        super().__init__(trainer, server)

        self.local_epoch = kwargs.get("local_epoch")
        assert self.local_epoch
        self.original_model = trainer.model
        self.quantized_model = None
        # # change ReLU6 to ReLU
        ModelUtil(self.original_model).change_sub_modules(
            torch.nn.modules.activation.ReLU6,
            lambda name, sub_module: torch.nn.modules.activation.ReLU(
                inplace=sub_module.inplace
            ),
        )
        get_logger().debug("model is %s", self.original_model)

    def train(self, device):
        self.__prepare_quantization()
        self.trainer.set_device(device)
        self.trainer.train(after_epoch_callbacks=[self.__send_parameters])

    def __get_fused_modules(self, model):
        modules = list(model.named_modules())
        list_of_list = []
        i = 0
        while i < len(modules):
            candidates: set = set(OP_LIST_TO_FUSER_METHOD.keys())
            j = i
            end_index = None
            while j < len(modules):
                module = modules[j][1]
                new_candidates = set()
                for candidate in candidates:
                    if isinstance(module, candidate[0]):
                        if len(candidate) == 1:
                            end_index = j
                        else:
                            new_candidates.add(candidate[1:])
                if not new_candidates:
                    break
                candidates = new_candidates
                j += 1
            if end_index is not None:
                module_name_list = []
                while i <= end_index:
                    module_name_list.append(modules[i][0])
                    i += 1
                list_of_list.append(module_name_list)
            else:
                i += 1
        get_logger().debug("list_of_list is %s", list_of_list)
        return list_of_list

    def __prepare_quantization(self):
        if ModelUtil(self.original_model).has_sub_module(torch.quantization.QuantStub):
            quant_model = copy.deepcopy(self.original_model)
        else:
            quant_model = torch.quantization.QuantWrapper(
                copy.deepcopy(self.original_model)
            )
        quant_model.cpu()
        quant_model.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")
        get_logger().debug("quant_model is %s", quant_model)

        if hasattr(quant_model, "fuse_model"):
            get_logger().debug("use fuse_model")
            quant_model.fuse_model()
        else:
            torch.quantization.fuse_modules(
                quant_model,
                self.__get_fused_modules(quant_model),
                inplace=True,
            )
        torch.quantization.prepare_qat(quant_model, inplace=True)
        self.trainer.set_model(quant_model)

    def __get_quantized_model(self) -> torch.nn.Module:
        self.trainer.model.cpu()
        self.trainer.model.eval()
        if self.quantized_model is None:
            self.quantized_model = torch.quantization.convert(self.trainer.model)
        return self.quantized_model

    def __get_quantized_parameters(self) -> dict:
        quantized_model = self.__get_quantized_model()
        # get_logger().info("quantized_model  is %s", quantized_model)
        processed_modules = set()
        state_dict = quantized_model.state_dict()
        quantized_model_util = ModelUtil(quantized_model)
        parameter_dict: dict = dict()
        for k in state_dict:
            # get_logger().info("k is %s", k)
            if k in ("scale", "zero_point", "quant.scale", "quant.zero_point"):
                continue
            if "." not in k:
                get_logger().info("skip k is %s", k)
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
                get_logger().info("batchnorm2d k is %s", k)
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

        for n, _ in quantized_model_util.get_parameter_dict().items():
            get_logger().info("quantized_model key is %s", n)
        return parameter_dict

    def __load_quantized_parameters(self, parameter_dict: dict) -> dict:
        model_util = ModelUtil(self.original_model)
        quantized_model = self.__get_quantized_model()
        processed_modules = set()
        state_dict = quantized_model.state_dict()
        # get_logger().info("state_dict is %s", state_dict)
        quantized_model_util = ModelUtil(quantized_model)
        # get_logger().info("aaaaaaaaaaa %s  %s ", name, type(module))
        for name, module in self.original_model.named_modules():
            if isinstance(module, torch.nn.modules.BatchNorm2d):
                # get_logger().info("set batchnorm  of %s ", name)
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

                # for suffix in [".bias", ".running_mean", ".running_var"]:
                for suffix in [".bias"]:
                    attr_name = module_name + suffix
                    if attr_name in parameter_dict:
                        model_util.set_attr(attr_name, parameter_dict[attr_name])
                continue
            get_logger().warning("unsupported sub_module type %s", type(sub_module))
            # raise RuntimeError("unsupported sub_module type " + str(type(sub_module)))
        return parameter_dict

    def __send_parameters(self, trainer: Trainer, epoch, **kwargs):
        if epoch % self.local_epoch != 0:
            return

        parameter_dict = self.__get_quantized_parameters()
        # self.trainer.set_model(self.quantized_model)

        # inferencer = self.trainer.get_inferencer(
        #     MachineLearningPhase.Test, copy_model=False
        # )
        # inferencer.set_device(get_cpu_device())

        # res = inferencer.inference()
        # get_logger().info("quantized res is %s", res)

        get_logger().debug("quantized_model is %s", self.quantized_model)

        self.server.add_parameter_dict(parameter_dict)
        parameter_dict = copy.deepcopy(self.server.get_parameter_dict())
        self.__load_quantized_parameters(parameter_dict)

        self.__prepare_quantization()
        inferencer = self.trainer.get_inferencer(
            MachineLearningPhase.Test, copy_model=False
        )
        inferencer.set_device(get_cpu_device())
        res = inferencer.inference()

        get_logger().info("after aggregating res is %s", res)
        optimizer: torch.optim.Optimizer = kwargs.get("optimizer")
        optimizer.param_groups.clear()
        optimizer.add_param_group({"params": self.trainer.model.parameters()})
        self.quantized_model = None
        return
