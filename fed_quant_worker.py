import copy

import torch
from cyy_naive_lib.log import get_logger
from cyy_naive_pytorch_lib.device import get_cpu_device, get_device
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

    def __send_parameters(self, trainer: Trainer, epoch, **kwargs):
        if epoch % self.local_epoch != 0:
            return

        trainer.model.cpu()
        trainer.model.eval()
        old_model = copy.deepcopy(trainer.model)
        quantized_model: torch.nn.Module = torch.quantization.convert(
            trainer.model, remove_qconfig=False
        )

        device = kwargs.get("device")
        self.trainer.set_model(quantized_model)
        res = self.trainer.get_inferencer(
            MachineLearningPhase.Test, copy_model=False
        ).inference(device=get_cpu_device())

        state_dict = quantized_model.state_dict()
        quantized_model_util = ModelUtil(quantized_model)
        model_util = ModelUtil(self.original_model)
        processed_modules = set()
        old_quant_scale = None
        old_quant_zero_point = None

        parameter_dict: dict = dict()
        for k, v in state_dict.items():
            if k in ("scale", "zero_point"):
                continue
            if k == "quant.scale":
                old_quant_scale = v
                continue
            if k == "quant.zero_point":
                old_quant_zero_point = v
                continue

            module_name = ".".join(k.split(".")[:-1])
            if module_name in processed_modules:
                continue
            if not quantized_model_util.has_attr(module_name):
                continue
            sub_module = quantized_model_util.get_attr(module_name)
            get_logger().info("sub_module type is %s", type(sub_module))
            if isinstance(sub_module, ConvReLU2d):
                weight, bias = sub_module._weight_bias()
                assert weight.is_quantized
                weight = weight.int_repr().float()
                processed_modules.add(module_name)
                module_name = module_name[len("module."):]
                # get_logger().info("conv_relu2d weight is %s", weight)
                # get_logger().info("conv_relu2d bias is %s", bias)
                # get_logger().info(
                #     "set bias for %s.weight %s.bias", module_name, module_name
                # )
                parameter_dict[module_name + ".weight"] = weight
                parameter_dict[module_name + ".bias"] = bias
                model_util.set_attr(module_name + ".weight", weight)
                model_util.set_attr(module_name + ".bias", bias)
                continue

            if isinstance(sub_module, Linear):
                weight, bias = sub_module._packed_params._weight_bias()
                assert weight.is_quantized
                weight = weight.int_repr().float()
                processed_modules.add(module_name)
                # get_logger().info(" weight is %s bias is %s", weight, bias)
                module_name = module_name[len("module."):]
                parameter_dict[module_name + ".weight"] = weight
                parameter_dict[module_name + ".bias"] = bias
                model_util.set_attr(module_name + ".weight", weight)
                model_util.set_attr(module_name + ".bias", bias)
                # get_logger().info( "set value for %s weight=%s bias =%s", module_name, weight, bias)
                continue
        self.__prepare_quantization()

        scale = old_model.scale * old_quant_scale
        zero_point = old_model.zero_point / old_quant_scale + old_quant_zero_point
        self.trainer.model.register_buffer("scale", scale)
        self.trainer.model.register_buffer("zero_point", zero_point)
        get_logger().info("register scale and zero_point %s %s", scale, zero_point)

        device = kwargs.get("device")
        res = self.trainer.get_inferencer(
            MachineLearningPhase.Test, copy_model=False
        ).inference(device=device)
        get_logger().info("res is %s", res)
        optimizer: torch.optim.Optimizer = kwargs.get("optimizer")
        optimizer.param_groups.clear()
        optimizer.add_param_group({"params": self.trainer.model.parameters()})
        return

        # self.__prepare_quantization()
        # self.trainer.set_model(self.original_model)

        # get_logger().info("aggregate parameters at epoch %s", epoch)
        # self.server.add_parameter(model_util.get_parameter_list())
        # parameter_list = copy.deepcopy(self.server.get_parameter_list())
        # ModelUtil(trainer.model).load_parameter_list(parameter_list)
        # get_logger().info("finish aggregating parameters at epoch %s", epoch)
