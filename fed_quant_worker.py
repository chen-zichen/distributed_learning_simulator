import copy
import os
import sys

import torch
from cyy_naive_lib.log import get_logger
from cyy_naive_pytorch_lib.device import get_cpu_device, get_device
from cyy_naive_pytorch_lib.inference import Inferencer
from cyy_naive_pytorch_lib.ml_types import MachineLearningPhase
from cyy_naive_pytorch_lib.model_util import ModelUtil
from cyy_naive_pytorch_lib.trainer import Trainer
from torch.optim.sgd import SGD

from fed_quant_server import FedQuantServer
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
        quant_model = torch.quantization.QuantWrapper(
            copy.deepcopy(self.original_model)
        )
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

        # for k, v in trainer.model.quant.named_buffers():
        #     print("buffer k=", k)
        #     print("buffer v=", v)

        trainer.model.cpu()
        trainer.model.eval()
        quantized_model: torch.nn.Module = torch.quantization.convert(trainer.model)
        get_logger().info("quantized_model is %s",quantized_model)

        state_dict = quantized_model.state_dict()

        model_util = ModelUtil(self.original_model)
        for k in state_dict.keys():
            prefix = "module."
            if not k.startswith(prefix):
                continue
            v = state_dict[k]
            k = k[len(prefix):]
            print("k=", k)
            if not isinstance(v, torch.Tensor):
                continue

            if v.is_quantized:
                v = v.int_repr().float()
                # get_logger().info("for quantization k=%s", k)
            # else:
            # get_logger().info("for not quantization k=%s", k)
            if model_util.has_attr(k):
                model_util.set_attr(k, v)
                get_logger().info("set value for k=%s", k)
            else:
                get_logger().info("no value for k=%s", k)

        self.__prepare_quantization()
        model_util = ModelUtil(self.trainer.model)
        device = kwargs.get("device")
        for k in state_dict.keys():
            print("k=", k)
            v = state_dict[k]
            if k == "quant.scale":
                fake_quant = model_util.get_attr("quant.activation_post_process")
                fake_quant.register_buffer("scale", v)
                # get_logger().info("register scale %s", v)
                continue
            if k == "quant.zero_point":
                fake_quant = model_util.get_attr("quant.activation_post_process")
                fake_quant.register_buffer("zero_point", v)
                get_logger().info("register zero_point %s", v)
                continue

        res = self.trainer.get_inferencer(
            MachineLearningPhase.Test, copy_model=False
        ).inference(device=device)
        get_logger().info("aaaaaaaaaaaaaaaaa=", res)
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
