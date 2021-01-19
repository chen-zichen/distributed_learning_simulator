import copy

import torch
from cyy_naive_lib.log import get_logger
from cyy_naive_pytorch_lib.model_util import ModelUtil
from cyy_naive_pytorch_lib.trainer import Trainer
from torch.optim.sgd import SGD

from fed_quant_server import FedQuantServer
from quant_model import QuantedModel, QuantModel
from worker import Worker


class FedQuantWorker(Worker):
    def __init__(self, trainer: Trainer, server: FedQuantServer, **kwargs):
        super().__init__(trainer, server)
        assert isinstance(trainer.get_optimizer(), SGD)

        self.local_epoch = kwargs.get("local_epoch")
        parameter_size = self.__get_parameter_list().shape[0]

        quant_model = QuantModel(in_features=parameter_size)
        quant_model.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")
        # quant_model_fused = torch.quantization.fuse_modules(
        #     quant_model, [["linear1", "relu1"], ["linear2", "relu2"]]
        # )
        # self.quant_model_prepared = torch.quantization.prepare_qat(quant_model_fused)
        self.quant_model_prepared = torch.quantization.prepare_qat(quant_model)
        self.quanted_model = QuantedModel(self.trainer.model, self.quant_model_prepared)
        self.trainer.set_model(self.quanted_model)

    def train(self, device):
        self.trainer.train(device=device, after_epoch_callbacks=[])

    def __get_parameter_list(self):
        return ModelUtil(self.trainer.model).get_parameter_list()

    def __send_parameters(self, trainer, epoch, **kwargs):
        if epoch % self.local_epoch != 0:
            return
        get_logger().info("aggregate parameters at epoch %s", epoch)
        self.server.add_parameter(self.__get_parameter_list())
        parameter_list = copy.deepcopy(self.server.get_parameter_list())
        ModelUtil(trainer.model).load_parameter_list(parameter_list)
        get_logger().info("finish aggregating parameters at epoch %s", epoch)
