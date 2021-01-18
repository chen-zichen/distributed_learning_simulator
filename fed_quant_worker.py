import copy

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
        self.parameter_names = list(
            sorted(ModelUtil(self.trainer.model).get_parameter_dict().keys())
        )
        self.trainer.set_model(
            QuantedModel(self.trainer.model, QuantModel(in_features=parameter_size))
        )

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
        ModelUtil(trainer.model).load_parameter_dict(
            dict(zip(self.parameter_names, parameter_list))
        )
        get_logger().info("finish aggregating parameters at epoch %s", epoch)
