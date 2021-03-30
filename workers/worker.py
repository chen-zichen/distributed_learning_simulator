import abc

from cyy_naive_pytorch_lib.trainer import Trainer


class Worker(abc.ABC):
    def __init__(self, trainer: Trainer, server, worker_id):
        super().__init__()
        self.trainer = trainer
        self.server = server
        self.__worker_id = worker_id

    @property
    def worker_id(self):
        return self.__worker_id

    # @abc.abstractmethod
    def train(self, device):
        pass
