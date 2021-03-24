import abc

from cyy_naive_pytorch_lib.trainer import Trainer


class Worker(abc.ABC):
    def __init__(self, trainer: Trainer, server):
        super().__init__()
        self.trainer = trainer
        self.server = server

    @abc.abstractmethod
    def train(self, device, worker_id):
        pass
