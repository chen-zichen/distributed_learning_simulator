import abc

from cyy_naive_pytorch_lib.trainer import Trainer


class Worker(abc.ABC):
    def __init__(self, worker_id, trainer: Trainer, worker_data_queue):
        super().__init__()
        self.__worker_id = worker_id
        self.trainer = trainer
        self.__worker_data_queue = worker_data_queue

    @property
    def worker_id(self):
        return self.__worker_id

    @property
    def worker_data_queue(self):
        return self.__worker_data_queue

    # @abc.abstractmethod
    def train(self, device):
        pass
