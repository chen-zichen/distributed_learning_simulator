import abc

from cyy_naive_lib.data_structure.task_queue import TaskQueue
from cyy_naive_pytorch_lib.trainer import Trainer

from worker import Worker


class SignSGDWorker(Worker):
    def __init__(self, trainer: Trainer, accumulated_queue: TaskQueue):
        super().__init__()
        self.trainer = trainer

    @abc.abstractmethod
    def train(self):
        pass
