import torch
from cyy_naive_lib.data_structure.process_task_queue import ProcessTaskQueue

from server import Server


class SignSGDServer(Server):
    def __init__(self, worker_number: int):
        super().__init__()
        self.worker_number = worker_number
        self.sign_gradients: list = []
        self.gradients_queue = ProcessTaskQueue(
            processor_fun=self.__processor, worker_num=1
        )

    def stop(self):
        self.gradients_queue.stop()

    def __processor(self, sign_gradient: torch.Tensor):
        self.sign_gradients.append(sign_gradient)
        if len(self.sign_gradients) != self.worker_number:
            return None
        total_sign_gradient = sum(self.sign_gradients)
        self.sign_gradients = []
        return total_sign_gradient
