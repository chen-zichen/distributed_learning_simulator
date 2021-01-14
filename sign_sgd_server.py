from typing import List

import torch
from cyy_naive_lib.data_structure.process_task_queue import ProcessTaskQueue
from cyy_naive_lib.data_structure.task_queue import RepeatedResult

from server import Server


class SignSGDServer(Server):
    def __init__(self, worker_number: int):
        super().__init__()
        self.worker_number = worker_number
        self.sign_gradients: list = []
        self.gradients_queue = ProcessTaskQueue(
            worker_fun=self.__worker, worker_num=1)

    def stop(self):
        self.gradients_queue.stop()

    def add_gradient(self, sign_gradient: List[torch.Tensor]):
        self.gradients_queue.add_task(sign_gradient)

    def get_gradient(self) -> List[torch.Tensor]:
        return self.gradients_queue.get_result()

    def __worker(self, sign_gradient: torch.Tensor, extra_args):
        self.sign_gradients.append(sign_gradient)
        if len(self.sign_gradients) != self.worker_number:
            return None
        total_sign_gradient = [sum(i) for i in zip(*self.sign_gradients)]
        for idx, grad in enumerate(total_sign_gradient):
            total_sign_gradient[idx] = torch.sign(grad)

        self.sign_gradients = []
        return RepeatedResult(data=total_sign_gradient, num=self.worker_number)
