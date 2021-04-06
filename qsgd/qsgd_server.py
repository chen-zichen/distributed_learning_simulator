from typing import List

import torch
import numpy as np
from cyy_naive_lib.data_structure.task_queue import RepeatedResult
from cyy_naive_lib.data_structure.thread_task_queue import ThreadTaskQueue

from server import Server


class QsgdServer(Server):
    def __init__(self, worker_number: int):
        super().__init__()
        self.worker_number = worker_number
        self.q_gradients: list = []
        self.gradients_queue = ThreadTaskQueue(
            worker_fun=self.__worker, worker_num=1)

    def stop(self):
        self.gradients_queue.stop()

    def add_gradient(self, q_gradients: List[torch.Tensor]):
        self.gradients_queue.add_task(q_gradients)

    def get_gradient(self) -> List[torch.Tensor]:
        return self.gradients_queue.get_result()


    def __worker(self, q_gradients: torch.Tensor, extra_args):
        self.q_gradients.append(q_gradients)
        if len(self.q_gradients) != self.worker_number:
            return None
        total_q_gradients = [sum(i) for i in zip(*self.q_gradients)]
        # for idx, grad in enumerate(total_q_gradients):
        #     total_q_gradients[idx] = torch.sign(q_gradients)

        self.q_gradients = []
        return RepeatedResult(data=total_q_gradients, num=self.worker_number)

