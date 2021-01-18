from typing import List

import torch
from cyy_naive_lib.data_structure.task_queue import RepeatedResult
from cyy_naive_lib.data_structure.thread_task_queue import ThreadTaskQueue
from cyy_naive_pytorch_lib.tensor import cat_tensors_to_vector

from server import Server


class FedQuantServer(Server):
    def __init__(self, worker_number: int):
        super().__init__()
        self.worker_number = worker_number
        self.client_parameters: list = []
        self.parameter_queue = ThreadTaskQueue(worker_fun=self.__worker, worker_num=1)

    def stop(self):
        self.parameter_queue.stop()

    def add_parameter(self, parameter: torch.Tensor):
        self.parameter_queue.add_task(parameter)

    def get_parameter(self) -> List[torch.Tensor]:
        return self.parameter_queue.get_result()

    def __worker(self, parameter: torch.Tensor, extra_args):
        self.client_parameters.append(parameter)
        if len(self.client_parameters) != self.worker_number:
            return None
        total_parameter = cat_tensors_to_vector(
            [sum(i) / self.worker_number for i in zip(*self.client_parameters)]
        )
        self.client_parameters = []
        return RepeatedResult(data=total_parameter, num=self.worker_number)
