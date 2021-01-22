from typing import List

import torch
from cyy_naive_lib.data_structure.task_queue import RepeatedResult
from cyy_naive_lib.data_structure.thread_task_queue import ThreadTaskQueue

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
        if self.worker_number == 1:
            return self.client_parameters[0]

        assert self.worker_number > 1

        total_parameter = self.client_parameters.pop(0)
        assert len(self.client_parameters) == self.worker_number - 1

        for k in total_parameter.keys():
            v = total_parameter[k]
            if isinstance(v, torch.Tensor):
                for client_parameter in self.client_parameters:
                    v += client_parameter[k]
                v /= self.worker_number
                continue
            if isinstance(v, tuple):
                v = tuple(
                    [
                        sum(x) / self.worker_number
                        for x in zip(*([p[k] for p in self.client_parameters] + [v]))
                    ]
                )
                continue
            if isinstance(v, torch.dtype):
                continue
            print(type(v))
            assert False

        self.client_parameters = []
        return RepeatedResult(data=total_parameter, num=self.worker_number)
