from typing import List

from cyy_naive_lib.data_structure.task_queue import RepeatedResult
from cyy_naive_lib.data_structure.thread_task_queue import ThreadTaskQueue
from cyy_naive_lib.log import get_logger

from server import Server


class FedQuantServer(Server):
    def __init__(self, worker_number: int):
        super().__init__()
        self.worker_number = worker_number
        self.client_parameters: list = []
        self.parameter_queue = ThreadTaskQueue(worker_fun=self.__worker, worker_num=1)

    def stop(self):
        self.parameter_queue.stop()

    def add_parameter_dict(self, parameter: dict):
        self.parameter_queue.add_task(parameter)

    def get_parameter_dict(self) -> List[dict]:
        return self.parameter_queue.get_result()

    def __worker(self, parameter_dict: dict, extra_args):
        self.client_parameters.append(parameter_dict)
        if len(self.client_parameters) != self.worker_number:
            get_logger().info(
                "%s %s,skip", len(self.client_parameters), self.worker_number
            )
            return None
        get_logger().info("begin aggregating")
        if self.worker_number == 1:
            return self.client_parameters[0]

        for idx, parameter_dict in enumerate(self.client_parameters):
            for k, v in parameter_dict.items():
                if isinstance(v, tuple):
                    (weight, scale, zero_point) = v
                    weight = weight.float()
                    for idx2, v2 in enumerate(weight):
                        weight[idx2] = (v2 - zero_point[idx2]) * scale[idx2]
                    self.client_parameters[idx][k] = weight

        total_parameter: dict = dict()
        for k in self.client_parameters[0]:
            get_logger().info("process %s", k)
            total_parameter[k] = (
                sum([p[k].float() for p in self.client_parameters]) / self.worker_number
            )

        self.client_parameters = []
        get_logger().info("end aggregating %s", total_parameter)
        return RepeatedResult(data=total_parameter, num=self.worker_number)
