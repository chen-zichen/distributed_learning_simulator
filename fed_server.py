from typing import Optional

from cyy_naive_lib.data_structure.task_queue import RepeatedResult
from cyy_naive_lib.data_structure.thread_task_queue import ThreadTaskQueue
from cyy_naive_lib.log import get_logger

from server import Server


class FedServer(Server):
    def __init__(self, worker_number: int):
        super().__init__()
        self.worker_number = worker_number
        self.joined_clients = 0
        self.sum_parameter: Optional[dict] = None
        self.parameter = None
        self.parameter_queue = ThreadTaskQueue(worker_fun=self.__worker, worker_num=1)

    def stop(self):
        self.parameter_queue.stop()

    def add_parameter_dict(self, parameter: dict):
        self.parameter = parameter
        self.parameter_queue.add_task(parameter)

    def get_parameter_dict(self):
        return self.parameter_queue.get_result()

    def _process_client_parameter(self, client_parameter: dict):
        return client_parameter

    def _process_aggregated_parameter(self, aggregated_parameter: dict):
        return aggregated_parameter

    def __worker(self, parameter_dict: dict, __):
        self.joined_clients += 1
        parameter_dict = self._process_client_parameter(parameter_dict)

        if self.sum_parameter is None:
            self.sum_parameter = parameter_dict
        else:
            for k, v in self.sum_parameter.items():
                self.sum_parameter[k] += parameter_dict[k]

        if self.joined_clients != self.worker_number:
            get_logger().info("%s %s,skip", self.joined_clients, self.worker_number)
            return None
        self.joined_clients = 0
        get_logger().info("begin aggregating")

        sum_parameter = self.sum_parameter
        self.sum_parameter = None
        for k, v in sum_parameter.items():
            sum_parameter[k] = v / self.worker_number

        data = self._process_aggregated_parameter(sum_parameter)

        get_logger().info("end aggregating")
        return RepeatedResult(
            data=data,
            num=self.worker_number,
        )