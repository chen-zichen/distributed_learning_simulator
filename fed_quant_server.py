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

        for idx, parameter_dict in enumerate(self.client_parameters):
            for k, v in parameter_dict.items():
                if isinstance(v, tuple):
                    (weight, scale, zero_point) = v
                    weight = weight.float()
                    for idx, v in enumerate(weight):
                        weight[idx] = (v - zero_point[idx]) * scale[idx]
                    parameter_dict[k] = weight

                    # parameter_dict[k] = (
                    #     weight.float(),
                    #     scale.float(),
                    #     zero_point.float(),
                    # )
                    # get_logger().error("client %s %s scale is %s", idx, k, scale)

        total_parameter: dict = dict()
        for k, v in self.client_parameters[0].items():
            if isinstance(v, tuple):
                (weight, scale, zero_point) = v
                get_logger().error("before weight %s client 0 is %s", k, weight)
                get_logger().error(
                    "before weight %s client 1 is %s",
                    k,
                    self.client_parameters[1][k][0],
                )
                total_parameter[k] = tuple(
                    map(sum, zip(*[p[k] for p in self.client_parameters]))
                )
                assert len(total_parameter[k]) == 3
                (weight, scale, zero_point) = total_parameter[k]
                weight /= self.worker_number
                get_logger().error("after weight %s is %s", k, weight)
                zero_point /= self.worker_number
                scale /= self.worker_number
                total_parameter[k] = (weight, scale, zero_point)
            else:
                total_parameter[k] = (
                    sum([p[k].float() for p in self.client_parameters])
                    / self.worker_number
                )

        self.client_parameters = []
        get_logger().info("end aggregating")
        return RepeatedResult(data=total_parameter, num=self.worker_number)
