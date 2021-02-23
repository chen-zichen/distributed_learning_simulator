from typing import List, Optional

from cyy_naive_lib.algorithm.mapping_op import get_mapping_values_by_order
from cyy_naive_lib.data_structure.task_queue import RepeatedResult
from cyy_naive_lib.data_structure.thread_task_queue import ThreadTaskQueue
from cyy_naive_lib.log import get_logger
from cyy_naive_pytorch_lib.algorithm.quantization.scheme import \
    stochastic_quantization
from cyy_naive_pytorch_lib.tensor import TensorUtil

from server import Server


class FedQuantServer(Server):
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
        quantized_pair, dequant = self.parameter_queue.get_result()
        tensor_util = TensorUtil(self.parameter)
        tensor_util.load_dict_values(dequant(quantized_pair))
        return tensor_util.data

    def __worker(self, parameter_dict: dict, __):
        self.joined_clients += 1

        for k, v in parameter_dict.items():
            if isinstance(v, tuple):
                (weight, scale, zero_point) = v
                weight = weight.float()
                for idx, v in enumerate(weight):
                    weight[idx] = (v - zero_point[idx]) * scale[idx]
                parameter_dict[k] = weight

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

        get_logger().info("begin quantization")
        tensor_util = TensorUtil(sum_parameter)
        quant, dequant = stochastic_quantization(256)
        quantized_pair = quant(tensor_util.concat_dict_values())

        get_logger().info("end quantization")
        get_logger().info("end aggregating")
        return RepeatedResult(
            data=(quantized_pair, dequant),
            num=self.worker_number,
        )
