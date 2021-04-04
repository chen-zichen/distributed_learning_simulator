import copy

from cyy_naive_lib.data_structure.task_queue import RepeatedResult
from cyy_naive_lib.log import get_logger
from cyy_naive_pytorch_lib.device import get_device
from cyy_naive_pytorch_lib.model_util import ModelUtil

from .server import Server


class FedServer(Server):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.round = 0
        self.parameters: dict = dict()
        self.__prev_model = copy.deepcopy(
            ModelUtil(self.tester.model).get_parameter_dict()
        )
        self.worker_data_queue.put_result(
            RepeatedResult(
                data=self.prev_model,
                num=self.worker_number,
            )
        )

    @property
    def prev_model(self):
        return self.__prev_model

    def _process_client_parameter(self, client_parameter: dict):
        return client_parameter

    def _process_aggregated_parameter(self, aggregated_parameter: dict):
        return aggregated_parameter

    def get_subset_model(self, client_subset):
        # empty set
        if not client_subset:
            return self.__prev_model
        avg_parameter: dict = dict()

        device = get_device()
        total_training_dataset_size = 0
        for idx in client_subset:
            total_training_dataset_size += self.parameters[idx][0]
        for idx in client_subset:
            training_dataset_size, parameter = self.parameters[idx]
            for k in parameter:
                tmp = (
                    parameter[k].to(device)
                    * training_dataset_size
                    / total_training_dataset_size
                )
                if k not in avg_parameter:
                    avg_parameter[k] = tmp
                else:
                    avg_parameter[k] += tmp
        return avg_parameter

    def _process_worker_data(self, data, __):
        worker_id, training_dataset_size, parameter_dict = data
        self.parameters[worker_id] = (
            training_dataset_size,
            self._process_client_parameter(parameter_dict),
        )

        if len(self.parameters) != self.worker_number:
            get_logger().debug("%s %s,skip", len(self.parameters), self.worker_number)
            return None
        self.round += 1
        get_logger().info("begin aggregating")

        avg_parameter = self.get_subset_model(self.parameters.keys())

        data = self._process_aggregated_parameter(avg_parameter)
        self.__prev_model = copy.deepcopy(data)
        get_logger().info("end aggregating")
        self.parameters.clear()
        return RepeatedResult(
            data=data,
            num=self.worker_number,
        )
