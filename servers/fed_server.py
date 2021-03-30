from cyy_naive_lib.data_structure.task_queue import RepeatedResult
from cyy_naive_lib.data_structure.thread_task_queue import ThreadTaskQueue
from cyy_naive_lib.log import get_logger

from .server import Server


class FedServer(Server):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.round = 0
        self.parameters: dict = dict()
        self.parameter_queue = ThreadTaskQueue(worker_fun=self.__worker, worker_num=1)

    def stop(self):
        self.parameter_queue.stop()

    def add_parameter_dict(self, worker_id, parameter: dict):
        self.parameter_queue.add_task((worker_id, parameter))

    def get_parameter_dict(self):
        return self.parameter_queue.get_result()

    def _process_client_parameter(self, client_parameter: dict):
        return client_parameter

    def _process_aggregated_parameter(self, aggregated_parameter: dict):
        return aggregated_parameter

    def get_subset_model(self, client_subset, init_model=None):
        # empty set
        if not client_subset:
            assert init_model is not None
            return init_model
        avg_parameter: dict = None

        for idx in client_subset:
            parameter = self.parameters[idx]
            if avg_parameter is None:
                avg_parameter = parameter
            else:
                for k in avg_parameter:
                    avg_parameter[k] += parameter[k]
        for k, v in avg_parameter.items():
            avg_parameter[k] = v / len(client_subset)
        return avg_parameter

    def __worker(self, data, __):
        worker_id, parameter_dict = data
        self.parameters[worker_id] = self._process_client_parameter(parameter_dict)

        if len(self.parameters) != self.worker_number:
            get_logger().info("%s %s,skip", len(self.parameters), self.worker_number)
            return None
        self.round += 1
        get_logger().info("begin aggregating")

        avg_parameter = self.get_subset_model(self.parameters.keys())

        data = self._process_aggregated_parameter(avg_parameter)
        get_logger().info("end aggregating")
        self.parameters.clear()
        return RepeatedResult(
            data=data,
            num=self.worker_number,
        )
