from cyy_naive_lib.log import get_logger
from cyy_naive_pytorch_lib.model_executor import ModelExecutorCallbackPoint
from cyy_naive_pytorch_lib.model_util import ModelUtil

from .worker import Worker


class FedWorker(Worker):
    def __init__(self, **kwargs):
        worker_round = kwargs.pop("round")
        super().__init__(**kwargs)
        self.round = worker_round
        self.trainer.add_named_callback(
            ModelExecutorCallbackPoint.AFTER_EXECUTE,
            "send_parameter",
            self.__send_parameters,
        )

    def train(self, device):
        self.trainer.set_device(device)
        for _ in range(self.round):
            self.trainer.train()

    def __send_parameters(self, **kwargs):
        trainer = kwargs["model_executor"]
        parameter_dict = ModelUtil(trainer.model).get_parameter_dict(detach=True)

        get_logger().info("add_parameter_dict")
        self.worker_data_queue.add_task(
            (self.worker_id, len(trainer.dataset), parameter_dict)
        )
        get_logger().info("end add_parameter_dict")
        parameter_dict = self.worker_data_queue.get_result()
        ModelUtil(trainer.model).load_parameter_dict(parameter_dict)
        get_logger().info("end load_parameter_dict")
