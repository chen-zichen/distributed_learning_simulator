from cyy_naive_lib.log import get_logger
from cyy_naive_pytorch_lib.algorithm.quantization.qat import \
    QuantizationAwareTraining
from cyy_naive_pytorch_lib.device import get_cpu_device
from cyy_naive_pytorch_lib.ml_type import MachineLearningPhase
from cyy_naive_pytorch_lib.model_executor import ModelExecutorCallbackPoint
from cyy_naive_pytorch_lib.model_util import ModelUtil
from cyy_naive_pytorch_lib.tensor import get_data_serialization_size
from cyy_naive_pytorch_lib.trainer import Trainer

from fed_quant_server import FedQuantServer
from worker import Worker


class FedQuantWorker(Worker):
    def __init__(self, trainer: Trainer, server: FedQuantServer, **kwargs):
        super().__init__(trainer, server, **kwargs)
        worker_round = kwargs.pop("round")
        self.qat = QuantizationAwareTraining(replace_layer=False)
        self.qat.append_to_model_executor(trainer)
        self.round = worker_round
        self.trainer.add_named_callback(
            ModelExecutorCallbackPoint.AFTER_EXECUTE,
            "quantization",
            self.__send_parameters,
        )
        model_util = ModelUtil(self.trainer.model)
        self.parameter_size = get_data_serialization_size(
            model_util.get_parameter_dict()
        )
        self.quantized_parameter_size = None

    def train(self, device):
        self.trainer.set_device(device)
        for _ in range(self.round):
            self.trainer.train()

    def __send_parameters(self, **kwargs):
        trainer = kwargs["model_executor"]
        parameter_dict = self.qat.get_quantized_parameters()

        if self.quantized_parameter_size is None:
            self.quantized_parameter_size = get_data_serialization_size(parameter_dict)

        get_logger().warning(
            "parameter_size is %s, quantized_parameter_size is %s, compression ratio is %s",
            self.parameter_size,
            self.quantized_parameter_size,
            float(self.quantized_parameter_size) / float(self.parameter_size),
        )

        get_logger().info("add_parameter_dict")
        self.server.add_parameter_dict(parameter_dict)
        get_logger().info("end add_parameter_dict")
        inferencer = trainer.get_inferencer(MachineLearningPhase.Test, copy_model=False)
        inferencer.set_device(get_cpu_device())
        loss, acc, _ = inferencer.inference()
        get_logger().warning("before aggregating loss is %s, acc is %s", loss, acc)

        parameter_dict = self.server.get_parameter_dict()
        self.qat.load_quantized_parameters(parameter_dict)

        if self.worker_id == 0:
            inferencer = trainer.get_inferencer(
                MachineLearningPhase.Test, copy_model=False
            )
            inferencer.set_device(self.trainer.trainer.device)
            loss, acc, _ = inferencer.inference()
            get_logger().warning("after aggregating loss is %s, acc is %s", loss, acc)
