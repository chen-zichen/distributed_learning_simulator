from cyy_naive_lib.log import get_logger
from cyy_naive_pytorch_lib.algorithm.quantization.scheme import \
    stochastic_quantization
from cyy_naive_pytorch_lib.tensor import (concat_dict_values,
                                          get_data_serialization_size,
                                          load_dict_values)

from .fed_server import FedServer


class FedQuantServer(FedServer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.parameter = None

    def add_parameter_dict(self, worker_id, parameter: dict):
        self.parameter = parameter
        super().add_parameter_dict(worker_id, parameter)

    def get_parameter_dict(self):
        quantized_pair, dequant = super().get_parameter_dict()
        load_dict_values(self.parameter, dequant(quantized_pair))
        return self.parameter

    def _process_client_parameter(self, client_parameter: dict):
        for k, v in client_parameter.items():
            if isinstance(v, tuple):
                (weight, scale, zero_point) = v
                weight = weight.float()
                for idx, v in enumerate(weight):
                    weight[idx] = (v - zero_point[idx]) * scale[idx]
                client_parameter[k] = weight
        return client_parameter

    def _process_aggregated_parameter(self, aggregated_parameter: dict):
        get_logger().info("begin quantization")
        quantization_level = 256
        quant, dequant = stochastic_quantization(quantization_level)
        quantized_pair = quant(concat_dict_values(aggregated_parameter))

        parameter_size = get_data_serialization_size(aggregated_parameter)
        quantized_parameter_size = get_data_serialization_size(quantized_pair)
        get_logger().warning(
            "parameter_size is %s, quantized_parameter_size is %s, compression ratio is %s",
            parameter_size,
            quantized_parameter_size,
            float(quantized_parameter_size) / float(parameter_size),
        )

        get_logger().info("end quantization")
        return (quantized_pair, dequant)
