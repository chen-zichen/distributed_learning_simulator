from cyy_naive_lib.log import get_logger
from cyy_naive_pytorch_lib.tensor import (concat_dict_values,
                                          get_data_serialization_size,
                                          load_dict_values)

from fed_server import FedServer


class ShapleyValueServer(FedServer):
    def get_metric(self):

        pass
