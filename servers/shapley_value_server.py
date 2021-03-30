import copy
from itertools import chain, combinations

from cyy_naive_pytorch_lib.model_util import ModelUtil

from .fed_server import FedServer


class ShapleyValueServer(FedServer):
    def get_metric(self, model: dict = None):
        tester = self.tester
        if model is not None:
            tester = copy.deepcopy(tester)
            ModelUtil(tester.model).load_parameter_dict(model)
        tester.inference()
        return tester.loss_metric.get_loss(1)

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
                for k, v in avg_parameter.items():
                    avg_parameter[k] += parameter[k]
        for k, v in avg_parameter.items():
            avg_parameter[k] = v / len(client_subset)
        return avg_parameter

    def powerset(self, iterable):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
