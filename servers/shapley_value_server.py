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

    def powerset(self, iterable):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
