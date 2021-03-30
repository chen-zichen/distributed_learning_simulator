from servers.fed_quant_server import FedQuantServer
from servers.multiround_shapley_value_server import \
    MultiRoundShapleyValueServer
from servers.sign_sgd_server import SignSGDServer
from workers.fed_quant_worker import FedQuantWorker
from workers.fed_worker import FedWorker
from workers.sign_sgd_worker import SignSGDWorker


def get_server(algorithm, **kwargs):
    if algorithm == "sign_SGD":
        return SignSGDServer(**kwargs)
    if algorithm == "fed_quant":
        return FedQuantServer(**kwargs)
    if algorithm == "multiround_shapley_value":
        return MultiRoundShapleyValueServer(**kwargs)
    raise RuntimeError("unknown algorithm:" + algorithm)


def get_worker(algorithm, **kwargs):
    if algorithm == "sign_SGD":
        return SignSGDWorker(**kwargs)
    if algorithm == "fed_quant":
        return FedQuantWorker(**kwargs)
    if algorithm == "multiround_shapley_value":
        return FedWorker(**kwargs)
    raise RuntimeError("unknown algorithm:" + algorithm)
