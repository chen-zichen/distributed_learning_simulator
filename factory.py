from fed_quant_server import FedQuantServer
from fed_quant_worker import FedQuantWorker
from sign_sgd_server import SignSGDServer
from sign_sgd_worker import SignSGDWorker


def get_server(algorithm, **kwargs):
    if algorithm == "sign_SGD":
        return SignSGDServer(**kwargs)
    if algorithm == "fed_quant":
        return FedQuantServer(**kwargs)
    raise RuntimeError("unknown algorithm:" + algorithm)


def get_worker(algorithm, **kwargs):
    if algorithm == "sign_SGD":
        return SignSGDWorker(**kwargs)
    if algorithm == "fed_quant":
        return FedQuantWorker(**kwargs)
    raise RuntimeError("unknown algorithm:" + algorithm)
