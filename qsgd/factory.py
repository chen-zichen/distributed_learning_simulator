# from fed_quant_server import FedQuantServer
# from fed_quant_worker import FedQuantWorker
# from sign_sgd_server import SignSGDServer
# from sign_sgd_worker import SignSGDWorker
from qsgd_server import QsgdServer
from qsgd_worker import QsgdWorker


def get_server(algorithm, **kwargs):
    if algorithm == "sign_SGD":
        return SignSGDServer(**kwargs)
    if algorithm == "fed_quant":
        return FedQuantServer(**kwargs)
    if algorithm == "QSGD":
        return QsgdServer(**kwargs)
    raise RuntimeError("unknown algorithm:" + algorithm)


def get_worker(algorithm, **kwargs):
    if algorithm == "sign_SGD":
        return SignSGDWorker(**kwargs)
    if algorithm == "fed_quant":
        return FedQuantWorker(**kwargs)
    if algorithm == "QSGD":
        return QsgdWorker(**kwargs)
    raise RuntimeError("unknown algorithm:" + algorithm)
